# Pokemon Imitation Learning Implementation Plan

## Overview
PokemonEnv 기반의 single-file imitation learning 구현 (cleanrl 스타일)
- **Algorithm**: Offline Behavioral Cloning (BC) / DQN-BC hybrid
- **Base Environment**: PokemonEnv (Gymnasium-style)
- **Framework**: PyTorch (cleanrl 대부분의 구현이 PyTorch 기반)
- **Reference**: cleanrl/qdagger_dqn_atari_jax_impalacnn.py (simplified)

**Simple Approach (v1)**:
- Expert 데이터 수집 → Offline 학습 (Behavioral Cloning)
- Online finetuning/DAgger는 나중에 확장 (code_agent를 실시간 쿼리 가능)

## Key Differences from Atari

| Aspect | Atari | Pokemon |
|--------|-------|---------|
| **Observation** | 84x84x4 grayscale stacked | 160x240x3 RGB single frame |
| **Action Space** | 18 (Atari buttons) | 10 (GBA buttons) |
| **Episode Length** | ~1000-5000 steps | ~10000 steps (longer) |
| **Reward Signal** | Game score | Milestone-based sparse rewards |
| **State Info** | Observation only | Rich state dict (player, map, milestones) |
| **Teacher** | Pre-trained DQN from HF | Expert demonstrations (human/LLM) |

## File Structure

```
rl_training/
├── bc_pokemon.py               # Main single-file BC implementation (cleanrl style)
├── collect_expert_data.py      # Script to collect expert demonstrations
├── expert_data/                # Expert demonstrations
│   ├── episode_0001.npz       # Saved trajectories
│   ├── episode_0002.npz
│   └── ...
├── models/                     # Saved model checkpoints
│   ├── checkpoint_*.pth
│   ├── best_model.pth
│   └── final_model.pth
└── runs/                       # TensorBoard logs
    └── bc_pokemon_YYYYMMDD_HHMMSS/
```

## Implementation Components

### 1. Network Architecture

**ImpalaCNN for Pokemon (Modified)**
```python
Input: 160x240x3 RGB image
│
├─ Preprocessing: Resize to 84x84, Normalize [0,1]
│
├─ ConvSequence 1: 16 channels
│   ├─ Conv2d(3→16, 3x3)
│   ├─ MaxPool2d(3x3, stride=2)
│   └─ ResidualBlock × 2
│
├─ ConvSequence 2: 32 channels
│   ├─ Conv2d(16→32, 3x3)
│   ├─ MaxPool2d(3x3, stride=2)
│   └─ ResidualBlock × 2
│
├─ ConvSequence 3: 32 channels
│   ├─ Conv2d(32→32, 3x3)
│   ├─ MaxPool2d(3x3, stride=2)
│   └─ ResidualBlock × 2
│
├─ Flatten
├─ ReLU
├─ Linear(→256)
├─ ReLU
└─ Linear(256→10)  # 10 actions

Output: Q-values for 10 actions
```

**ResidualBlock**
```python
Conv2d(ch→ch, 3x3) → ReLU → Conv2d(ch→ch, 3x3) → Add input → ReLU
```

### 2. Expert Data Format

**Trajectory Storage (NPZ format)**
```python
{
    'observations': np.array,  # (T, 160, 240, 3) uint8
    'actions': np.array,       # (T,) int32
    'rewards': np.array,       # (T,) float32
    'dones': np.array,         # (T,) bool
    'infos': List[dict],       # Episode metadata
    'episode_return': float,   # Total episode reward
    'episode_length': int      # Episode steps
}
```

**Expert Data Collection Script**
- Option 1: Record human play sessions
- Option 2: Record LLM agent episodes (from existing code_agent.py)
- Option 3: Manual demonstration tool with keyboard control

### 3. Replay Buffer

**PrioritizedReplayBuffer (Optional) or SimpleReplayBuffer**
```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.observations = np.zeros((capacity, 160, 240, 3), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, 160, 240, 3), dtype=np.uint8)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        ...

    def sample(self, batch_size):
        ...
```

### 4. Training Algorithm (Simplified Offline IL)

**Phase 1: Load Expert Data**
```python
# Load pre-collected expert demonstrations
expert_buffer = load_expert_data(expert_data_dir)
# Returns list of (obs, action) pairs from all episodes

print(f"Loaded {len(expert_buffer)} expert transitions")
```

**Phase 2: Offline Behavioral Cloning**
```python
for step in range(training_steps):  # e.g., 100,000 - 500,000 steps
    # Sample batch from expert data
    batch = expert_buffer.sample(batch_size)
    obs, actions = batch['observations'], batch['actions']

    # Forward pass
    logits = policy_network(obs)  # or q_network(obs)

    # Compute imitation loss (cross-entropy)
    loss = F.cross_entropy(logits, actions)

    # Update network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Logging
    if step % log_frequency == 0:
        writer.add_scalar("train/bc_loss", loss.item(), step)

    # Evaluation
    if step % eval_frequency == 0:
        eval_return = evaluate(env, policy_network, n_episodes=5)
        writer.add_scalar("eval/episodic_return", eval_return, step)
```

**Future Work: Online DAgger with Code Agent**
```python
# Phase 3 (나중에 구현):
# - Student policy로 rollout
# - 각 state에서 code_agent 쿼리해서 expert action 받기
# - New data로 buffer 업데이트하며 계속 학습
# - Adaptive mixing coefficient로 점진적 독립

# Pseudocode:
# for step in range(online_steps):
#     obs = env.step(student_action)
#     expert_action = code_agent.get_action(obs, info)  # Query LLM
#     buffer.add(obs, expert_action)  # DAgger aggregation
#     train_on_mixed_buffer(expert_buffer + new_buffer)
```

### 5. Loss Function (Behavioral Cloning)

**Simple Cross-Entropy Loss**
```python
def compute_bc_loss(policy_net, batch):
    """
    Behavioral Cloning: Supervised learning on (obs, expert_action) pairs
    """
    obs = batch['observations']  # (B, 160, 240, 3) or preprocessed
    actions = batch['actions']   # (B,) int64

    # Forward pass
    logits = policy_net(obs)  # (B, 10) action logits

    # Classification loss
    loss = F.cross_entropy(logits, actions)

    # Optional: Accuracy metric
    with torch.no_grad():
        pred_actions = logits.argmax(dim=-1)
        accuracy = (pred_actions == actions).float().mean()

    return loss, accuracy
```

**Optional: DQN-BC Hybrid (if using Q-values)**
```python
def compute_hybrid_loss(q_net, target_net, batch, bc_weight=1.0):
    """
    Combines Q-learning loss + BC loss
    - Useful if you have reward signals and want RL + IL
    """
    obs, actions, rewards, next_obs, dones = batch

    # BC loss (imitation)
    logits = q_net(obs)
    bc_loss = F.cross_entropy(logits, actions)

    # Q-learning loss (optional, if rewards available)
    q_values = logits.gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q_values = target_net(next_obs).max(1)[0]
        target_q = rewards + 0.99 * next_q_values * (1 - dones)
    td_loss = F.mse_loss(q_values, target_q)

    # Combined
    total_loss = bc_weight * bc_loss + td_loss
    return total_loss, bc_loss, td_loss
```

### 6. Training Hyperparameters (Simplified for BC)

```python
@dataclass
class Args:
    # Experiment
    exp_name: str = "bc_pokemon"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False  # Set to True for W&B logging
    wandb_project_name: str = "pokemon-il"

    # Environment
    rom_path: str = "Emerald-GBAdvance/rom.gba"

    # Training
    expert_data_dir: str = "rl_training/expert_data"
    total_training_steps: int = 100_000  # BC는 보통 적은 step으로 충분
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5  # L2 regularization

    # Network
    hidden_dim: int = 256  # FC layer size after CNN

    # Evaluation
    eval_frequency: int = 5_000
    eval_episodes: int = 5

    # Checkpointing
    save_frequency: int = 10_000
    model_dir: str = "rl_training/models"
    save_best: bool = True  # Save best model by eval return
```

### 7. Training Loop Structure (Simplified BC)

```python
def train():
    # 1. Setup
    args = tyro.cli(Args)

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.torch_deterministic:
        torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    # 2. Load expert data
    print(f"Loading expert data from {args.expert_data_dir}...")
    expert_buffer = load_expert_data(args.expert_data_dir)
    print(f"Loaded {len(expert_buffer)} expert transitions")

    # 3. Create environment for evaluation
    eval_env = PokemonEnv(
        rom_path=args.rom_path,
        headless=True,
        base_fps=240,  # Fast FPS for evaluation
    )

    # 4. Create policy network
    policy_net = ImpalaCNN(action_dim=10, hidden_dim=args.hidden_dim).to(device)
    optimizer = optim.Adam(
        policy_net.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # 5. TensorBoard logging
    writer = SummaryWriter(f"runs/{args.exp_name}_{int(time.time())}")

    # 6. Training loop
    print("Starting Behavioral Cloning training...")
    best_eval_return = -float('inf')

    for step in range(args.total_training_steps):
        # Sample batch from expert buffer
        batch = expert_buffer.sample(args.batch_size)
        obs = torch.FloatTensor(batch['observations']).to(device)
        actions = torch.LongTensor(batch['actions']).to(device)

        # Compute BC loss
        logits = policy_net(obs)
        loss = F.cross_entropy(logits, actions)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        if step % 100 == 0:
            with torch.no_grad():
                accuracy = (logits.argmax(dim=-1) == actions).float().mean()
            writer.add_scalar("train/bc_loss", loss.item(), step)
            writer.add_scalar("train/accuracy", accuracy.item(), step)
            print(f"Step {step}: Loss={loss.item():.4f}, Acc={accuracy.item():.2%}")

        # Evaluation
        if step % args.eval_frequency == 0 and step > 0:
            eval_return = evaluate(eval_env, policy_net, args.eval_episodes, device)
            writer.add_scalar("eval/episodic_return", eval_return, step)
            print(f"Eval at step {step}: Return={eval_return:.2f}")

            # Save best model
            if args.save_best and eval_return > best_eval_return:
                best_eval_return = eval_return
                save_path = os.path.join(args.model_dir, "best_model.pth")
                torch.save({
                    'step': step,
                    'model_state_dict': policy_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'eval_return': eval_return,
                }, save_path)
                print(f"Saved best model: {save_path} (return={eval_return:.2f})")

        # Periodic checkpoint
        if step % args.save_frequency == 0 and step > 0:
            save_path = os.path.join(args.model_dir, f"checkpoint_{step}.pth")
            torch.save({
                'step': step,
                'model_state_dict': policy_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, save_path)
            print(f"Saved checkpoint: {save_path}")

    # Final save
    final_path = os.path.join(args.model_dir, "final_model.pth")
    torch.save(policy_net.state_dict(), final_path)
    print(f"Training complete! Final model saved to {final_path}")

    eval_env.close()
    writer.close()


def evaluate(env, policy_net, n_episodes, device):
    """Evaluate policy on environment for n episodes"""
    policy_net.eval()
    total_returns = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        episode_return = 0
        done = False

        while not done:
            # Preprocess observation and get action
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = policy_net(obs_tensor)
                action = logits.argmax(dim=-1).item()

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_return += reward

        total_returns.append(episode_return)

    policy_net.train()
    return np.mean(total_returns)


if __name__ == "__main__":
    train()
```

## Expert Data Collection

### Option 1: Record from Existing Code Agent

```python
# collect_expert_data.py
from pokemon_env.gym_env import PokemonEnv
from agent.code_agent import CodeAgent

env = PokemonEnv(rom_path="Emerald-GBAdvance/rom.gba", record_video=True)
agent = CodeAgent(...)

for episode in range(num_episodes):
    obs, info = env.reset()
    episode_data = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'dones': []
    }

    done = False
    while not done:
        # Get action from code agent
        action = agent.get_action(obs, info)

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Store
        episode_data['observations'].append(obs)
        episode_data['actions'].append(action)
        episode_data['rewards'].append(reward)
        episode_data['dones'].append(done)

        obs = next_obs

    # Save episode
    np.savez_compressed(
        f"rl_training/expert_data/episode_{episode:04d}.npz",
        **episode_data,
        episode_return=sum(episode_data['rewards']),
        episode_length=len(episode_data['rewards'])
    )
```

### Option 2: Human Demonstrations via Keyboard

```python
# collect_human_demos.py
import pygame
from pokemon_env.gym_env import PokemonEnv

env = PokemonEnv(rom_path="Emerald-GBAdvance/rom.gba", headless=False)

# Map keyboard to actions
KEY_TO_ACTION = {
    pygame.K_z: 0,      # A
    pygame.K_x: 1,      # B
    pygame.K_RETURN: 2, # START
    pygame.K_RSHIFT: 3, # SELECT
    pygame.K_UP: 4,
    pygame.K_DOWN: 5,
    pygame.K_LEFT: 6,
    pygame.K_RIGHT: 7,
    pygame.K_a: 8,      # L
    pygame.K_s: 9       # R
}

# Record episodes with keyboard control
...
```

## Evaluation Metrics

1. **Average Episodic Return**: Mean reward over N evaluation episodes
2. **Milestone Completion Rate**: % of key milestones reached
3. **Episode Length**: Average steps per episode
4. **Success Rate**: % of episodes reaching terminal goal
5. **Q-value Statistics**: Mean/std of predicted Q-values
6. **Distillation Coefficient**: Tracks student progress relative to teacher
7. **Loss Curves**: TD loss, distillation loss, total loss

## Next Steps (Simple BC v1)

1. ✅ Create folder structure
2. ⏳ Implement `bc_pokemon.py` single-file BC script (cleanrl style)
3. ⏳ Implement expert data collection script (`collect_expert_data.py`)
4. ⏳ Collect initial expert demonstrations (10-50 episodes from code_agent or human)
5. ⏳ Run offline BC training
6. ⏳ Evaluate trained policy
7. ⏳ Iterate on hyperparameters and network architecture

## Future Extensions

- **Online DAgger**: Student rollout → code_agent query → aggregated buffer training
- **Hybrid DQN-BC**: Add RL loss with reward signals for better generalization
- **Multimodal**: Augment vision with text state embeddings
- **Curriculum Learning**: Train on progressively harder milestones

## Questions / Decisions Needed

1. **Teacher Policy Source**:
   - Use existing LLM code_agent.py? (likely best quality but slow)
   - Collect human demonstrations? (faster but may need many episodes)
   - Pre-train a DQN first then use as teacher? (most similar to cleanrl)

2. **Observation Preprocessing**:
   - Keep 160x240x3 or resize to 84x84x3? (84x84 is standard for Atari)
   - Single frame or stack 4 frames? (Pokemon is slower-paced, single frame OK)

3. **Training Time**:
   - PokemonEnv steps are slower than Atari (60 frames per step)
   - May need to reduce total_timesteps or increase FPS

4. **Milestone-based Rewards**:
   - Current reward is sparse (milestone completion)
   - Consider adding dense rewards? (distance to goal, exploration bonus)

5. **State Information Usage**:
   - Currently only using RGB observation
   - Could augment with text state embedding? (multimodal approach)
