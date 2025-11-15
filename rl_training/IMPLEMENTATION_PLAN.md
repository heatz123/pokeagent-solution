# Pokemon IL/RL Implementation Plan - Two Approaches

## Overview

PokemonEnv 기반 Imitation Learning 구현 - **두 가지 접근 방식 모두 구현**

### Approach 1: Offline Behavioral Cloning (`bc_pokemon.py`)
- Pre-collected expert demonstrations로 학습
- 장점: 빠름, 단순, 안정적
- 단점: Distribution shift 문제 가능성

### Approach 2: Online DAgger (`dagger_pokemon.py`)
- Student policy rollout + code_agent real-time query
- 장점: Distribution shift 해결, 더 강력한 성능
- 단점: code_agent 쿼리가 느림 (LLM latency)

## File Structure

```
rl_training/
├── bc_pokemon.py               # Offline BC implementation
├── dagger_pokemon.py           # Online DAgger with code_agent
├── common/                     # Shared utilities
│   ├── __init__.py
│   ├── networks.py            # ImpalaCNN, ResidualBlock
│   ├── buffer.py              # ReplayBuffer
│   └── utils.py               # Preprocessing, evaluation
├── expert_data/                # Pre-collected demos (for BC)
│   ├── episode_0001.npz
│   └── ...
├── models/                     # Model checkpoints
│   ├── bc_best_model.pth
│   ├── dagger_best_model.pth
│   └── ...
└── runs/                       # TensorBoard logs
    ├── bc_pokemon_*/
    └── dagger_pokemon_*/
```

## Implementation 1: Offline BC (`bc_pokemon.py`)

### Algorithm
```
1. Load pre-collected expert demonstrations from expert_data/
2. Train policy network with supervised learning (cross-entropy)
3. Periodically evaluate on environment
4. Save best model by eval return
```

### Training Loop
```python
for step in range(total_steps):
    # Sample batch from expert buffer
    batch = expert_buffer.sample(batch_size)
    obs, actions = batch['observations'], batch['actions']

    # Forward + Loss
    logits = policy_net(obs)
    loss = F.cross_entropy(logits, actions)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Eval periodically
    if step % eval_freq == 0:
        eval_return = evaluate(env, policy_net)
```

### Hyperparameters
```python
@dataclass
class BCArgs:
    exp_name: str = "bc_pokemon"
    expert_data_dir: str = "rl_training/expert_data"
    total_training_steps: int = 100_000
    batch_size: int = 64
    learning_rate: float = 1e-4
    eval_frequency: int = 5_000
    eval_episodes: int = 5
```

### Pros/Cons
**Pros:**
- Fast training (no environment interaction during training)
- Simple implementation
- Reproducible (fixed dataset)

**Cons:**
- Requires pre-collecting expert data
- Distribution shift: policy visits states not in expert data
- Limited by quality/coverage of expert demonstrations

---

## Implementation 2: Online DAgger (`dagger_pokemon.py`)

### Algorithm (DAgger)
```
1. Initialize policy π (student)
2. for iteration = 1, 2, ..., N:
    a. Collect trajectories using current policy π
    b. For each state s visited:
       - Query expert (code_agent) for action a*
       - Add (s, a*) to aggregated dataset D
    c. Train policy π on aggregated dataset D
    d. Evaluate and save checkpoint
```

### Key Difference from BC
- **BC**: Train only on pre-collected expert data
- **DAgger**: Train on data from student's own state distribution
  - Student explores → gets stuck/confused
  - Ask expert "what would you do here?" (code_agent query)
  - Add to dataset → retrain
  - Repeat: student improves → visits new states → query expert → ...

### Training Loop
```python
# Initialize
policy_net = ImpalaCNN(...)
buffer = ReplayBuffer(capacity=1_000_000)
code_agent = CodeAgent(...)  # Expert oracle

# DAgger iterations
for iteration in range(n_iterations):
    print(f"DAgger Iteration {iteration}")

    # Phase 1: Rollout with current policy
    for episode in range(episodes_per_iteration):
        obs, info = env.reset()
        done = False

        while not done:
            # Student action (current policy)
            with torch.no_grad():
                student_action = policy_net(obs).argmax()

            # Execute student action
            next_obs, reward, terminated, truncated, info = env.step(student_action)
            done = terminated or truncated

            # Query expert for optimal action at this state
            expert_action = code_agent.get_action(obs, info)

            # Add (obs, EXPERT_action) to buffer
            buffer.add(obs, expert_action, reward, next_obs, done)

            obs = next_obs

    # Phase 2: Train on aggregated dataset
    for train_step in range(train_steps_per_iteration):
        batch = buffer.sample(batch_size)
        obs, actions = batch['observations'], batch['actions']

        logits = policy_net(obs)
        loss = F.cross_entropy(logits, actions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Phase 3: Evaluate
    eval_return = evaluate(env, policy_net)
    print(f"Iteration {iteration}: Eval Return = {eval_return}")
```

### Hyperparameters
```python
@dataclass
class DaggerArgs:
    exp_name: str = "dagger_pokemon"

    # DAgger-specific
    n_iterations: int = 10
    episodes_per_iteration: int = 5  # Rollout episodes per iteration
    train_steps_per_iteration: int = 10_000  # Training steps per iteration

    # Training
    batch_size: int = 64
    learning_rate: float = 1e-4
    buffer_capacity: int = 1_000_000

    # Evaluation
    eval_frequency: int = 1  # Eval every iteration
    eval_episodes: int = 5

    # Code Agent (Expert)
    expert_model: str = "claude-3-5-sonnet-20241022"
    expert_timeout: int = 60  # LLM query timeout
```

### Code Agent Integration
```python
class ExpertOracle:
    """Wrapper for querying code_agent as expert"""

    def __init__(self, code_agent):
        self.code_agent = code_agent

    def get_action(self, obs, info) -> int:
        """
        Query code_agent for expert action at current state

        Args:
            obs: PIL Image (screenshot)
            info: Dict with state information

        Returns:
            action: int (0-9, GBA button index)
        """
        # Format state for LLM
        state_text = format_state_for_llm(info)

        # Query code agent
        response = self.code_agent.get_action(
            observation=obs,
            state=state_text,
            timeout=60
        )

        # Parse action from response
        action = self._parse_action(response)

        return action
```

### Pros/Cons
**Pros:**
- **No distribution shift**: Train on student's own state distribution
- Iteratively improves: student gets better → explores new states → queries expert
- Doesn't require pre-collecting data
- Can handle long-horizon tasks better

**Cons:**
- **Slow**: LLM queries add significant latency (~5-30s per action)
- Expensive: Many LLM calls (episodes × steps × iterations)
- More complex implementation
- Non-deterministic (LLM responses may vary)

### Optimizations for Speed
1. **Batched queries**: Collect multiple states, query expert in batch
2. **Async queries**: Use async LLM API calls
3. **Caching**: Cache expert responses for similar states
4. **Mixed training**: Start with BC (fast), then switch to DAgger

---

## Comparison Table

| Aspect | Offline BC | Online DAgger |
|--------|-----------|---------------|
| **Data Source** | Pre-collected demos | Real-time code_agent queries |
| **Training Speed** | Fast (no env interaction) | Slow (LLM latency) |
| **Distribution Shift** | Possible problem | Solved by design |
| **Expert Required** | Only for data collection | Every training step |
| **Implementation** | Simpler | More complex |
| **Cost** | Low (one-time data collection) | High (continuous LLM calls) |
| **Final Performance** | Good (if data covers states) | Better (adapts to policy) |

---

## Shared Components (`common/`)

### 1. Network Architecture (`networks.py`)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        residual = x
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x = x + residual
        x = F.relu(x)
        return x

class ConvSequence(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x

class ImpalaCNN(nn.Module):
    def __init__(self, action_dim=10, hidden_dim=256):
        super().__init__()

        # Three conv sequences (16 → 32 → 32 channels)
        self.conv_seq1 = ConvSequence(3, 16)
        self.conv_seq2 = ConvSequence(16, 32)
        self.conv_seq3 = ConvSequence(32, 32)

        # FC layers (compute size after convs)
        # 84x84 input → 11x11 after 3 maxpools
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(32 * 11 * 11, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        # Input: (B, 3, 84, 84) - preprocessed
        x = self.conv_seq1(x)
        x = self.conv_seq2(x)
        x = self.conv_seq3(x)
        x = self.fc(x)
        return x  # (B, action_dim)
```

### 2. Replay Buffer (`buffer.py`)
```python
import numpy as np
from typing import Dict

class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape=(84, 84, 3)):
        self.capacity = capacity
        self.observations = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = next_obs
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict:
        idxs = np.random.randint(0, self.size, size=batch_size)
        return {
            'observations': self.observations[idxs],
            'actions': self.actions[idxs],
            'rewards': self.rewards[idxs],
            'next_observations': self.next_observations[idxs],
            'dones': self.dones[idxs]
        }

    def __len__(self):
        return self.size
```

### 3. Utils (`utils.py`)
```python
import torch
import numpy as np
from PIL import Image

def preprocess_observation(obs):
    """
    Preprocess observation for network

    Args:
        obs: PIL Image (160x240x3) or numpy array

    Returns:
        Tensor (3, 84, 84) float32 [0, 1]
    """
    if isinstance(obs, Image.Image):
        obs = np.array(obs)

    # Resize to 84x84
    obs = Image.fromarray(obs).resize((84, 84))
    obs = np.array(obs)

    # Transpose to (C, H, W) and normalize
    obs = obs.transpose(2, 0, 1)  # (H, W, C) → (C, H, W)
    obs = obs.astype(np.float32) / 255.0

    return obs

def evaluate(env, policy_net, n_episodes, device):
    """Evaluate policy on environment"""
    policy_net.eval()
    returns = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        episode_return = 0
        done = False

        while not done:
            obs_tensor = torch.FloatTensor(preprocess_observation(obs)).unsqueeze(0).to(device)
            with torch.no_grad():
                action = policy_net(obs_tensor).argmax(dim=-1).item()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_return += reward

        returns.append(episode_return)

    policy_net.train()
    return np.mean(returns), np.std(returns)
```

---

## Implementation Plan

### Phase 1: Shared Components (Week 1)
1. ✅ Create folder structure
2. ⏳ Implement `common/networks.py` (ImpalaCNN)
3. ⏳ Implement `common/buffer.py` (ReplayBuffer)
4. ⏳ Implement `common/utils.py` (preprocessing, evaluation)
5. ⏳ Test components independently

### Phase 2: Offline BC (Week 1-2)
1. ⏳ Implement `bc_pokemon.py`
2. ⏳ Collect expert demonstrations (10-50 episodes)
3. ⏳ Train BC model
4. ⏳ Evaluate and baseline performance

### Phase 3: Online DAgger (Week 2-3)
1. ⏳ Implement `dagger_pokemon.py`
2. ⏳ Integrate code_agent as expert oracle
3. ⏳ Run DAgger training (3-10 iterations)
4. ⏳ Compare with BC baseline

### Phase 4: Analysis & Optimization (Week 3-4)
1. ⏳ Compare BC vs DAgger performance
2. ⏳ Analyze failure cases
3. ⏳ Optimize DAgger (caching, async queries)
4. ⏳ Hyperparameter tuning

---

## Expected Results

### Offline BC
- Training: ~2-4 hours (100K steps)
- Final performance: 60-80% of expert (if good data coverage)
- Failure mode: Gets stuck in unseen states

### Online DAgger
- Training: ~1-2 days (10 iterations × 5 episodes × ~1000 steps, with LLM latency)
- Final performance: 80-95% of expert (adapts to policy distribution)
- Failure mode: Expensive, slow

### Best Strategy
1. Start with BC for quick baseline
2. If BC performance plateaus → switch to DAgger
3. Or: Hybrid approach (pre-train BC → finetune with DAgger)

---

## Questions to Answer

1. **BC vs DAgger performance gap**: How much better is DAgger?
2. **Data efficiency**: How many expert demos needed for BC? How many DAgger iterations?
3. **LLM latency impact**: Can we cache/batch queries to speed up DAgger?
4. **Hybrid approach**: Does BC→DAgger work better than pure DAgger?
5. **Observation preprocessing**: 84x84 vs 160x240? Single frame vs stacked?
