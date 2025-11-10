# Meta-Agent Knowledge Curation 전략

## 현재 문제점
- 287개 지식 중 177개가 ROUTE_102 관련 (62%)
- (12,18) 좌표에 대해 75개 중복 언급
- UPDATE는 1개만 (0.3%), 나머지는 계속 ADD만
- 상충되는 지식들 존재 ("down works" vs "down blocked")
- Agent가 5시간 넘게 같은 위치에서 무한 루프

## 제안하는 Meta-Agent 전략

### 🎯 **Option 1: Knowledge Consolidation Agent (추천 ⭐)**
**목표**: 중복/유사한 지식을 통합하여 명확한 단일 지침으로 변환

**작동 방식**:
```python
# 1. 같은 좌표/위치에 대한 지식 그룹핑
# 2. LLM으로 유사 지식 묶기
# 3. 통합된 1-2개 지침으로 변환
# 4. 기존 중복 지식 삭제

예시:
Before (75개):
- "At (12,18), down goes to (12,19)"
- "From (12,18), pressing down moves to (12,19)"
- "In Oldale at (12,18), the south corridor is clear"
- ... 72개 더

After (2개):
- "Oldale Town navigation: (12,18)↔(12,19) oscillation confirmed. South from (12,19) is BLOCKED by hedge. Must find alternative route."
- "Route 102 exit: NOT south from center. Try: west side of town or different entrance."
```

**장점**:
- 노이즈 대폭 감소 (287개 → ~30-50개)
- Agent가 이해하기 쉬운 명확한 지침
- 토큰 사용량 감소

**구현 난이도**: 중간


### 🔍 **Option 2: Pattern Detection + Meta-Strategy Injection (추천 ⭐⭐)**
**목표**: 반복 패턴 감지 후 높은 수준의 전략 제안

**작동 방식**:
```python
# 1. 최근 100 step 분석
# 2. 반복 패턴 감지 (같은 위치, 같은 실패)
# 3. 근본 원인 분석
# 4. 새로운 메타 전략 주입

예시:
Detected Pattern:
- 2500+ steps at Oldale Town
- 177 knowledge entries for same area
- 69% backtrack ratio
- Stuck in (12,18)↔(12,19) loop

Meta-Strategy Injection:
ADD_KNOWLEDGE: "🚨 CRITICAL: Oldale Town center is a TRAP. The southern path from (12,19) is permanently blocked. Route 102 entrance is on the WEST side of town at approximately (3-5, 14-16). Navigate around the west side, not through center."

ADD_KNOWLEDGE: "🎯 ROUTE_102 Strategy: From current position, try: 1) Move LEFT (west) repeatedly to reach x<=5, 2) Then move down to find western exit, 3) Look for 'Route 102' sign on west edge."
```

**장점**:
- 단순 통합이 아닌 **새로운 인사이트** 제공
- Agent에게 "큰 그림" 전략 제시
- 막힌 상황 돌파 가능

**구현 난이도**: 중간-높음


### 🗑️ **Option 3: Smart Pruning (기본)**
**목표**: 오래되고 중복되고 틀린 지식 제거

**작동 방식**:
```python
# 1. 중복 ID 제거
# 2. 60 step 이상 오래된 + 최근 재확인 안된 지식 삭제
# 3. 모순되는 지식 중 최신 것만 유지

기준:
- 동일 내용 중복 → 최신 것 1개만
- 60+ steps 지난 지식 → 삭제
- 모순 감지 → 최신/더 구체적인 것 유지
```

**장점**:
- 구현 간단
- 즉시 효과

**단점**:
- 근본적 문제 해결 안됨


### 🎨 **Option 4: Contradiction Resolution (보조)**
**목표**: 상충되는 지식 해결

**작동 방식**:
```python
# LLM으로 모순 감지
# 예: "down is walkable" vs "down is blocked"
# → 더 구체적이고 최근 것 유지
# → 또는 두 개를 통합
```


## 🏆 **최종 추천: Hybrid Approach**

### Phase 1 (즉시): Pruning + Consolidation
```
1. 중복 ID 제거
2. 같은 위치 관련 지식 통합 (75개 → 2-3개)
3. 60+ steps 오래된 지식 삭제
```

### Phase 2 (10-20 step마다): Pattern Detection
```
1. 반복 패턴 감지
2. 막혀있으면 메타 전략 주입
3. 새로운 접근법 제안
```

### Phase 3 (지속적): Smart Monitoring
```
1. Knowledge growth rate 모니터링
2. 급증하면 (>20/100 steps) 다시 consolidation
3. Backtrack ratio 70% 넘으면 emergency intervention
```

---

## 구현 예시 코드 구조

```python
class MetaAgent:
    def __init__(self):
        self.kb = KnowledgeBase()
        self.log_file = "submission.log"

    def run_cycle(self):
        # 1. 현재 상태 분석
        stats = self.analyze_current_state()

        # 2. 조건부 실행
        if stats['knowledge_growth_rate'] > 20:  # 최근 100 step에 20개 이상
            self.consolidate_knowledge()

        if stats['backtrack_ratio'] > 0.7:  # 70% 이상 역추적
            self.inject_meta_strategy()

        if stats['same_position_count'] > 50:  # 50 step 이상 같은 위치
            self.emergency_intervention()

    def consolidate_knowledge(self):
        # LLM 사용해서 유사 지식 통합
        pass

    def inject_meta_strategy(self):
        # 패턴 분석 후 새 전략 주입
        pass
```

---

## 실행 주기 권장사항

- **Consolidation**: 100 steps마다 또는 knowledge > 200개
- **Pattern Detection**: 매 20 steps
- **Emergency Intervention**: Backtrack ratio > 70% 감지 즉시
