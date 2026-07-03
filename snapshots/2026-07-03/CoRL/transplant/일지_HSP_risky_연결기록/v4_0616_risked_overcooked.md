# v4 2026-06-16 Risked Overcooked subgoal hidden utility 연결 구조

## 요약

이번 버전은 새 `/home/isl_jhoh/CoRL/risked_overcooked` 환경의 subgoal 의미를 HSP hidden utility에 어떻게 연결하는지 정리한다.

핵심은 hidden utility key 이름은 기존 호환성을 위해 `subgoal_activated`로 유지하지만, 의미는 새 env 기준인 "active puddle을 실제로 끈 useful G press"로 고정한다는 점이다. 이미 puddle이 비활성화된 상태에서 반복해서 G를 눌러도 env shaping reward와 HSP hidden utility credit은 들어가지 않는다.

## 배경

old `/home/isl_jhoh/CoRL/risky` 기준에서는 `subgoal_activated`가 raw event에 있었고, G interaction 자체를 event로 해석하기 쉬웠다.

새 `/home/isl_jhoh/CoRL/risked_overcooked` 환경에서는 `EVENT_TYPES`에 `subgoal_activated`가 없다. 대신 subgoal 보상은 env transition 내부에서 다음 기준으로 들어간다.

```text
G terrain에서 INTERACT
-> 연결된 puddle 중 active 상태(timer <= 0)가 있음
-> puddle disable timer 설정
-> SUBGOAL_PRESS_REW 지급
```

따라서 HSP hidden utility도 단순 G press가 아니라 useful press만 보상/선호축으로 봐야 한다.

## 전체 흐름

1. HSP algorithm code가 action index를 낸다.
2. `RiskyOvercooked.step(...)`이 action index를 새 env의 `Action`으로 변환한다.
3. adapter가 base env step 직전에 `prev_state`와 `water_disable_timers`를 읽는다.
4. `_subgoal_activated_agents(prev_state, joint_action)`가 agent별 useful G press 여부를 계산한다.
5. 새 risked env가 실제 transition과 reward를 계산한다.
6. adapter가 `mdp_info["event_infos"]`에 compatibility event `subgoal_activated`를 주입한다.
7. `risky_events_to_category_info(...)`가 event dict를 hidden utility category dict로 변환한다.
8. `shaped_info_to_array(...)`가 `(2, len(HIDDEN_UTILITY_KEYS))` hidden utility vector를 만든다.
9. HSP hidden reward는 이 vector와 `w0/w1`을 dot product해서 계산된다.

## 사용하는 것 / 사용하지 않는 것

- HSP algorithm code에서 사용하는 것:
  - 기존 runner/trainer/policy 흐름
  - hidden utility vector와 `w0/w1` weight 구조
  - hidden utility key 이름 `subgoal_activated`

- HSP env에서 사용하지 않는 것:
  - HSP Overcooked env transition
  - HSP env의 subgoal event/shaping 구현

- 새 risked env에서 source of truth로 삼는 것:
  - 실제 MDP transition
  - `water_disable_timers`
  - `subgoal_to_water`
  - `subgoal_disable_steps`
  - `SUBGOAL_PRESS_REW`

- transplant에서 담당하는 것:
  - raw event에는 없는 `subgoal_activated` compatibility event 생성
  - useful activation 기준을 hidden utility vector에 반영
  - future raw event schema 검증

## 인터페이스와 데이터 구조

주요 파일:

```text
transplant/adapters/risky_overcooked_env.py
transplant/hsp_hidden_utility.py
transplant/tests/test_risky_pickup_mapping.py
```

핵심 함수:

```text
RiskyOvercooked._subgoal_activated_agents(prev_state, joint_action)
RiskyOvercooked._apply_subgoal_activated_event(event_infos, activated)
risky_events_to_category_info(event_infos)
shaped_info_to_array(shaped_info_by_agent)
```

주요 info dict key:

```text
info["step_shaped_info_by_agent"]
info["vec_shaped_info_by_agent"]
info["risky_event_infos"]
info["risky_event_counts"]
```

hidden utility key:

```text
subgoal_activated
```

현재 vector 기준:

```text
len(HIDDEN_UTILITY_KEYS) = 30
len(w0) = len(w1) = 31
vec_shaped_info_by_agent shape = (2, 30)
```

마지막 weight 1개는 sparse reward weight다.

## event / feature / log 기준

old risky 기준:

```text
G interact raw event -> subgoal_activated
```

new risked 기준:

```text
G interact + active linked puddle 있음 -> SUBGOAL_PRESS_REW
G interact + linked puddle이 이미 disabled 상태 -> no subgoal reward
```

transplant 기준:

```text
G interact + active linked puddle 있음 -> compatibility subgoal_activated=True
G interact + linked puddle이 이미 disabled 상태 -> compatibility subgoal_activated=False
```

중요한 점은 hidden utility key 이름만 old 호환을 유지하고, event의 의미는 new risked semantics를 따른다는 것이다.

## 결정 사항

### 결정 1: hidden utility key 이름은 유지한다

`subgoal_activated`는 기존 HSP selection/logging 흐름과 weight vector에서 이미 쓰이는 이름이므로 유지한다.

이름을 바꾸면 script, policy config, 과거 로그 분석 코드가 함께 흔들린다. 대신 문서와 테스트에서 이 key의 의미가 "useful activation only"임을 명확히 고정한다.

### 결정 2: raw env source는 수정하지 않는다

새 risked env의 `EVENT_TYPES`에는 `subgoal_activated`가 없지만, 원본 env를 수정하지 않는다.

이유:

- env transition과 reward logic은 새 repo를 source of truth로 유지해야 한다.
- HSP 연결 호환성은 transplant 계층의 책임이다.
- future env update와 비교하기 쉬워진다.

### 결정 3: `SUBGOAL_PRESS_REW`는 hidden utility 축으로 추가하지 않는다

`SUBGOAL_PRESS_REW`는 env reward shaping scalar이고, hidden utility는 행동/event 선호축이다.

따라서 `SUBGOAL_PRESS_REW` 자체를 새 hidden utility dimension으로 만들지 않고, useful press 여부만 `subgoal_activated` event axis에 기록한다. scalar 값은 config/W&B metadata와 env reward로 관리한다.

### 결정 4: 반복 G press는 hidden utility credit을 받지 않는다

이미 puddle disable timer가 남아 있으면 G를 다시 눌러도 active puddle을 새로 연 것이 아니다.

따라서 이 반복 action에는 다음이 모두 들어가지 않아야 한다.

```text
SUBGOAL_PRESS_REW
subgoal_activated hidden utility credit
```

## 검증

targeted unittest:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/isl_jhoh/miniconda3/envs/corl/bin/python \
  -m unittest transplant.tests.test_risky_pickup_mapping
```

확인하는 subgoal 시나리오:

```text
첫 G + INTERACT
-> active puddle 비활성화
-> subgoal_activated=1
-> SUBGOAL_PRESS_REW=2.0

반복 G + INTERACT
-> puddle disable timer가 남아 있음
-> subgoal_activated=0
-> shaped reward=0.0
```

## 이전 버전과 달라진 점

v3는 pickup source mapping을 정리했다. 특히 raw pickup event만으로는 counter/dispenser/pot source를 구분할 수 없어서 adapter가 `prev_state -> next_state` transition context를 이용하도록 정리했다.

v4는 새 risked env migration 이후 subgoal hidden utility semantics를 정리한다. old raw `subgoal_activated`를 그대로 기대하지 않고, 새 env의 useful press 기준에 맞춰 transplant adapter가 compatibility event를 만든다는 점이 핵심 차이다.

## 주의 사항 / 남은 문제

- `subgoal_activated`는 새 risked env raw event가 아니라 transplant compatibility event다.
- future risked env에 raw `subgoal_activated`가 추가되면 double count가 생기지 않는지 확인해야 한다.
- future `EVENT_TYPES`가 추가되면 `hsp_hidden_utility.py` schema 검증이 실패하도록 되어 있으므로, hidden utility 포함 여부를 다시 판단해야 한다.
- 원본 `/home/isl_jhoh/CoRL/risked_overcooked` env는 수정하지 않는 것이 기본 원칙이다.
