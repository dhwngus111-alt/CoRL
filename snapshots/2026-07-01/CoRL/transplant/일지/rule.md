"일지" 폴더에는 매일 진행한 수정사항, 발전사항, 트러블슈팅 등의 업무 진행을 기록한다.
기록하는 주체는 설계/분석을 담당하는 'antigravity 에이전트'와 실제 코드 구현/디버깅을 담당하는 'codex 에이전트' 둘이다.

[기본 규칙]
1. 날짜별로 하나의 마크다운(`.md`) 파일을 생성하여 작성한다. (예: 2026-05-16.md)
2. 두 에이전트는 하나의 파일 안에서 헤딩(##)을 사용하여 각자의 영역에 나누어 작성한다.
3. 각 에이전트는 자신의 역할에 맞는 필수 기재 사항을 반드시 포함해야 한다.

[우리 사이의 용어 정리]
1. HSP :
   - 기본 의미는 **HSP 논문/방법론의 학습 알고리즘 과정 전체**를 뜻한다.
   - 즉 policy pool을 만들고, 다양한 편향 정책/파트너 정책을 학습하고, adaptive policy를 학습하는 절차를 말한다.
   - 우리가 "HSP를 Risky에 적용한다"고 말할 때는 **HSP 환경을 사용한다는 뜻이 아니라, HSP 알고리즘 절차를 Risky 환경 위에서 실행한다는 뜻**이다.

2. HSP repo :
   - `/home/isl_jhoh/CoRL/HSP` 폴더 전체를 뜻한다.
   - 이 안에는 알고리즘 코드, runner, trainer, policy, buffer, population 관리 코드뿐 아니라 원본 Overcooked env도 같이 들어 있다.
   - 따라서 "HSP repo의 파일을 사용한다"는 말은 곧바로 "HSP env를 사용한다"는 뜻이 아니다.

3. HSP algorithm code :
   - HSP repo 안에서 실제 학습 알고리즘을 구성하는 코드만 뜻한다.
   - 예: runner, trainer, policy, replay buffer, policy pool, population training loop, checkpoint loading/saving 등.
   - Risky 이식에서는 이 부분만 최소한으로 재사용한다.

4. HSP env / HSP Overcooked env :
   - HSP repo 안의 원본 환경 구현을 뜻한다.
   - 예: `HSP/hsp/envs/overcooked*` 아래의 Overcooked env, MDP, shaped_info key, event 처리 로직.
   - Risky 실험에서는 이 환경의 dynamics, event 이름, shaped_info 의미 체계를 source of truth로 사용하지 않는다.
   - 특히 `pickup_onion_from_O`, `PLACEMENT_IN_POT`, `USEFUL_DISH_PICKUP` 같은 HSP env의 shaped-info 이름을 Risky event에 억지로 매핑하지 않는다.

5. risky :
   - `/home/isl_jhoh/CoRL/risky` 폴더의 Risky Overcooked 환경을 뜻한다.
   - Risky 실험에서 실제 state transition, reward, slip, risked, handoff, event 판정은 이 환경이 담당한다.
   - Risky의 `EVENT_TYPES`가 hidden utility feature와 로그/분석 이름의 기준이다.

6. transplant :
   - `/home/isl_jhoh/CoRL/transplant` 폴더의 이식 계층을 뜻한다.
   - Risky env를 HSP algorithm code가 호출할 수 있는 API 형태로 감싸되, event 의미 체계는 Risky를 따른다.
   - 원칙적으로 원본 HSP env와 원본 risky env는 직접 수정하지 않고, 필요한 연결/호환/검증 코드는 transplant 쪽에 둔다.

7. old_env :
   - HSP repo 안에 들어 있던 원본 Overcooked 환경을 가리킬 때만 사용한다.
   - risky와 비교하기 위한 과거 기준 환경이라는 의미이며, 현재 Risky 실험의 source of truth가 아니다.



[혼동 방지 규칙]
1. "HSP"라고만 말하면 기본적으로 **HSP algorithm / HSP training pipeline**을 뜻한다.
2. HSP 폴더의 환경을 말해야 할 때는 반드시 **HSP env** 또는 **HSP Overcooked env**라고 쓴다.
3. HSP 폴더 전체를 말해야 할 때는 반드시 **HSP repo**라고 쓴다.
4. Risky 실험의 event/category/log 이름은 Risky `EVENT_TYPES`를 따른다.
5. HSP env의 shaped-info 이름을 Risky event의 기준 이름으로 사용하지 않는다.
6. HSP algorithm code를 재사용하더라도, 그것이 HSP env를 사용한다는 뜻은 아니다.


[필수 기재 사항]
- 어떤 과정을 진행했는가 (작업 내용 요약)
- 왜 그 과정이 필요한가 (작업의 당위성 및 목적)
- 기존 환경인 HSP에서는 어떤 것을 사용/참조/수정했는가 (원본 호환성 관점)
- 기존 환경인 risky에서는 어떤 것을 사용/참조/수정했는가 (신규 환경 적용 관점)

[파일 템플릿 예시]
# 📝 YYYY-MM-DD 일일 업무 일지

> **[🤖 에이전트 역할 분담 가이드]**
> - **Antigravity (수석 아키텍트):** 압도적인 컨텍스트 창(Context Window)을 활용하여 전체 저장소를 훑고, 모듈 간의 의존성을 파악하며 구현 계획과 아키텍처를 설계한다.
> - **Codex (수석 개발자):** 설계된 아키텍처를 바탕으로 뛰어난 코딩/디버깅 능력을 발휘하여 실제 로직을 빠르고 정확하게 구현한다.

## 🤖 1. antigravity (설계 및 분석)
### 🔹 어떤 과정을 진행했는가
(아키텍처 분석, 이벤트 리스트업, 구현 계획 수립 등)
### 🔹 왜 그 과정이 필요한가
(문제 정의, 설계 의도 등)
### 🔹 기존 환경인 HSP에서는 어떤 것을 사용하거나 수정했는가
(참조한 HSP 로직, 유지한 제약사항 등)
### 🔹 기존 환경인 risky에서는 어떤 것을 사용했는가
(참조한 Risky 로직, 어댑터 설계 등)

---

## 💻 2. codex (구현 및 검증)
### 🔹 어떤 과정을 진행했는가
(코드 작성, 디버깅, 테스트 등)
### 🔹 왜 그 과정이 필요한가
(코드 레벨에서의 최적화, 에러 해결 등)
### 🔹 기존 환경인 HSP에서는 어떤 것을 사용하거나 수정했는가
(실제로 수정한 파일/코드 등)
### 🔹 기존 환경인 risky에서는 어떤 것을 사용했는가
(실제로 수정한 파일/코드 등)
