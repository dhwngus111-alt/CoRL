# HSP-Risky 연결기록 작성 규칙

이 폴더는 일일 업무 일지가 아니다.  
`/home/isl_jhoh/CoRL/transplant/일지`가 날짜별 작업 기록이라면, 이 폴더는 **HSP 알고리즘과 Risky 환경을 어떻게 연결했는지 전체 흐름을 계속 누적해서 정리하는 기준 문서 저장소**다.

## 1. 목적

이 폴더의 문서는 다음 질문에 답할 수 있어야 한다.

1. HSP 알고리즘을 Risky 환경 위에서 실행할 때 전체 데이터 흐름이 어떻게 이어지는가?
2. HSP repo에서 무엇을 재사용하고, 무엇을 사용하지 않는가?
3. Risky env에서 어떤 정보가 source of truth인가?
4. transplant 계층이 어떤 변환/호환/검증을 담당하는가?
5. 특정 구현이 왜 그렇게 결정되었고, 나중에 무엇을 조심해야 하는가?

## 2. 일지와의 차이

- `transplant/일지/YYYY-MM-DD.md`
  - 그날 실제로 진행한 작업, 디버깅, 수정, 검증을 기록한다.
  - 시간 순서와 작업 단위가 중요하다.

- `transplant/일지_HSP_risky_연결기록/*.md`
  - 날짜와 무관하게 유지되는 구조 문서를 기록한다.
  - 전체 흐름, 설계 결정, 연결 방식, 데이터 의미, 인터페이스가 중요하다.
  - 같은 주제에 대해 구현이 바뀌면 기존 문서를 갱신하거나 새 버전 문서로 이어서 쓴다.

## 3. 용어 기준

1. **HSP**
   - 기본적으로 HSP 논문/방법론의 학습 알고리즘 절차를 뜻한다.
   - policy pool 생성, biased policy 학습, diverse policy 선택, adaptive policy 학습 흐름을 포함한다.
   - HSP env를 뜻하지 않는다.

2. **HSP repo**
   - `/home/isl_jhoh/CoRL/HSP` 폴더 전체를 뜻한다.
   - 알고리즘 코드와 원본 Overcooked env가 함께 들어 있으므로, 문서에서는 필요한 경우 반드시 구분해서 쓴다.

3. **HSP algorithm code**
   - HSP repo 안에서 실제 학습을 구성하는 runner, trainer, policy, buffer, policy pool, population loop 등을 뜻한다.
   - Risky 이식에서 재사용하는 대상은 이 부분이다.

4. **HSP env / HSP Overcooked env**
   - `HSP/hsp/envs/overcooked*` 아래의 원본 환경 구현을 뜻한다.
   - Risky 실험의 state transition, event, shaped-info 기준으로 사용하지 않는다.

5. **Risky env**
   - `/home/isl_jhoh/CoRL/risky`의 Risky Overcooked 환경을 뜻한다.
   - 실제 MDP, state transition, reward, slip/risked/handoff/event 판정의 기준이다.

6. **transplant**
   - `/home/isl_jhoh/CoRL/transplant`의 이식 계층을 뜻한다.
   - Risky env를 HSP algorithm code가 호출할 수 있게 감싸고, 필요한 호환/로그/검증을 담당한다.

## 4. 작성해야 하는 내용

새 문서를 만들거나 기존 문서를 갱신할 때는 가능한 한 아래 항목을 포함한다.

1. **주제**
   - 이 문서가 설명하는 연결 흐름이나 설계 범위를 짧게 적는다.

2. **배경**
   - 왜 이 연결이 필요한지, 어떤 혼동이나 문제가 있었는지 적는다.

3. **전체 흐름**
   - 입력에서 출력까지의 흐름을 단계별로 쓴다.
   - 예: policy action → transplant adapter → Risky env step → event_infos → hidden utility vector → HSP reward/logging.

4. **사용하는 것과 사용하지 않는 것**
   - HSP repo에서 재사용하는 것.
   - HSP env에서 사용하지 않는 것.
   - Risky env에서 source of truth로 삼는 것.
   - transplant에서 새로 담당하는 것.

5. **인터페이스**
   - 함수, class, info dict key, vector shape, weight dimension, script argument처럼 서로 맞물리는 경계값을 적는다.
   - 특히 shape와 key order는 반드시 명시한다.

6. **event / feature / log 의미**
   - event 이름이 어디서 만들어지는지.
   - 어떤 기준으로 true/false가 되는지.
   - hidden utility나 log에서 어떤 이름으로 남는지.
   - 이름이 비슷하지만 다른 개념이면 반드시 분리해서 설명한다.

7. **결정 사항**
   - 왜 그 방향을 선택했는지 적는다. 이게 매우 중요하다. 간단하게 정리하면서도 이 부분은 강조하고 꼼꼼하게 정리한다.
   - 예 : Risky `EVENT_TYPES`를 기준으로 삼기로 했다면, HSP shaped-info 이름이 어떤 혼동이나 정보 손실을 만들었는지, 왜 Risky event 이름을 source of truth로 두는 것이 맞는지, 이 결정이 hidden utility와 로그/분석 기준에 어떤 영향을 주는지 함께 적는다.

8. **검증**
   - 어떤 명령으로 확인했는지.
   - 어떤 shape, 출력, 로그 키가 맞으면 성공인지 적는다.

9. **이전 버전과 달라진 점**
   - 이전 문서나 이전 구현과 비교해서 무엇이 바뀌었는지 간단하게만 적는다.
   - 자세한 작업 경과는 일지에 기록하므로, 여기서는 구조 이해에 필요한 핵심 차이만 남긴다.

10. **주의 사항 / 남은 문제**
   - 아직 위험한 부분, 나중에 바꿔야 할 부분, 건드리면 안 되는 파일을 적는다.

## 5. 파일 작성 방식

- 문서는 주제별로 만든다.
- 파일명은 되도록 다음 형식을 따른다.
  - `v0_0517.md`처럼 큰 흐름 버전 문서
  - `event_flow.md`
  - `hidden_utility_schema.md`
  - `runner_policy_pool_flow.md`
  - `troubleshooting_dtype.md`
- 특정 날짜의 작업 사실은 일지에 쓰고, 구조적으로 계속 남겨야 하는 설명은 이 폴더 문서에 반영한다.
- 같은 내용을 두 군데에 중복해서 길게 쓰지 않는다. 일지는 요약, 연결기록은 구조 설명을 담당한다.

## 6. 혼동 방지 규칙

1. "HSP"라고만 쓰면 HSP algorithm / HSP training pipeline을 뜻한다.
2. HSP 폴더 전체는 "HSP repo"라고 쓴다.
3. HSP 폴더 안의 원본 환경은 반드시 "HSP env" 또는 "HSP Overcooked env"라고 쓴다.
4. Risky 실험의 event/category/log 이름은 Risky `EVENT_TYPES`를 따른다.
5. HSP env의 shaped-info 이름을 Risky event 이름으로 사용하지 않는다.
6. HSP algorithm code를 재사용하더라도 HSP env를 사용한다는 뜻은 아니다.
7. 원본 Risky env와 원본 HSP env를 직접 수정하지 않는 것이 기본 원칙이다.
8. 변경이 필요하면 우선 transplant 계층에서 해결할 수 있는지 확인한다.

## 7. 문서 템플릿

```md
# 문서 제목

## 요약

이 문서가 설명하는 연결 흐름을 3~5줄로 요약한다.

## 배경

왜 이 흐름을 정리하는지, 어떤 혼동이나 문제가 있었는지 적는다.

## 전체 흐름

1. ...
2. ...
3. ...

## 사용하는 것 / 사용하지 않는 것

- HSP algorithm code에서 사용하는 것:
- HSP env에서 사용하지 않는 것:
- Risky env에서 기준으로 삼는 것:
- transplant에서 담당하는 것:

## 인터페이스와 데이터 구조

- 주요 함수/class:
- 주요 info dict key:
- vector shape:
- weight dimension:

## event / feature / log 기준

- source of truth:
- key order:
- 로그 이름:

## 결정 사항

- 결정:
- 이유:

## 검증

- 실행한 명령:
- 기대 결과:

## 이전 버전과 달라진 점

- ...

## 주의 사항

- ...
```
