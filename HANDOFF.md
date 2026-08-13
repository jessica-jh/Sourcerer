# Reference Finder — 세션 핸드오프 (2026-07-31 기준)

새 세션 시작할 때 이 파일을 먼저 읽어주세요.

## 프로젝트 상태

`reference_finder.py` CLI 도구는 기능적으로 완성됨 (S2/OpenAlex/arXiv 검색 + keyword/embedding/claude 스코어링 + APA/BibTeX 출력). `.env`에 `OPENAI_API_KEY`, `SEMANTIC_SCHOLAR_API_KEY`, `OPENALEX_API_KEY` 세팅 완료.

## 이번 세션에서 고친 중요 버그 2개

1. **arXiv 리다이렉트 버그** ([refsearch/pipeline.py](refsearch/pipeline.py)): httpx 클라이언트가 `follow_redirects=True` 없이 만들어져서, arXiv API가 301을 반환할 때 빈 결과로 처리되고 있었음. 이 세션 초반의 모든 arXiv 관련 결과/eval 수치는 이 버그가 있던 상태라 **신뢰하면 안 됨**. 지금은 고쳐짐.
2. **토큰화 정규식 버그** ([refsearch/query_gen.py](refsearch/query_gen.py), [refsearch/scoring/keyword.py](refsearch/scoring/keyword.py)): `T2I`, `GPT2` 같은 숫자 포함 용어가 `T I`처럼 깨지던 문제. 정규식에 `0-9` 추가해서 고침.
3. **S2 rate limit 자체 유발**: [refsearch/sources/semantic_scholar.py](refsearch/sources/semantic_scholar.py)에 arXiv처럼 최소 1초 간격 쓰로틀 추가함.

## Eval 인프라

- [eval/build_eval_set.py](eval/build_eval_set.py): FullTextPeerRead(FTPR) 기반, `eval/cs_ai_eval.jsonl` (100건)
- [eval/build_eval_set_citeme.py](eval/build_eval_set_citeme.py): CiteME(사람이 직접 만든 벤치마크, 더 깨끗함) 기반, `eval/cs_ai_eval_citeme.jsonl` (119건)
- [eval/run_eval.py](eval/run_eval.py): `python -m eval.run_eval --eval-set <path> --method <keyword|embedding|claude> --venue-preset <none|cs_ai> --limit N`

### 버그 수정 후 재확인한 수치 (n=30, venue-preset none, Recall@5=HR@5)
| 데이터셋 | embedding | claude(gpt-4o-mini) |
|---|---|---|
| FTPR | 10% (3/30) | 10% (3/30) |
| CiteME | 0% (0/30) | 3.3% (1/30) |

두 데이터셋 다 낮고, claude가 embedding보다 뚜렷하게 낫지도 않음. CiteME가 유독 낮은 건 버그 때문이라기보다, 벤치마크 자체가 인용 마커 자리의 고유명사(논문 제목 등)를 문장에서 지워버려서 원래 어렵게 설계된 것으로 보임 (CiteGuard 논문도 사람 정확도 69%, 최고 모델 45~68%밖에 안 됨).

### 해결된 이슈: candidate id hallucination (2026-07-30)

FTPR/claude를 여러 번 돌렸을 때 6.7~10% 사이로 수치가 흔들렸던 원인을 찾아서 고침. 두 단계로 조사:

1. **1차 완화**: [refsearch/scoring/claude.py](refsearch/scoring/claude.py)에 `MAX_CANDIDATES=30` 추가, claude judge 호출 전 embedding score 기준 상위 30개로 후보를 줄이도록 [refsearch/pipeline.py](refsearch/pipeline.py) 수정. 후보가 80~150개까지 불어나던 문제(재검색 라운드마다 `papers = papers + new_papers`로 계속 누적되는데 줄이는 로직이 없었음)는 줄었지만, `unknown candidate id` 경고는 완전히 사라지지 않음.
2. **근본 원인**: candidate id가 `s2:003`, `arxiv:012`, `openalex:016`처럼 **소스 접두어 + 인덱스** 조합이었는데, 인덱스는 유효 범위인데 접두어만 틀린 hallucination이 계속 나왔음 — gpt-4o-mini가 프롬프트에 찍힌 id를 그대로 베끼지 않고, 소스별로 자기 나름대로 다시 번호를 매기는 것으로 추정. **id 체계를 소스 접두어 없는 단순 정수(`0`, `1`, `2`, ...)로 교체**하니 n=5, n=30 재실행 모두에서 `unknown candidate id` 경고가 0건으로 완전히 사라짐.

S2 `HTTP 429`는 별개 이슈로, 재시도(지수 백오프, [refsearch/sources/retry.py](refsearch/sources/retry.py))로 대부분 커버되지만 같은 날 30건 eval을 반복 실행하면 일시적 쿼터 소진으로 몇 시간까지 느려질 수 있음(그날 하루 지나거나 시간 지나면 자연 해소되는 걸로 보임) — 코드 버그는 아니고 그냥 그날 반복 실행을 줄이는 수밖에 없어 보임.

### 핵심 진단 결과 (2026-07-31): 10% recall의 진짜 원인은 검색(retrieval), 랭킹이 아님

[eval/diagnose_recall.py](eval/diagnose_recall.py)를 새로 작성해서 FTPR n=30에 대해 "골드 논문이 최종 후보 풀 어디까지 도달했는지" 랭크를 추적함 (claude.py의 LLM judge는 스킵, 검색+dedupe+venue필터+embedding 랭킹까지만 재현). 결과:

| 단계 | 건수 |
|---|---|
| 후보 풀에 애초에 없었음 (검색 실패) | 27/30 (90%) |
| 풀에는 있었는데 top-30 숏리스트 밖 | 0/30 |
| top-30엔 들었는데 top-5 밖 | 3/30 (rank 6/74, 12/70, 13/87) |
| top-5 안에 있었음 (embedding만으로) | 0/30 |

**결론: recall 10%가 낮은 건 압도적으로 검색 단계에서 골드 논문을 못 찾아오는 문제 (90%), 스코어링/랭킹 문제가 아님.** cross-encoder 재랭킹을 붙여도 영향받는 건 최대 3건(그나마 rank 12·13은 재랭킹으로 top-5까지 끌어올리기 어려움 — 현실적으로 rank 6 1건 정도만 개선 가능해서 Recall@5 10%→13% 수준)이라 **우선순위가 아님**.

### query_gen 진단 및 1차 수정 (2026-07-31)

실패 케이스 27개 **전부**(6개 + 21개, 두 배치로 확인) 골드 논문 제목으로 S2 직접 검색하면 1등으로 정확히 나옴 — 즉 API/인덱스 한계가 전혀 아니고 100% 우리 쿼리 생성(`refsearch/query_gen.py`)이 원인임을 확인.

원인: `sentence` 필드가 실제로는 "한 문장"이 아니라 FTPR의 `masked_text` — 평균 63단어(50~101 단어)짜리 문단([eval/build_eval_set.py:62](eval/build_eval_set.py#L62)). `base_queries()`가 이 전체 문단에서 불용어만 뺀 30+ 단어를 통째로 쿼리로 던지거나(`clean_query`), "가장 긴 단어 8개"라는 부정확한 휴리스틱(`keyphrase_query`)을 씀. 실측으로 S2 검색 API는 **쿼리 단어 수에 매우 민감** — 9단어 쿼리는 결과 0건, 같은 주제를 3~5단어로 줄이면 20건 반환되는 걸 직접 확인함 (AND에 가까운 매칭으로 추정).

**수정**: `use_hyde`/`use_llm_query`를 `method`와 독립적인 축으로 분리 (`RunConfig`에 `bool | None` 필드 추가, `None`=auto로 method가 claude/all일 때만 기본 켜짐, 명시적으로 True/False 지정 가능 — CLI에 `--use-hyde`/`--no-use-hyde`, `--use-llm-query`/`--no-use-llm-query` 추가, [reference_finder.py](reference_finder.py), [eval/run_eval.py](eval/run_eval.py)). [refsearch/scoring/claude.py](refsearch/scoring/claude.py)에 `extract_search_query()` 신규 추가 — gpt-4o-mini로 긴 문단에서 3~5개 핵심어만 뽑아 쿼리로 씀 (처음엔 5~8개로 시작했다가 위 AND-매칭 관찰 후 3~5개로 더 줄임).

**결과 (n=30, 전체 파이프라인 재검색 라운드 포함)**: Recall@5 = 10% (3/30) — **숫자상 baseline과 동일**, 다만 hit 구성은 바뀜(이전 5·10·30번 → 지금 5·10·26번). [eval/diagnose_recall.py](eval/diagnose_recall.py) 단독 진단으로는 후보 풀 진입/랭크가 개선되는 게 보였지만(예: rank 12→1, 13→7), 그 개선이 top-5 진입까지는 이어지지 못함 — 여러 케이스가 이제 풀에는 들어오지만 rank 7~21 사이에 걸려있음.

**의미**: 병목이 "검색 자체가 안 됨"에서 "후보 풀엔 들어왔는데 순위가 낮음"으로 옮겨감 — **cross-encoder 재랭킹이 다시 의미 있어질 수 있는 시점**. 처음엔(90% retrieval miss 상태) cross-encoder 효과가 최대 1건이라 우선순위 아니라고 판단했었는데, 이제 재확인 필요.

**주의**: LLM 쿼리 추출은 temperature 기본값이라 매 실행마다 살짝 다른 쿼리가 나옴(non-deterministic) — n=10 진단에서 같은 코드로 두 번 돌렸을 때도 결과가 갈렸음. 작은 표본으로 개선 여부를 판단할 때 이 변동성을 감안할 것. 필요하면 `temperature=0` 고정도 고려.

### cross-encoder 재랭킹 적용 (2026-07-31) — 효과 있었음

[refsearch/scoring/rerank.py](refsearch/scoring/rerank.py) 신규 추가: `cross-encoder/ms-marco-MiniLM-L-6-v2` (로컬, sentence-transformers에 이미 포함, 추가 의존성 없음)로 bi-encoder 상위 `RERANK_CANDIDATES=30`개를 재채점. `RunConfig.use_rerank: bool = False`로 독립 플래그화 (`--use-rerank`, [reference_finder.py](reference_finder.py) / [eval/run_eval.py](eval/run_eval.py) / [eval/diagnose_recall.py](eval/diagnose_recall.py) 전부 지원). [refsearch/pipeline.py](refsearch/pipeline.py)에서 `embedding` 방식 결과와 claude 숏리스트 둘 다 이 재랭킹된 리스트를 공유하도록 정리.

**결과 (n=30, `--use-rerank` 켜고 전체 파이프라인)**: Recall@5 = HR@5 = **13.3% (4/30)**, no-rerank baseline 10%(3/30) 대비 개선. hit이 5·10·26·30번으로 (기존 3개 유지 + item 30 "Playing Atari" 추가 hit). 사전에 예측했던 "rank 6위 1건 정도 개선, 10%→13% 수준"과 정확히 일치. 429는 12건 나왔지만 재시도로 전부 정상 처리됨.

### CiteME 재평가 결과 (2026-07-31) — 개선 없음

같은 수정사항(LLM 쿼리 추출 + cross-encoder rerank)으로 CiteME n=30도 재평가함. 첫 시도는 인터넷이 잠깐 끊겨서 6~29번이 전부 `ERROR: Connection error`로 실패 → 재실행함(이런 경우 결과가 0%처럼 보여도 무효이니 misses 개수나 ERROR 라인 유무를 꼭 확인할 것). 재실행 결과 깨끗하게 완료됨(connection error 0건, 429는 8건이지만 재시도로 처리됨):

**Recall@5 = HR@5 = 3.3% (1/30) — baseline(3.3%, 1/30)과 동일, 개선 없음.**

FTPR에서는 먹힌 개선이 CiteME에서는 안 먹힘. CiteME는 애초에 인용 마커 자리의 고유명사(논문 제목 등)를 문장에서 지워버리는 설계라([HANDOFF.md](HANDOFF.md) 위쪽 참고), 쿼리를 아무리 잘 뽑고 재랭킹을 잘해도 "애초에 문맥에 핵심 단서가 없는" 근본적 어려움은 못 건드리는 것으로 보임. FTPR과 CiteME은 서로 다른 종류의 병목을 가진 벤치마크로 접근해야 할 듯.

### 참고: SOTA 대비 위치 (2026-07-31, 웹 검색으로 확인)

- FullTextPeerRead SOTA(BERT-GCN 계열): Recall@5 ≈ **48.6%** — 우리 13.3%는 한참 못 미침. 다만 SOTA는 보통 고정 후보 집합에서 재랭킹하는 세팅이라 직접 비교는 조심스러움.
- CiteME: 인간 69.7%, SPECTER2(순수 검색) 0%, 프론티어 LM(검색 없이 지식만) 4.2~18.5%, **CiteAgent**(GPT-4o가 직접 검색+논문 읽으며 반복 탐색) **35.3%**, CiteGuard 65.4%.
- **시사점**: CiteAgent처럼 "한 번 검색 후 재랭킹"이 아니라 **여러 번 읽고 재검색하는 에이전틱 루프**가 핵심 격차로 보임. 다음 확장 방향 후보:
  1. 에이전틱 반복 탐색 — 상위 후보를 실제로 더 깊이 읽고 검증, 재검색을 고정 2라운드가 아니라 LLM이 스스로 판단할 때까지 반복
  2. 인용 그래프 활용 — 느슨하게라도 관련된 논문을 찾으면 그 논문의 reference list를 S2 API로 가져와 제목 매칭 (지금 전혀 안 쓰는 신호, FTPR SOTA가 그래프 기반인 이유이기도 함). §14(멀티턴 액션)와 겹칠 수 있어 스펙 범위 확인 필요.

### 연구(논문) 트랙 탐색 — 2026-07-31, 보류 결정

사용자가 ICLR 2027(paper 마감 9/25, abstract 9/18 — 오늘부터 약 8주) + 취업 포트폴리오를 동시에 노리는 연구 방향을 탐색함. 최종적으로 **연구 트랙은 보류하고 제품(도구) 완성도를 우선하기로 결정** — 아래는 그 과정과, 나중에 재개할 경우를 위한 요약.

**아이디어**: 자동 인용 귀속(citation attribution) 시스템의 실패를, 과학사회학의 "obliteration by incorporation"(Merton) — 유명해진 발견일수록 정식 인용 없이 언급되는 현상 — 으로 설명. Bornmann/Meng/Varol/Barabási(2024, PNAS Nexus, "Hidden citations obscure true impact in science")가 이 현상을 ML로 계산적 검증한 바 있음. 이 사회학적 현상과, CiteME/CiteAgent/CiteGuard 같은 자동 인용 찾기 AI 계열이 지금까지 연결된 적 없다는 게 출발점.

**Novelty 검증 (2회 검토 + 사용자가 가져온 외부 리뷰 1건)**:
- 키워드 검색 + Bornmann 논문 forward-citation snowballing(17건) + CiteME/CiteGuard related work 직접 확인 + "implicit citation detection"(SciCite/Jurgens/Abu-Jbara — 다른 문제를 다룸, 각주가 이미 있는 경우의 문맥 추출) 분야 구분 확인 → 연결 자체는 novelty 있어 보인다는 결론
- 외부(사용자 제공) 리뷰가 핵심 결함 지적: **CiteME 개별 항목을 "obliteration 사례"라고 부르면 안 됨** — OBI는 원래 topic/corpus 수준 구성개념인데 CiteME 항목은 원문에 인용이 있었다가 지워진 것. 측정 대상은 "그 항목이 가리키는 개념 주변 담화의 편입 정도"여야 함. 이 재정의가 맞음.
- 서브에이전트 리뷰 2회로 스코프 축소: 원래 RQ3(corpus-mining으로 원 출처 다수결 추정)는 순환논증 결함(다수결이 진짜 원저작물이 아니라 가장 많이 인용되는 2차 서베이/재서술 논문으로 수렴할 위험 — CiteGuard 사이드 조사에서 발견한 재인용 착각 문제와 동일한 함정) 지적받아 future work로 미룸. 최종 스코프: RQ1(원인 검증, citation count 아닌 discourse-기반 지표) + RQ2(FTPR/CiteME 벤치마크 간 일관성, **특히 CiteME 내부 분산으로 검증해야 함** — "CiteME가 원래 더 어렵다"는 당연한 결과와 구분하기 위해) + RQ3(축소판: 기존 저비용/고비용 기법 사이 라우팅 실증, 새 방법론 발명 아님).

**Gut-check 실행 결과 (2026-07-31)**:
- S2 citation-context 기반 정밀 지표(EAR: 정식 인용된 문장 중 제목 핵심어를 실제로 풀어 쓴 비율)를 만들려 했으나, 이날 S2를 너무 많이 써서 재시도(최대 126초 백오프)로도 못 뚫는 수준으로 rate limit 걸림 → OpenAlex 기반 거친 지표(citation count + 발표연도)로 대체
- 결과 (FTPR+CiteME 합쳐 n=60, hit=5/miss=52): HITS 평균 citation count 1,694 vs MISS 평균 3,212 — **방향은 가설과 일치하지만 hit 표본이 5개뿐이라 근거 약함**. FTPR 단독으로 봤던 이전 파일럿(n=16, 598 vs 7,800, 13배 차이)보다 효과 크기가 훨씬 작아짐 — CiteME(거의 다 miss)를 섞으면서 신호가 희석된 것으로 보임. 연도는 차이 거의 없음(2016 vs 2015).
- **결론**: 약한 초록불. 표본을 더 늘리려면 우리 파이프라인을 더 돌려야 하는데(S2/OpenAI 의존, 오늘 하루 종일 겪은 그 병목) 비용이 크고, 8주 타임라인과 "포트폴리오가 우선" 순위를 고려해 **지금은 투자하지 않기로 결정**.

**나중에 재개하려면**: 이 섹션 + `/private/tmp/.../scratchpad/research_proposal.md`(v2, 두 번의 서브에이전트 리뷰 포함) + `/private/tmp/.../scratchpad/obi_gutcheck*.py`(S2/OpenAlex 두 버전) 참고. scratchpad는 세션 종료 시 사라질 수 있으니 재개할 마음이 있으면 이 파일들을 프로젝트 안으로 옮겨두는 게 좋음.

### 다음에 볼 것 (제품 트랙 우선)

- **지금부터 우선순위**: reference_finder를 포트폴리오용 제품으로 다듬기 (문서화, 데모/UI, 에러 처리 정리 등 — 아직 구체적 계획 없음, 다음 세션에서 논의)
- LLM 쿼리 추출 `temperature=0`으로 고정해서 run-to-run 변동성 제거 후 다시 비교
- 소스별 `limit`(현재 20)을 늘리거나 쿼리 개수를 늘려서 후보 풀 자체를 넓히는 게 추가로 도움되는지
- 표본을 30건보다 늘려서 13.3%가 통계적으로 유의미한 개선인지 확인 (n=30 표본에서 1건 차이라 노이즈일 가능성도 있음)
- (연구 트랙 재개 시) 위 "에이전틱 반복 탐색" / "인용 그래프 활용" 아이디어, 그리고 obliteration-by-incorporation 방향

## 별도로 진행했던 사이드 조사 (CiteGuard 관련) — 완료, 참고용

외부 리포 `github.com/KathCYM/CiteGuard`를 스크래치패드(`/private/tmp/.../scratchpad/CiteGuard`)에 클론해서, "본문/스니펫 검색이 재인용(다른 논문을 인용한 문장)을 원저작물로 착각하는 실패모드"를 실증 확인함:
- 원시 스니펫 검색 단독 테스트(n=16): 대부분이 재인용 패턴
- CiteGuard 실제 최종 선택(PDF 제외, n=25): 오답 19건 중 14건(74%)이 "discusses/relates to" 같은 귀속 아닌 연관어로 정당화됨
- 사용자가 이걸 별도 연구/논문 주제로 발전시킬지는 미정 (진행하려면 완전히 새 프로젝트로 분리 권장 — 아이디어: 근거 소스를 초록/스니펫/섹션필터/LLM요약으로 나눠 비교 + S2 citation-context로 자동 라벨링)

## 아직 안 한 것

- business/psych_soc 도메인 eval (ISR 논문 PDF 받으면 GROBID로 처리 예정, 보류 중)
- CS/AI eval 표본을 30건보다 늘려서 더 안정적인 수치 확보
- §14: 멀티턴 액션(search_text_snippet 등)을 우리 도구에 실제로 이식하는 건 보류 상태 (스펙 범위 밖이라고 판단, §9에 향후 확장 아이디어로만 기록 권장)

## 참고 문서
- [reference_finder_dev_spec.md](.) (사용자가 PDF로 전달, 로컬엔 없음 — 필요하면 사용자에게 재요청)
- [reference_finder_eval_spec.md](.) (마찬가지로 PDF 전달분)
