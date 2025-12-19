# 🔍 자연어 기반 패널 검색 및 추출 시스템

# Introduction

## 1. 연구 목적
본 프로젝트는 기존 패널 검색 방식의 비효율성을 개선하기 위한 자연어 기반 검색 시스템 개발을 목표로 합니다. 기존의 체크박스 조합 방식은 복합적인 조건의 패널을 설정하기 어렵고, 사용자의 편의성이 낮았습니다.
이에 따라 자연어 처리 기술을 도입하여 사용자가 문장 형태로 검색 의도를 입력하면, 시스템이 이를 자동으로 해석해 해당 조건에 부합하는 패널 그룹을 추출하고 요약하도록 구현합니다. 이를 통해 사용자는 기존보다 직관적이고 신속하게 데이터를 분석하고, 즉각적인 비즈니스 인사이트를 도출할 수 있습니다.

## 2. 목표 및 내용
-   **질의 분류:** Sonnet 모델을 통해 사용자의 입력 문장을 분석하여 **정형 데이터 질의**인지 **비정형 데이터 질의**인지 판단합니다.
-   **데이터 검색:** 
    -   **정형 질의:** PostgreSQL에 SQL문을 생성하여 직접 조회합니다.
    -   **비정형 질의:** KoSimCSE 임베딩 모델을 이용해 문장을 벡터로 변환하고 `pgvector`를 통해 유사도 검색과 FTS(Full Text Search)를 이용한 키워드 검색을 수행합니다.
-   **패널 추출 및 분석:** 검색 결과를 바탕으로 최종 선정된 패널들을 추출합니다.
-   **결과 시각화 및 인사이트:** 선정된 패널들의 데이터를 사용자 인터페이스에 표시하고, 다양한 차트를 이용해 관련 결과를 시각적으로 제공합니다. 또한 **RAG 기술**을 활용하여 LLM이 해당 패널 그룹의 주요 특성, 숨겨진 특성, 특이사항 등을 자연어 형태의 분석 텍스트를 생성하여 비즈니스 인사이트 도출을 보조합니다.

## 3. 연구 결과
본 프로젝트는 기존 체크박스 기반의 패널 검색 방식이 지닌 비효율성과 낮은 활용성을 개선하기 위해, 자연어 질의 기반의 패널 검색 및 분석 시스템을 구축하는 것을 목표로 진행되었습니다. 이를 위해 설문 패널의 정형·비정형 데이터를 통합적으로 처리할 수 있는 데이터 파이프라인을 설계하고, 자연어 처리 모델을 통해 사용자의 질의를 분석한 뒤 최적의 패널 그룹을 추출하는 기능을 구현하였습니다.

-   **데이터 통합 및 처리:** 다양한 설문 데이터를 통합하여 재구조화하고, LLM을 활용해 각 패널의 성향 요약 텍스트를 생성하였습니다. 이후 **KoSimCSE** 임베딩 모델을 적용하여 768차원 벡터로 변환하고, FTS 검색을 위해 형태소 분석을 거친 뒤 TSVECTOR로 변환하여 **PostgreSQL + pgvector** 환경에 적재해 비정형 검색 성능을 확보했습니다.
-   **하이브리드 검색 구현:** 
    1.  **정형 데이터:** SQL 기반 필터링
    2.  **비정형 데이터:** 벡터 유사도 검색 + FTS(Full Text Search)
    두 경로를 병렬로 수행하는 하이브리드 검색 구조를 구현하였습니다. 검색된 두 결과는 **RRF(Reciprocal Rank Fusion)** 방식으로 결합하여 최종 패널을 선정하며, 이는 모호한 자연어 질의에서도 안정적인 검색 성능을 보장하게 합니다.
-   **프론트엔드 및 시각화:** 사용자가 자연어 입력을 통해 패널을 탐색하고 결과를 직관적으로 확인할 수 있도록 UI를 설계하였습니다. 질의 입력 시 연관 검색어 추천을 제공하고, 검색된 패널의 속성 정보·요약 텍스트·성향 분석 결과를 테이블과 다양한 차트의 형태로 시각화하였습니다. 또한 CSV 다운로드 기능을 제공하며 RAG 기술을 활용해 LLM을 통한 데이터 분석 어시스턴트를 구현하였습니다.
-   **데이터 분석 기능 구현:** 선정된 패널의 응답 데이터와 메타데이터를 기반으로 **Pandas**를 활용해 통계적 수치를 산출하고, 이를 시각화 및 LLM 인사이트 생성 파이프라인과 연동하여 입체적인 데이터 분석 기능을 구현하였습니다.

# Preview

 <table>
<tbody>
<tr>
<td align="center" colspan="2">
<b>초기 검색 페이지<br>
<img src="https://github.com/user-attachments/assets/2692cb6f-6519-4a03-86c7-748380a051cc" width="100%">
</td>
</tr>
<tr>
<td align="center" width="50%">
<b>패널 출력 페이지<br>
<img src="https://github.com/user-attachments/assets/80d60641-9fc4-4f33-8edf-68af63c5e483" width="100%">
</td>
<td align="center" width="50%">
<b>패널 상세 페이지<br>
<img src="https://github.com/user-attachments/assets/3662971e-a7fd-4f66-a2fb-8444b29b3fe4" width="100%">
</td>
</tr>
<tr>
<td align="center" width="50%">
<b>통계 시각화<br>
<img src="https://github.com/user-attachments/assets/ac4a5312-5533-4b5f-afed-9058ee3f6329" width="100%">
</td>
<td align="center" width="50%">
<b>분석 코멘트<br>
<img src="https://github.com/user-attachments/assets/7323cf89-fb29-4f1c-af4e-530b3af13b3b" width="100%">
</td>
</tr>
</tbody>
</table>

# Members
<table>
  <tbody>
    <tr>
      <td align="center"><a href="https://github.com/Codinggnewbe"><img src="https://avatars.githubusercontent.com/u/166682803?v=4" width="150px;" alt=""/><br /><sub><b> 김민수 </b></sub></a><br /></td>
       <td align="center"><a href="https://github.com/KimMinHyuk02"><img src="https://avatars.githubusercontent.com/u/185067106?v=4" width="150px;" alt=""/><br /><sub><b> 김민혁 </b></sub></a><br /></td>
       <td align="center"><a href="https://github.com/DoKyems"><img src="https://avatars.githubusercontent.com/u/85345952?v=4" width="150px;" alt=""/><br /><sub><b> 김도겸 </b></sub></a><br /></td>
       <td align="center"><a href="https://github.com/leesk0007"><img src="https://avatars.githubusercontent.com/u/174444051?v=4" width="150px;" alt=""/><br /><sub><b> 이성규 </b></sub></a><br /></td>
       <td align="center"><a href="https://github.com/Jihwan17"><img src="https://avatars.githubusercontent.com/u/232972776?v=4" width="150px;" alt=""/><br /><sub><b> 최지환 </b></sub></a><br /></td>
    </tr>
    <tr>
      <td align="center">Front / AI</td>
      <td align="center"> Data </td>
      <td align="center">Front / AI</td>
      <td align="center">Lead / Back</td>
      <td align="center"> Data </td>
    </tr>
  </tbody>
</table>

# Tech Stack
## Frontend
- **Framework:** React 18
- **Build Tool:** Vite
- **Language:** TypeScript
- **Styling:** Tailwind CSS
- **Visualization:** Recharts

## Backend
- **Framework:** FastAPI (Python 3.10+)
- **Server:** Uvicorn (ASGI)
- **Database:** PostgreSQL 15+
    - **Extension:** `pgvector` (벡터 검색), `ko_r3` (한국어 형태소 분석)
- **ORM:** SQLAlchemy (AsyncIO)
- **AI/LLM:** 
    - **Generative AI:** AWS Bedrock (Claude 3.5 Sonnet, Haiku)
    - **Embedding:** `BM-K/KoSimCSE-roberta` (768 dim, Sentence-Transformers)
    - **Orchestration:** LangChain
- **Analysis:** Pandas, NumPy

  
# Getting Started

### 저장소 클론 (서브모듈 포함)
```bash
git clone --recurse-submodules https://github.com/hansung-sw-capstone-2025-2/2025_8_D_BE.git
cd 2025_8_D_BE
```

### 가상환경 생성 및 활성화
```bash
python -m venv venv
source venv/bin/activate
# Windows의 경우: venv\Scripts\activate
```

### 의존성 설치
```bash
pip install -r backend/requirements.txt
```

### 환경 변수 설정
`.env` 파일을 `backend/` 디렉토리 내에 생성하고 다음 변수를 설정하세요.

```ini
# Database (RDS)
DATABASE_URL=postgresql+asyncpg://swcd_admin:비밀번호@swcd-db.cmvgu4w4gsw4.us-east-1.rds.amazonaws.com:5432/panel_db

# AWS Bedrock
AWS_BEARER_TOKEN_BEDROCK=여기에_토큰_입력
AWS_REGION=us-west-2

# ETL 설정
ETL_ENABLE_SUMMARY=false
```

### 서버 실행
```bash
# 프로젝트 루트에서 실행 시
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

# Project Structure

```bash
backend/
├── routers/                        # API 라우터 (엔드포인트)
│   ├── search_router.py            # 검색 API (/api/search/nl) - RRF 하이브리드 검색 엔드포인트
│   ├── panel_router.py             # 패널 API - 패널 목록/상세 조회
│   └── analysis_router.py          # 분석 API - RAG 기반 인사이트 생성 및 통계 분석
├── services/                       # 비즈니스 로직 계층
│   ├── search_service.py           # 검색 로직 (Structured + Unstructured + RRF 통합)
│   ├── analysis_service.py         # 분석 로직 (통계 산출 + LLM 프롬프트 엔지니어링)
│   ├── statistics_calculator.py    # 데이터 통계 계산 유틸리티
│   └── metadata_loader.py          # 메타데이터 로딩 및 캐싱
├── repositories/                   # 데이터 액세스 계층 (DB)
│   ├── panel_repository.py         # 정형 데이터 쿼리 (필터링, 매핑)
│   ├── document_repository.py      # 비정형 데이터 쿼리 (Vector Similarity, FTS)
│   ├── value_normalizer.py         # 검색어/필터 값 정규화 처리
│   └── database.py                 # DB 세션 및 연결 관리
├── db/                             # 데이터베이스 관리 (데이터 유출 방지를 위해 비공개)
├── data/                           # 정적 데이터 파일 (데이터 유출 방지를 위해 비공개)
├── scripts/                        # 유틸리티 및 ETL 스크립트
│   └── etl_pipeline.py             # 전체 데이터 ETL 파이프라인 (전처리, 임베딩 재생성 및 적재)
├── logs/                           # 런타임 로그 저장소
└── main.py                         # FastAPI 애플리케이션 진입점 (설정 로드, 미들웨어)
```


```bash
frontend/
├── src/
│   ├── components/                 # UI 컴포넌트
│   │   ├── ui/                     # 재사용 가능한 기본 UI 요소 (Button, Input 등 - Shadcn UI)
│   │   ├── DataVisualization.tsx   # 검색 결과 통계 시각화 (Charts)
│   │   ├── PanelDataTable.tsx      # 패널 목록 테이블 (Pagination, Sort)
│   │   ├── PanelDetailDialog.tsx   # 패널 상세 모달 (상세 정보, 요약)
│   │   ├── SearchBar.tsx           # 자연어 검색 입력창
│   │   └── DataAnalysis.tsx        # AI 분석 인사이트 표시 컴포넌트
│   ├── utils/                      # 유틸리티 함수
│   │   ├── api.ts                  # 백엔드 API 클라이언트 (Axios)
│   │   ├── panelSearchUtils.ts     # 검색 관련 헬퍼 함수
│   │   └── mockPanelData.ts        # 테스트용 Mock 데이터
│   ├── styles/                     # 전역 스타일
│   │   └── globals.css             # Tailwind CSS 디렉티브 및 글로벌 스타일
│   ├── App.tsx                     # 메인 애플리케이션 컴포넌트 (레이아웃, 상태 관리)
│   └── main.tsx                    # React 진입점 (DOM 렌더링)
├── public/                         # 정적 리소스 (아이콘, 매니페스트)
├── index.html                      # HTML 템플릿
├── vite.config.ts                  # Vite 빌드 설정
├── tsconfig.json                   # TypeScript 메인 설정
├── tsconfig.node.json              # TypeScript Node 관련 설정
├── package.json                    # 의존성 및 스크립트 설정
├── package-lock.json               # 의존성 버전 잠금 파일
└── .env.example                    # 환경변수 예시 파일
```

# API Endpoints

| Tag | Method | Endpoint | Description |
|---|---|---|---|
| **Search** | `POST` | `/api/search/nl` | 자연어 질의 패널 검색 (RRF 적용) |
| **Panels** | `GET` | `/api/panels/` | 전체 패널 목록 조회 (Pagination) |
| **Panels** | `GET` | `/api/panels/{id}` | 특정 패널 상세 정보 조회 |
| **Analysis** | `POST` | `/api/analysis/analyze` | 패널 그룹 심층 분석 및 인사이트 생성 |

# Key Features

-   **자연어 쿼리 파싱**: LLM(Claude 3.5 Sonnet)을 활용하여 사용자의 모호한 검색 의도를 정형 청크와 비정형 청크로 정밀하게 변환
-   **하이브리드 검색**: 키워드 매칭기반의 FTS 검색과 KoSimCSE 벡터 유사도 검색을 병렬 수행하고 RRF 알고리즘으로 결과 통합
-   **지능형 패널 프로파일링**: 응답 데이터를 분석하여 각 패널의 라이프스타일과 성향을 요약한 자연어 프로필 자동 생성
-   **RAG 기반 심층 분석**: 검색된 패널 데이터와 메타데이터를 컨텍스트로 활용하여 단순 통계 수치를 넘어선 비즈니스 인사이트와 행동 패턴 발견
-   **대화형 데이터 시각화**: 검색 결과의 인구통계학적 분포와 주요 특성을 직관적인 동적 차트(Bar, Pie, Line)로 실시간 시각화
  
# LLM Models

-   **Anthropic Claude sonnet 3.5**
-   **Anthropic Claude Haiku 4.0**


# License

본 프로젝트는 한성대학교 기업연계 SW캡스톤디자인 수업에서 진행되었습니다.
