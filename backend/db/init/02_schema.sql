-- ==========================================
-- SWCD Panel Database Schema
-- ==========================================

-- 패널 기본 정보 테이블 (정형 데이터 + LLM 생성 요약)
CREATE TABLE IF NOT EXISTS panels (
    panel_id VARCHAR(100) PRIMARY KEY,
    
    -- 기본 인구통계 정보
    gender VARCHAR(20),
    age INTEGER,
    region_city VARCHAR(100),      -- 지역(시)
    region_gu VARCHAR(100),        -- 지역(구)
    marital_status VARCHAR(50),    -- 결혼 여부
    children_count INTEGER,         -- 자녀수
    family_size INTEGER,           -- 가족수
    education_level VARCHAR(100),  -- 최종학력
    occupation VARCHAR(100),       -- 직업
    monthly_personal_income INTEGER,   -- 월평균 개인소득
    monthly_household_income INTEGER,  -- 월평균 가구소득
    
    -- 기기/차량 보유 정보
    phone_brand VARCHAR(100),      -- 보유 휴대폰 브랜드
    phone_model VARCHAR(100),      -- 보유 휴대폰 모델명
    car_ownership BOOLEAN,         -- 차량 보유 여부
    car_manufacturer VARCHAR(100), -- 보유 차량 제조사
    car_model VARCHAR(100),        -- 보유 차량 모델
    
    -- 복수선택 배열 필드들 (ARRAY 타입)
    owned_electronics VARCHAR(255)[],         -- 보유 전자 제품
    smoking_experience VARCHAR(255)[],        -- 흡연경험
    smoking_brand VARCHAR(255)[],            -- 흡연경험브랜드
    e_cig_heated_brand VARCHAR(255)[],       -- 궐련 / 가열형 전자담배 흡연 경험 브랜드
    e_cig_liquid_brand VARCHAR(255)[],       -- 액상형 전자담배 흡연경험 브랜드
    drinking_experience VARCHAR(255)[],       -- 음주 경험
    
    -- LLM 생성 데이터 (선택사항)
    panel_summary_text TEXT,                  -- 전체 패널 종합 요약 텍스트 (패널 상세보기용, NULL 허용)
    search_labels VARCHAR(100)[] DEFAULT '{}', -- 검색 성능 향상을 위한 핵심 라벨 (GIN 인덱스)
    
    -- 메타데이터
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

-- G1~G7 문항별 요약 텍스트 세그먼트 테이블
CREATE TABLE IF NOT EXISTS panel_summary_segments (
    id SERIAL PRIMARY KEY,
    panel_id VARCHAR(100) NOT NULL REFERENCES panels(panel_id) ON DELETE CASCADE,
    segment_name VARCHAR(255) NOT NULL,
    summary_text TEXT,
    embedding VECTOR(768),
    ts_vector_korean TSVECTOR, -- FTS를 위한 tsvector 컬럼
    updated_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    UNIQUE(panel_id, segment_name) -- UPSERT를 위한 UNIQUE 제약조건
);

-- 스키마 일관성 확인: panel_id는 VARCHAR(100)으로 panels 테이블과 일치
-- 마이그레이션 필요 시: 07_fix_panel_summary_segments_schema.sql 실행

-- FTS 인덱스 생성 (검색 속도 향상)
CREATE INDEX idx_fts_korean ON panel_summary_segments USING GIN(ts_vector_korean);

-- ==========================================
-- 인덱스 생성
-- ==========================================

-- panels 테이블 인덱스
CREATE INDEX IF NOT EXISTS idx_panels_gender ON panels(gender);
CREATE INDEX IF NOT EXISTS idx_panels_age ON panels(age);
CREATE INDEX IF NOT EXISTS idx_panels_region_city ON panels(region_city);
CREATE INDEX IF NOT EXISTS idx_panels_marital_status ON panels(marital_status);
CREATE INDEX IF NOT EXISTS idx_panels_occupation ON panels(occupation);

-- GIN 인덱스 (배열 검색 최적화)
CREATE INDEX IF NOT EXISTS idx_panels_owned_electronics ON panels USING GIN(owned_electronics);
CREATE INDEX IF NOT EXISTS idx_panels_smoking_experience ON panels USING GIN(smoking_experience);
CREATE INDEX IF NOT EXISTS idx_panels_search_labels ON panels USING GIN(search_labels);

-- panel_summary_segments 테이블 인덱스
CREATE INDEX IF NOT EXISTS idx_segments_panel_id ON panel_summary_segments(panel_id);
CREATE INDEX IF NOT EXISTS idx_segments_segment_name ON panel_summary_segments(segment_name);
CREATE INDEX IF NOT EXISTS idx_segments_embedding ON panel_summary_segments USING ivfflat(embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_segments_ts_vector_korean ON panel_summary_segments USING GIN(ts_vector_korean);
