-- ==========================================
-- 메타데이터 및 라벨링 테이블
-- ==========================================

-- 1. 컬럼 메타데이터 테이블
CREATE TABLE IF NOT EXISTS column_metadata (
    column_name VARCHAR(100) PRIMARY KEY,
    name_ko VARCHAR(200),
    name_en VARCHAR(200),
    type VARCHAR(50) NOT NULL, -- 'categorical', 'numerical', 'boolean', 'multi_select', 'text'
    description TEXT,
    unit VARCHAR(50),
    range_min INTEGER,
    range_max INTEGER,
    analysis_priority VARCHAR(20), -- 'high', 'medium', 'low'
    chart_types TEXT[], -- 배열: ['pie', 'bar', 'histogram', ...]
    statistics TEXT[], -- 배열: ['mean', 'median', 'std', ...]
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

-- 2. 라벨 값 테이블 (label.json의 값들)
CREATE TABLE IF NOT EXISTS label_values (
    id SERIAL PRIMARY KEY,
    column_name VARCHAR(100) NOT NULL REFERENCES column_metadata(column_name) ON DELETE CASCADE,
    value TEXT NOT NULL, -- 실제 값 (문자열 또는 숫자 문자열)
    value_type VARCHAR(20) NOT NULL, -- 'string', 'number', 'boolean'
    display_order INTEGER DEFAULT 0, -- 표시 순서
    is_active BOOLEAN DEFAULT TRUE, -- 활성화 여부
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    UNIQUE(column_name, value)
);

-- 3. 카테고리 그룹 테이블
CREATE TABLE IF NOT EXISTS category_groups (
    group_key VARCHAR(100) PRIMARY KEY,
    name_ko VARCHAR(200) NOT NULL,
    name_en VARCHAR(200),
    description TEXT,
    analysis_focus TEXT[], -- 배열: ['분포', '집중도', '대표성']
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

-- 4. 카테고리 그룹 - 컬럼 매핑 테이블
CREATE TABLE IF NOT EXISTS category_group_columns (
    id SERIAL PRIMARY KEY,
    group_key VARCHAR(100) NOT NULL REFERENCES category_groups(group_key) ON DELETE CASCADE,
    column_name VARCHAR(100) NOT NULL REFERENCES column_metadata(column_name) ON DELETE CASCADE,
    display_order INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    UNIQUE(group_key, column_name)
);

-- 인덱스 생성
CREATE INDEX IF NOT EXISTS idx_label_values_column ON label_values(column_name);
CREATE INDEX IF NOT EXISTS idx_label_values_active ON label_values(column_name, is_active) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_category_group_columns_group ON category_group_columns(group_key);
CREATE INDEX IF NOT EXISTS idx_category_group_columns_column ON category_group_columns(column_name);

-- 코멘트 추가
COMMENT ON TABLE column_metadata IS '컬럼별 메타데이터 (타입, 설명, 차트 추천 등)';
COMMENT ON TABLE label_values IS '각 컬럼의 가능한 값 목록 (label.json 데이터)';
COMMENT ON TABLE category_groups IS '의미있는 카테고리 그룹 (인구통계, 경제력 등)';
COMMENT ON TABLE category_group_columns IS '카테고리 그룹과 컬럼의 매핑';

