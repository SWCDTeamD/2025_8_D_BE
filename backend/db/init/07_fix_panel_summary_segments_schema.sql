-- ==========================================
-- 마이그레이션: panel_summary_segments 테이블 스키마 수정
-- 비정형 검색이 제대로 동작하도록 수정
-- ==========================================

-- 1. panel_id 타입을 VARCHAR(100)으로 변경 (panels 테이블과 일치)
-- 기존 데이터는 유지되며, 타입만 변경됨
ALTER TABLE panel_summary_segments 
    ALTER COLUMN panel_id TYPE VARCHAR(100);

-- 2. updated_at 컬럼 추가 (ETL 코드에서 사용하지 않지만, 추후를 위해 추가)
-- 기존 데이터는 현재 시간으로 설정됨
ALTER TABLE panel_summary_segments 
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW() NOT NULL;

-- 3. 인덱스 확인 및 최적화
-- 이미 존재하는 인덱스는 재생성하지 않음 (CREATE INDEX IF NOT EXISTS 사용)

-- 4. 외래키 제약조건 확인 (ON DELETE CASCADE 유지)
-- 이미 존재하므로 변경 불필요

-- 완료 메시지
DO $$
BEGIN
    RAISE NOTICE '✅ panel_summary_segments 테이블 스키마 수정 완료';
    RAISE NOTICE '   - panel_id: VARCHAR(255) → VARCHAR(100)';
    RAISE NOTICE '   - updated_at 컬럼 추가됨';
END $$;

