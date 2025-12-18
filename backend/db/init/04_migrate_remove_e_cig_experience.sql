-- ==========================================
-- 마이그레이션: e_cigarette_experience 필드 제거 및 panel_summary_text NULL 허용
-- ==========================================

-- panel_summary_text를 NULL 허용으로 변경
ALTER TABLE panels ALTER COLUMN panel_summary_text DROP NOT NULL;

-- e_cigarette_experience 컬럼 삭제 (기존 데이터가 있다면 먼저 백업 권장)
ALTER TABLE panels DROP COLUMN IF EXISTS e_cigarette_experience;

