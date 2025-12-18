-- ==========================================
-- 사용되지 않는 panel_attribute_lookup 테이블 삭제
-- ==========================================

-- 이 테이블은 코드베이스 어디에서도 사용되지 않으며,
-- panels 테이블의 배열 필드에 직접 값이 저장되므로 불필요합니다.

DROP TABLE IF EXISTS panel_attribute_lookup CASCADE;

