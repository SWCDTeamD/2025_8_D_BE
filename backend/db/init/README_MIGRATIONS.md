# 데이터베이스 마이그레이션 가이드

## 마이그레이션 파일 목록

### 07_fix_panel_summary_segments_schema.sql
**목적**: `panel_summary_segments` 테이블 스키마 수정 (비정형 검색 최적화)

**변경 사항**:
- `panel_id` 타입: `VARCHAR(255)` → `VARCHAR(100)` (panels 테이블과 일치)
- `updated_at` 컬럼 추가

**실행 방법**:
```bash
docker exec -i swcd-db-1 psql -U user -d panel_db < backend/db/init/07_fix_panel_summary_segments_schema.sql
```

**주의사항**:
- 기존 데이터는 유지됩니다
- 타입 변경 시 기존 데이터가 호환되는 경우에만 성공합니다
- 실행 전 백업 권장

## 마이그레이션 실행 순서

1. **기존 ETL 프로세스 중지**
   ```bash
   ./backend/scripts/stop_etl.sh
   ```

2. **마이그레이션 실행**
   ```bash
   docker exec -i swcd-db-1 psql -U user -d panel_db < backend/db/init/07_fix_panel_summary_segments_schema.sql
   ```

3. **스키마 확인**
   ```bash
   docker exec swcd-db-1 psql -U user -d panel_db -c "\d panel_summary_segments"
   ```

4. **ETL 재시작**
   ```bash
   ./backend/scripts/resume_etl.sh
   ```

## 기존 데이터 보존

모든 마이그레이션은 기존 데이터를 보존하도록 설계되었습니다:
- `ALTER TABLE` 명령 사용 (테이블 재생성 없음)
- `IF EXISTS` / `IF NOT EXISTS` 조건 사용
- 데이터 변환 시 호환성 확인

## 롤백 방법

마이그레이션을 롤백하려면 역순으로 실행:
```sql
-- 예시 (07_fix_panel_summary_segments_schema.sql 롤백)
ALTER TABLE panel_summary_segments ALTER COLUMN panel_id TYPE VARCHAR(255);
ALTER TABLE panel_summary_segments DROP COLUMN IF EXISTS updated_at;
```

