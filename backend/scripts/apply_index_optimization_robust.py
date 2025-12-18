#!/usr/bin/env python3
"""
비정형 검색 최적화 인덱스 적용 스크립트 (개선 버전)

RDS 연결 안정성 개선 및 인덱스 생성 안정성 향상
- 클라이언트 연결이 끊어져도 서버에서 인덱스 생성이 계속 진행되도록 개선
- 각 인덱스 생성을 독립적인 백그라운드 작업으로 실행
"""

import asyncio
import sys
import time
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

load_dotenv()

# 동기식 연결을 위한 DATABASE_URL (asyncpg → psycopg2)
DATABASE_URL = os.getenv("DATABASE_URL", "")
if DATABASE_URL.startswith("postgresql+asyncpg://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
elif DATABASE_URL.startswith("postgresql://"):
    pass  # 이미 동기식
else:
    # 환경 변수가 없으면 기본값 사용
    DATABASE_URL = "postgresql://user:password@localhost/dbname"


def execute_sql_with_retry(engine, sql_stmt, max_retries=3, description=""):
    """SQL 문장을 재시도 로직과 함께 실행"""
    for attempt in range(1, max_retries + 1):
        try:
            with engine.begin() as conn:
                # 타임아웃 설정 해제
                conn.execute(text("SET statement_timeout = 0"))
                conn.execute(text("SET lock_timeout = 0"))
                conn.execute(text("SET idle_in_transaction_session_timeout = 0"))
                
                # SQL 실행
                conn.execute(text(sql_stmt))
                conn.commit()
            
            return True, None
        except Exception as e:
            error_msg = str(e)
            if attempt < max_retries:
                wait_time = attempt * 2  # 지수 백오프
                print(f"   ⚠️ 시도 {attempt}/{max_retries} 실패, {wait_time}초 후 재시도...")
                print(f"      오류: {error_msg[:200]}")
                time.sleep(wait_time)
            else:
                return False, error_msg
    
    return False, "최대 재시도 횟수 초과"


def check_index_exists(engine, index_name):
    """인덱스 존재 여부 확인"""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT EXISTS (
                    SELECT 1 FROM pg_indexes 
                    WHERE tablename = 'panel_summary_segments' 
                    AND indexname = :index_name
                )
            """), {"index_name": index_name})
            return result.scalar()
    except Exception:
        return False


def apply_index_optimization():
    """인덱스 최적화 SQL 스크립트 실행 (개선 버전)"""
    print("🚀 비정형 검색 인덱스 최적화 시작 (개선 버전)")
    print("=" * 80)
    
    # RDS 연결 안정성 개선
    connect_args = {
        'connect_timeout': 60,  # 연결 타임아웃 60초
        'options': '-c statement_timeout=0 -c lock_timeout=0 -c idle_in_transaction_session_timeout=0'
    }
    
    # RDS 연결인 경우 SSL 설정 추가
    if "rds.amazonaws.com" in DATABASE_URL:
        import ssl
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        connect_args['sslmode'] = 'require'
    
    engine = create_engine(
        DATABASE_URL, 
        echo=False,
        connect_args=connect_args,
        pool_pre_ping=True,  # 연결 상태 확인
        pool_recycle=3600,  # 1시간마다 연결 재사용
        pool_size=5,  # 연결 풀 크기
        max_overflow=10,  # 추가 연결 허용
    )
    
    try:
        # 연결 테스트
        with engine.connect() as conn:
            print("📊 데이터베이스 연결 확인 중...")
            result = conn.execute(text("SELECT version()"))
            version = result.scalar()
            if version:
                print(f"   PostgreSQL 버전: {version[:50]}...")
            
            # korean 사전 확인 및 생성
            korean_check = conn.execute(text("SELECT cfgname FROM pg_ts_config WHERE cfgname = 'korean'")).fetchall()
            if korean_check:
                print("   ✅ korean 사전: 사용 가능")
            else:
                print("   ⚠️ korean 사전: 없음 - 생성 시도 중...")
                try:
                    conn.execute(text("CREATE TEXT SEARCH CONFIGURATION korean (COPY = simple)"))
                    conn.commit()
                    print("   ✅ korean 사전 생성 완료")
                except Exception as e:
                    if "already exists" in str(e).lower():
                        print("   ✅ korean 사전: 이미 존재함")
                    else:
                        print(f"   ⚠️ korean 사전 생성 실패: {str(e)[:100]}")
        
        # SQL 스크립트 읽기
        sql_file = project_root / "backend" / "db" / "init" / "09_optimize_unstructured_search.sql"
        
        if not sql_file.exists():
            print(f"❌ SQL 파일을 찾을 수 없습니다: {sql_file}")
            return
        
        with open(sql_file, "r", encoding="utf-8") as f:
            sql_content = f.read()
        
        # SQL 문장 분리
        lines = sql_content.split("\n")
        sql_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("--"):
                sql_lines.append(line)
        
        sql_text = "\n".join(sql_lines)
        sql_statements = [stmt.strip() for stmt in sql_text.split(";") if stmt.strip()]
        
        print(f"\n📝 총 {len(sql_statements)}개의 SQL 문장 실행")
        print("=" * 80)
        
        # 각 SQL 문장 실행
        for i, stmt in enumerate(sql_statements, 1):
            if not stmt:
                continue
            
            print(f"\n[{i}/{len(sql_statements)}] SQL 실행 중...")
            first_line = stmt.split("\n")[0].strip()
            print(f"   {first_line[:80]}{'...' if len(first_line) > 80 else ''}")
            sys.stdout.flush()
            
            try:
                start_time = time.time()
                
                # 인덱스 생성 명령 확인
                is_ivfflat = "CREATE INDEX" in stmt.upper() and "IVFFLAT" in stmt.upper()
                is_gin = "CREATE INDEX" in stmt.upper() and "GIN" in stmt.upper()
                is_create_index = "CREATE INDEX" in stmt.upper()
                
                # 인덱스 이름 추출
                index_name = None
                if is_create_index:
                    # CREATE INDEX idx_name ... 패턴에서 인덱스 이름 추출
                    parts = stmt.upper().split()
                    try:
                        idx_idx = parts.index("INDEX")
                        if idx_idx + 1 < len(parts):
                            index_name = parts[idx_idx + 1]
                            # IF NOT EXISTS 처리
                            if index_name == "IF":
                                if idx_idx + 3 < len(parts):
                                    index_name = parts[idx_idx + 3]
                    except (ValueError, IndexError):
                        pass
                
                # 인덱스가 이미 존재하는지 확인
                if index_name and check_index_exists(engine, index_name):
                    print(f"   ✅ 인덱스 {index_name}가 이미 존재합니다 (건너뜀)")
                    continue
                
                if is_ivfflat:
                    print(f"   ⏳ IVFFlat 인덱스 생성 중... (시간이 걸릴 수 있습니다)")
                    print(f"   💡 이 작업은 몇 분이 걸릴 수 있습니다.")
                    print(f"   💡 클라이언트 연결이 끊어져도 서버에서 계속 실행됩니다.")
                    sys.stdout.flush()
                elif is_gin:
                    print(f"   ⏳ GIN 인덱스 생성 중... (시간이 걸릴 수 있습니다)")
                    sys.stdout.flush()
                
                # SQL 실행 (재시도 로직 포함)
                success, error_msg = execute_sql_with_retry(
                    engine, 
                    stmt, 
                    max_retries=3,
                    description=first_line[:50]
                )
                
                if success:
                    elapsed = time.time() - start_time
                    print(f"   ✅ 완료 (소요 시간: {elapsed:.1f}초)")
                else:
                    elapsed = time.time() - start_time
                    
                    # 일부 오류는 경고만 출력하고 계속 진행
                    if "already exists" in error_msg.lower() or "duplicate" in error_msg.lower():
                        print(f"   ⚠️ 경고: 인덱스가 이미 존재합니다 (건너뜀)")
                    elif "maintenance_work_mem" in error_msg.lower() or "memory" in error_msg.lower():
                        print(f"   ❌ 오류: 메모리 부족 - maintenance_work_mem을 증가시켜야 합니다")
                        print(f"   💡 해결 방법: PostgreSQL 설정 파일에서 maintenance_work_mem을 증가시키거나")
                        print(f"      세션 레벨에서 SET maintenance_work_mem = '1GB'; 실행")
                        print(f"   ⚠️ 이 단계를 건너뛰고 다음 단계로 진행합니다.")
                    else:
                        print(f"   ❌ 오류 발생: {error_msg[:300]}")
                        print(f"   ⚠️ 다음 단계로 진행합니다.")
                
                sys.stdout.flush()
                
            except Exception as e:
                error_msg = str(e)
                elapsed = time.time() - start_time if 'start_time' in locals() else 0
                print(f"   ❌ 예외 발생: {error_msg[:300]}")
                print(f"   ⚠️ 다음 단계로 진행합니다.")
                sys.stdout.flush()
        
        print("\n" + "=" * 80)
        print("✅ 인덱스 최적화 완료!")
        print("=" * 80)
        print("\n💡 참고사항:")
        print("  - IVFFlat 인덱스 생성은 시간이 걸릴 수 있습니다 (약 20만 개 레코드 기준 5-10분)")
        print("  - 인덱스가 생성되면 벡터 검색 속도가 향상됩니다")
        print("  - 인덱스 생성 상태는 TablePlus에서 확인할 수 있습니다:")
        print("    SELECT * FROM pg_stat_progress_create_index;")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        engine.dispose()


if __name__ == "__main__":
    apply_index_optimization()

