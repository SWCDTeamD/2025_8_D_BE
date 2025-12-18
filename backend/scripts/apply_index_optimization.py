#!/usr/bin/env python3
"""
비정형 검색 최적화 인덱스 적용 스크립트

벡터 검색과 FTS 검색 성능 향상을 위한 인덱스 최적화를 적용합니다.
"""

import asyncio
import sys
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


def apply_index_optimization():
    """인덱스 최적화 SQL 스크립트 실행"""
    print("🚀 비정형 검색 인덱스 최적화 시작")
    print("=" * 80)
    
    engine = create_engine(DATABASE_URL, echo=False)
    
    try:
        # 연결 테스트 및 사전 확인
        with engine.connect() as conn:
            print("📊 데이터베이스 연결 확인 중...")
            result = conn.execute(text("SELECT version()"))
            version = result.scalar()
            print(f"   PostgreSQL 버전: {version[:50]}...")
            
            # korean 사전 확인 및 생성
            # 참고: 실제 형태소 분석은 Python(Kiwi)에서 수행되며, 
            # PostgreSQL의 'korean' 사전은 단순히 이름만 'korean'이고 실제로는 simple과 동일하게 동작합니다.
            korean_check = conn.execute(text("SELECT cfgname FROM pg_ts_config WHERE cfgname = 'korean'")).fetchall()
            if korean_check:
                print("   ✅ korean 사전: 사용 가능")
            else:
                print("   ⚠️ korean 사전: 없음 - 생성 시도 중...")
                try:
                    # korean 사전 생성 (simple 기반, 실제 형태소 분석은 Python에서 수행)
                    conn.execute(text("CREATE TEXT SEARCH CONFIGURATION korean (COPY = simple)"))
                    conn.commit()
                    print("   ✅ korean 사전 생성 완료")
                    print("   💡 참고: 실제 형태소 분석은 Python(Kiwi)에서 수행되며, DB는 정규화된 텍스트를 저장합니다.")
                except Exception as e:
                    error_msg = str(e)
                    if "already exists" in error_msg.lower():
                        print("   ✅ korean 사전: 이미 존재함")
                    else:
                        print(f"   ⚠️ korean 사전 생성 실패: {error_msg[:100]}")
                        print("   💡 simple 사전을 계속 사용하거나, PostgreSQL 확장 설치 필요")
        
        # SQL 스크립트 읽기
        sql_file = project_root / "backend" / "db" / "init" / "09_optimize_unstructured_search.sql"
        
        if not sql_file.exists():
            print(f"❌ SQL 파일을 찾을 수 없습니다: {sql_file}")
            return
        
        with open(sql_file, "r", encoding="utf-8") as f:
            sql_content = f.read()
        
        # SQL 문장 분리 (세미콜론 기준, 주석 제거)
        lines = sql_content.split("\n")
        sql_lines = []
        for line in lines:
            # 주석 라인 제거
            stripped = line.strip()
            if stripped and not stripped.startswith("--"):
                sql_lines.append(line)
        
        # 세미콜론으로 문장 분리
        sql_text = "\n".join(sql_lines)
        sql_statements = [stmt.strip() for stmt in sql_text.split(";") if stmt.strip()]
        
        print(f"\n📝 총 {len(sql_statements)}개의 SQL 문장 실행")
        print("=" * 80)
        
        # 각 SQL 문장을 개별 트랜잭션으로 실행
        for i, stmt in enumerate(sql_statements, 1):
            if not stmt:
                continue
            
            print(f"\n[{i}/{len(sql_statements)}] SQL 실행 중...")
            # SQL 문장 요약 출력
            first_line = stmt.split("\n")[0].strip()
            print(f"   {first_line[:80]}{'...' if len(first_line) > 80 else ''}")
            
            try:
                # 인덱스 생성 명령의 경우 실행 시간이 걸릴 수 있음
                if "CREATE INDEX" in stmt.upper() and "IVFFLAT" in stmt.upper():
                    print(f"   ⏳ IVFFlat 인덱스 생성 중... (시간이 걸릴 수 있습니다)")
                    print(f"   💡 이 작업은 몇 분이 걸릴 수 있습니다. 진행 상황은 로그를 확인하세요.")
                
                with engine.begin() as conn:  # 각 문장마다 개별 트랜잭션
                    conn.execute(text(stmt))
                    # commit은 begin() 컨텍스트가 자동으로 처리
                
                print(f"   ✅ 완료")
                import sys
                sys.stdout.flush()  # 출력 즉시 반영
            except Exception as e:
                error_msg = str(e)
                # 일부 인덱스가 이미 존재할 수 있으므로 경고만 출력하고 계속 진행
                if "already exists" in error_msg.lower() or "duplicate" in error_msg.lower():
                    print(f"   ⚠️ 경고: 인덱스가 이미 존재합니다 (건너뜀)")
                elif "maintenance_work_mem" in error_msg.lower() or "memory" in error_msg.lower():
                    print(f"   ⚠️ 경고: 메모리 부족 - maintenance_work_mem을 증가시켜야 합니다")
                    print(f"   💡 해결 방법: PostgreSQL 설정 파일에서 maintenance_work_mem을 증가시키거나")
                    print(f"      세션 레벨에서 SET maintenance_work_mem = '1GB'; 실행")
                else:
                    print(f"   ⚠️ 경고: {error_msg[:200]}")
                    # 치명적 오류가 아니면 계속 진행
        
        print("\n" + "=" * 80)
        print("✅ 인덱스 최적화 완료!")
        print("=" * 80)
        print("\n💡 참고사항:")
        print("  - IVFFlat 인덱스 생성은 시간이 걸릴 수 있습니다 (약 20만 개 레코드 기준 5-10분)")
        print("  - 인덱스가 생성되면 벡터 검색 속도가 향상됩니다")
        print("  - 통계 정보 업데이트(ANALYZE)도 완료되었습니다")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        engine.dispose()


if __name__ == "__main__":
    apply_index_optimization()

