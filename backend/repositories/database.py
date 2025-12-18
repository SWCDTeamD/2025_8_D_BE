"""
데이터베이스 연결 설정

SQLAlchemy async 엔진 및 세션 관리
"""

import os
from pathlib import Path
from typing import AsyncGenerator

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

# .env 파일 로드
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=env_path)

# DB 연결 URL
# 환경 변수에서 로드하고, 동기 드라이버 URL이면 비동기 드라이버로 변환
raw_url = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://swcd:swcdpw@127.0.0.1:5432/swcddb"
)

# sslmode 파라미터 제거 (충돌 방지)
if "?sslmode=" in raw_url or "&sslmode=" in raw_url:
    import re
    raw_url = re.sub(r'[?&]sslmode=[^&]*', '', raw_url)

# postgresql://를 postgresql+asyncpg://로 변환 (비동기 지원)
if raw_url.startswith("postgresql://") and "+asyncpg" not in raw_url:
    DATABASE_URL = raw_url.replace("postgresql://", "postgresql+asyncpg://", 1)
else:
    DATABASE_URL = raw_url

# RDS 연결인 경우 SSL 설정 단순화
# asyncpg는 URL 쿼리 파라미터로 sslmode를 받지 않으므로 connect_args로 설정
is_rds = "rds.amazonaws.com" in DATABASE_URL
connect_args = {}

if is_rds:
    # RDS SSL 설정 단순화 (asyncpg가 알아서 처리)
    connect_args = {
        "ssl": "require",  # asyncpg가 알아서 SSL 컨텍스트 생성
        "server_settings": {
            "statement_timeout": "60000",  # 60초 (쿼리 타임아웃)
        }
    }

# async 엔진 생성
engine = create_async_engine(
    DATABASE_URL,
    echo=False,  # SQL 로깅 (디버깅 시 True)
    pool_pre_ping=True,  # 연결 상태 확인
    connect_args=connect_args if connect_args else None,
)

# 세션 팩토리
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """데이터베이스 세션 의존성 (FastAPI Depends용)"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

