from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 라우터 임포트 (형제 모듈에서)
from backend.routers.search_router import router as search_router
from backend.routers.panel_router import router as panel_router
from backend.routers.analysis_router import router as analysis_router


def create_app() -> FastAPI:
    """FastAPI 애플리케이션 팩토리

    - 라우터 등록
    """
    app = FastAPI(title="Panel NL Search API", version="0.1.0")

    # CORS (프론트 개발 서버 허용)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:5174",
            "http://127.0.0.1:5174",
            "http://localhost:5175",
            "http://127.0.0.1:5175",
        ],
        allow_origin_regex=r"https?://(localhost|127\\.0\\.0\\.1)(:\\d+)?$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 라우터 등록
    app.include_router(search_router, prefix="/api")
    app.include_router(panel_router, prefix="/api")
    app.include_router(analysis_router, prefix="/api")

    return app


app = create_app()


