from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .process_manager import ProcessManager
from .repository import ProjectRepository
from .routes import api, pages
from .runtime import build_runtime_info


PACKAGE_DIR = Path(__file__).resolve().parent


def create_app(workspace: Optional[str] = None) -> FastAPI:
    repository = ProjectRepository(
        workspace or os.getenv("STORYMEM_WEB_WORKSPACE", ".runtime/web")
    )
    process_manager = ProcessManager(repository)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        process_manager.reconcile()
        yield

    app = FastAPI(title="StoryMem Notebook", lifespan=lifespan)
    app.state.repository = repository
    app.state.process_manager = process_manager
    app.state.runtime_info = build_runtime_info(
        schema_version=repository.schema_version(),
        workspace=repository.workspace,
    )
    app.state.templates = Jinja2Templates(directory=PACKAGE_DIR / "templates")
    app.mount("/static", StaticFiles(directory=PACKAGE_DIR / "static"), name="static")
    app.include_router(pages.router)
    app.include_router(api.router)
    return app


app = create_app()
