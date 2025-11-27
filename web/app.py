from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from evaluation.parsing_utils import init_parser, SpoofingParser

ROOT_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"


def resolve_dataset_path(path_str: str) -> Path:
    if not path_str:
        raise ValueError("数据路径不能为空")
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path.resolve()


@lru_cache(maxsize=16)
def get_dataset(resolved_path: str) -> list[dict[str, Any]]:
    dataset_path = Path(resolved_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {dataset_path}")
    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("期望数据为列表格式")
    return data


@lru_cache(maxsize=16)
def get_parser(resolved_path: str) -> SpoofingParser:
    dataset = get_dataset(resolved_path)
    if not dataset:
        raise ValueError("数据集为空，无法初始化解析器")
    return init_parser(dataset)


def parse_entry(entry: dict[str, Any], source: str, dataset_path: str) -> dict[str, Any]:
    text = entry.get(source)
    if not isinstance(text, str):
        raise ValueError(f"数据字段 {source} 缺失或格式错误")

    parser = get_parser(dataset_path)
    parsed = parser(text)

    return {
        "source": source,
        "raw_text": text,
        "parsed": parsed,
        "extra": {
            "audio": entry.get("audio"),
        },
    }


app = FastAPI(title="Spoofing 数据展示", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def root() -> FileResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="前端页面缺失，请先构建。")
    return FileResponse(index_path)


@app.get("/api/random-entry")
def random_entry(path: str = Query(..., description="数据集 JSON 文件路径（支持相对路径）")) -> dict[str, Any]:
    try:
        dataset_path = resolve_dataset_path(path)
        resolved_path = str(dataset_path)
        dataset = get_dataset(resolved_path)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not dataset:
        raise HTTPException(status_code=500, detail="数据集为空")

    entry = random.choice(dataset)
    try:
        pred = parse_entry(entry, source="pred", dataset_path=resolved_path)
        ref = parse_entry(entry, source="ref", dataset_path=resolved_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"pred": pred, "ref": ref}


@app.get("/api/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}
