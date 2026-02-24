"""
main.py — API FastAPI para leitura de gabaritos.

Endpoints:
  POST /gabarito          → analisa imagem, retorna respostas em JSON
  POST /gabarito/debug    → idem + imagem anotada em base64
  GET  /health            → healthcheck
  GET  /docs              → Swagger UI (automático pelo FastAPI)
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.gabarito_reader import GabaritoResult, process_image_bytes

# ════════════════════════════════════════════════════════════════════════════════
# Logging
# ════════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("gabarito_api")

# ════════════════════════════════════════════════════════════════════════════════
# Constantes
# ════════════════════════════════════════════════════════════════════════════════

MAX_FILE_SIZE = 10 * 1024 * 1024          # 10 MB
ALLOWED_CONTENT_TYPES = {
    "image/jpeg", "image/jpg", "image/png",
    "image/bmp", "image/webp", "image/tiff",
}


# ════════════════════════════════════════════════════════════════════════════════
# Schemas Pydantic (documentação automática no Swagger)
# ════════════════════════════════════════════════════════════════════════════════

class QuestaoSchema(BaseModel):
    numero: int = Field(..., description="Número da questão (começa em 1)")
    resposta: str | None = Field(None, description="Letra marcada (A-E) ou null se em branco / múltipla")
    multipla_marcacao: bool = Field(False, description="True se mais de uma bolha foi marcada")
    marcacoes: list[str] = Field(default_factory=list, description="Todas as letras marcadas")


class GabaritoResponse(BaseModel):
    alinhamento: str = Field(..., description="Método de alinhamento aplicado na imagem")
    total_questoes: int
    total_respondidas: int
    total_nao_respondidas: int
    total_multiplas: int
    questoes: list[QuestaoSchema]

    class Config:
        json_schema_extra = {
            "example": {
                "alinhamento": "perspectiva (68%)",
                "total_questoes": 10,
                "total_respondidas": 9,
                "total_nao_respondidas": 1,
                "total_multiplas": 0,
                "questoes": [
                    {"numero": 1, "resposta": "B", "multipla_marcacao": False, "marcacoes": ["B"]},
                    {"numero": 2, "resposta": "C", "multipla_marcacao": False, "marcacoes": ["C"]},
                    {"numero": 3, "resposta": None, "multipla_marcacao": False, "marcacoes": []},
                ],
            }
        }


class GabaritoDebugResponse(GabaritoResponse):
    debug_image: str | None = Field(
        None,
        description="Imagem PNG anotada em base64 (círculos e grade identificados)",
    )


class HealthResponse(BaseModel):
    status: str
    version: str


# ════════════════════════════════════════════════════════════════════════════════
# App
# ════════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Gabarito API iniciada ✓")
    yield
    logger.info("Gabarito API encerrada.")


app = FastAPI(
    title="Gabarito Reader API",
    description=(
        "API para leitura automática de gabaritos de múltipla escolha.\n\n"
        "Suporta:\n"
        "- Caneta **azul** e **preta**\n"
        "- Correção automática de **perspectiva** e **rotação**\n"
        "- Imagens JPEG, PNG, BMP, WebP\n"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # ajuste em produção
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ════════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════════

async def _validate_and_read(file: UploadFile) -> bytes:
    """Valida content-type e tamanho; retorna bytes da imagem."""
    content_type = (file.content_type or "").lower().split(";")[0].strip()
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Tipo de arquivo não suportado: '{content_type}'. "
                f"Use: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}"
            ),
        )

    data = await file.read()
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Arquivo muito grande ({len(data) / 1024 / 1024:.1f} MB). Máximo: 10 MB.",
        )
    if len(data) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Arquivo vazio.",
        )
    return data


def _result_to_response(result: GabaritoResult, include_debug: bool = False) -> dict:
    d = result.to_dict()
    if not include_debug:
        d.pop("debug_image", None)
    return d


# ════════════════════════════════════════════════════════════════════════════════
# Endpoints
# ════════════════════════════════════════════════════════════════════════════════

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Healthcheck",
    tags=["Utilitários"],
)
async def health():
    """Verifica se a API está operacional."""
    return {"status": "ok", "version": app.version}


@app.post(
    "/gabarito",
    response_model=GabaritoResponse,
    status_code=status.HTTP_200_OK,
    summary="Analisa gabarito",
    tags=["Gabarito"],
    responses={
        400: {"description": "Imagem inválida ou gabarito não reconhecido"},
        413: {"description": "Arquivo maior que 10 MB"},
        415: {"description": "Formato de arquivo não suportado"},
        422: {"description": "Nenhum arquivo enviado"},
        500: {"description": "Erro interno no processamento"},
    },
)
async def analisar_gabarito(
    file: UploadFile = File(..., description="Imagem do gabarito (JPEG, PNG, BMP, WebP)"),
):
    """
    Recebe uma imagem de gabarito e retorna as respostas detectadas.

    - Corrige automaticamente perspectiva e rotação
    - Detecta marcações com caneta azul ou preta
    - Identifica múltiplas marcações na mesma questão
    """
    t0 = time.perf_counter()
    data = await _validate_and_read(file)

    try:
        result = process_image_bytes(data, return_debug_image=False)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception("Erro inesperado ao processar gabarito: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro interno ao processar a imagem.",
        )

    elapsed = time.perf_counter() - t0
    logger.info(
        "Processado '%s' — %d questão(ões), alinhamento: %s — %.2fs",
        file.filename, result.total_questoes, result.alinhamento, elapsed,
    )
    return JSONResponse(content=_result_to_response(result))


@app.post(
    "/gabarito/debug",
    response_model=GabaritoDebugResponse,
    status_code=status.HTTP_200_OK,
    summary="Analisa gabarito (com imagem de debug)",
    tags=["Gabarito"],
    responses={
        400: {"description": "Imagem inválida ou gabarito não reconhecido"},
        413: {"description": "Arquivo maior que 10 MB"},
        415: {"description": "Formato de arquivo não suportado"},
    },
)
async def analisar_gabarito_debug(
    file: UploadFile = File(..., description="Imagem do gabarito"),
):
    """
    Igual ao `POST /gabarito`, mas inclui o campo `debug_image` na resposta:
    um PNG em **base64** com os círculos detectados coloridos e a grade sobreposta.

    Útil para inspecionar visualmente os resultados da detecção.
    """
    t0 = time.perf_counter()
    data = await _validate_and_read(file)

    try:
        result = process_image_bytes(data, return_debug_image=True)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception("Erro inesperado: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Erro interno ao processar a imagem.")

    elapsed = time.perf_counter() - t0
    logger.info("Debug processado '%s' — %.2fs", file.filename, elapsed)
    return JSONResponse(content=_result_to_response(result, include_debug=True))
