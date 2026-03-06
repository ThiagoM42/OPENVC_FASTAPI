from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, HTMLResponse, Response
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
import cv2
import numpy as np
from sklearn.cluster import KMeans
import asyncio
import concurrent.futures
from functools import partial

# ─────────────────────────────────────────────
# CONFIGURAÇÕES
# ─────────────────────────────────────────────

PROCESSING_TIMEOUT_SECONDS = 10
MAX_IMAGE_SIZE_MB = 10
TARGET_WIDTH = 1201
TARGET_HEIGHT = 1600
DENSIDADE_MINIMA_AZUL  = 0.05   # caneta azul: gap claro entre 0.00 e 0.33
DENSIDADE_MINIMA_PRETA = 0.15   # lápis/caneta preta: threshold maior devido ao ruído das bordas

# ─────────────────────────────────────────────
# DOCS
# ─────────────────────────────────────────────

description = """
## 📋 Leitor de Gabarito API

API para leitura automática de gabaritos de múltipla escolha via visão computacional.

### Como funciona

1. **Envie** uma foto do gabarito preenchido (JPG ou PNG)
2. A API **detecta** automaticamente as bolhas usando OpenCV + HoughCircles
3. As colunas A–E são fixadas via **KMeans** no eixo X
4. As linhas (questões) são agrupadas por **proximidade de Y** com tolerância dinâmica
5. Círculos espúrios são removidos mantendo **1 círculo por coluna por linha**
6. Retorna um **JSON** com a resposta de cada questão

### Requisitos da imagem

- Formato: **JPG ou PNG**
- Tamanho máximo: **10 MB**
- A folha deve estar **visível e enquadrada** na foto
- Bolhas preenchidas com **caneta azul ou preta**
- Iluminação razoável, sem sombras fortes sobre as bolhas

### Exemplo de resposta

```json
{
  "total": 10,
  "answers": [
    {"question_number": 1, "student_answer": "B"},
    {"question_number": 2, "student_answer": "C"},
    {"question_number": 3, "student_answer": null},
    {"question_number": 4, "student_answer": "D"},
    {"question_number": 5, "student_answer": "E"}
  ]
}
```

> Questões sem nenhuma bolha marcada retornam `student_answer: null`.

### Limites

| Recurso | Limite |
|---------|--------|
| Tamanho máximo da imagem | 10 MB |
| Tempo máximo de processamento | 10 segundos |
| Questões por gabarito | 1 – 100 |
| Alternativas por questão | 2 – 5 |

### Códigos de erro

| Código | Motivo |
|--------|--------|
| `408`  | Tempo de processamento excedido (> 10s) |
| `413`  | Imagem maior que 10 MB |
| `422`  | Imagem inválida, formato não suportado ou bolhas não detectadas |
| `500`  | Erro interno de processamento |
"""

tags_metadata = [
    {"name": "Health",   "description": "Endpoints para verificar se a API está no ar."},
    {"name": "Gabarito", "description": "Leitura e processamento de gabaritos preenchidos."},
]

app = FastAPI(
    title="Leitor de Gabarito API",
    description=description,
    version="1.0.0",
    openapi_tags=tags_metadata,
    contact={"name": "Suporte", "email": "suporte@exemplo.com"},
    license_info={"name": "MIT"},
    docs_url=None,
    redoc_url=None,
)


# ─────────────────────────────────────────────
# FUNÇÕES DE PROCESSAMENTO
# ─────────────────────────────────────────────

def recortar_tabela(img: np.ndarray) -> np.ndarray:
    """Localiza e recorta a grade do gabarito na imagem."""
    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(cinza, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10
    )
    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return img
    maior = max(contornos, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(maior)
    return img[y : y + h, x : x + w]


def detectar_circulos(tabela: np.ndarray) -> np.ndarray | None:
    """Detecta todos os círculos (bolhas) via HoughCircles."""
    cinza = cv2.cvtColor(tabela, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(cinza, (7, 7), 2)
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=80,
        param1=50,
        param2=22,   # valor baixo para garantir detecção mesmo em imagens com baixo contraste
        minRadius=25,
        maxRadius=80,
    )
    if circles is None:
        return None
    return np.round(circles[0]).astype("int")


def criar_mascara_tinta(tabela: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Detecta automaticamente o tipo de tinta (azul ou preta/lápis)
    e retorna a máscara binária + o threshold de densidade adequado.
    """
    hsv = cv2.cvtColor(tabela, cv2.COLOR_BGR2HSV)
    mask_azul = cv2.inRange(hsv, np.array([100, 80, 50]), np.array([140, 255, 255]))

    cinza = cv2.cvtColor(tabela, cv2.COLOR_BGR2GRAY)
    _, mask_preta = cv2.threshold(cinza, 100, 255, cv2.THRESH_BINARY_INV)
    mask_preta = cv2.morphologyEx(
        mask_preta, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )

    # Se houver pixels azuis suficientes, usa máscara azul; caso contrário, preta
    if cv2.countNonZero(mask_azul) > cv2.countNonZero(mask_preta) * 0.1:
        return mask_azul, DENSIDADE_MINIMA_AZUL
    return mask_preta, DENSIDADE_MINIMA_PRETA


def densidade_bolha(mask: np.ndarray, cx: int, cy: int, r: int) -> float:
    """Fração de pixels marcados dentro da bolha."""
    roi = mask[max(0, cy - r) : cy + r, max(0, cx - r) : cx + r]
    return cv2.countNonZero(roi) / roi.size if roi.size > 0 else 0.0


def processar_gabarito(img: np.ndarray, num_questoes: int, n_alternativas: int) -> dict:
    """
    Pipeline completo:
      1. Recorta a tabela
      2. Detecta todos os círculos com HoughCircles
      3. Fixa as N colunas (A–E) via KMeans no eixo X
      4. Agrupa linhas por proximidade de Y (tolerância dinâmica)
      5. Remove círculos espúrios: mantém 1 por coluna por linha (maior raio)
      6. Calcula densidade de tinta azul e retorna a letra marcada
    """
    tabela = recortar_tabela(img)

    circles = detectar_circulos(tabela)
    if circles is None:
        raise ValueError("Nenhuma bolha detectada na imagem.")

    # Filtrar coluna de números das questões (esquerda ~25%) e cabeçalho (topo)
    x_min_resp = int(tabela.shape[1] * 0.25)
    body = [c for c in circles if c[1] > 150 and c[0] > x_min_resp]

    if len(body) < n_alternativas:
        raise ValueError(f"Bolhas insuficientes detectadas: {len(body)}")

    # ── Passo 1: fixar colunas via KMeans no eixo X ──
    # Usar todos os círculos garante amostras suficientes por coluna → resultado estável
    xs_all = np.array([c[0] for c in body]).reshape(-1, 1)
    col_centers = sorted(
        KMeans(n_clusters=n_alternativas, random_state=0, n_init=20)
        .fit(xs_all)
        .cluster_centers_.flatten()
    )

    def col_idx(cx: int) -> int:
        return int(np.argmin([abs(cx - c) for c in col_centers]))

    # ── Passo 2: detectar linhas via KMeans no eixo Y ──
    # Usar KMeans com num_questoes clusters é mais robusto que agrupamento por
    # tolerância fixa, pois funciona mesmo quando o raio da bolha é maior que
    # o espaçamento entre linhas (gabaritos compactos / fotos de perto).
    ys_arr = np.array([c[1] for c in body]).reshape(-1, 1)
    row_centers = sorted(
        KMeans(n_clusters=num_questoes, random_state=0, n_init=20)
        .fit(ys_arr)
        .cluster_centers_.flatten()
    )

    def nearest_row(cy: int) -> int:
        return int(np.argmin([abs(cy - r) for r in row_centers]))

    # ── Passo 3: montar grid — 1 círculo por (linha, coluna), maior raio vence ──
    grid: dict[int, dict[int, any]] = {}
    for c in body:
        cx, cy, r = int(c[0]), int(c[1]), int(c[2])
        ri = nearest_row(cy)
        ci = col_idx(cx)
        grid.setdefault(ri, {})
        if ci not in grid[ri] or c[2] > grid[ri][ci][2]:
            grid[ri][ci] = c

    if not grid:
        raise ValueError("Não foi possível identificar linhas de questões.")

    # ── Passo 4: calcular densidades e determinar resposta ──
    mask, DENSIDADE_MINIMA = criar_mascara_tinta(tabela)
    LETRAS = ["A", "B", "C", "D", "E"]

    answers: dict[str, str | None] = {}
    for ri in sorted(grid.keys()):
        linha = grid[ri]
        densidades = {
            ci: densidade_bolha(mask, int(c[0]), int(c[1]), int(c[2]))
            for ci, c in linha.items()
        }
        melhor_col = max(densidades, key=densidades.get)
        melhor_dens = densidades[melhor_col]

        # Se nenhuma bolha atingiu o threshold, a questão está em branco
        if melhor_dens < DENSIDADE_MINIMA:
            answers[str(ri + 1)] = None
        else:
            answers[str(ri + 1)] = LETRAS[melhor_col] if melhor_col < len(LETRAS) else None

    return {
        "total": len(answers),
        "answers": [
            {"question_number": int(q), "student_answer": a}
            for q, a in answers.items()
        ],
    }


# ─────────────────────────────────────────────
# ENDPOINTS — HEALTH
# ─────────────────────────────────────────────

@app.get("/", tags=["Health"], summary="Root")
def root():
    """Verifica se a API está no ar."""
    return {"status": "ok", "message": "Leitor de Gabarito API rodando!"}


@app.get("/health", tags=["Health"], summary="Health Check")
def health():
    """Retorna o status de saúde da aplicação."""
    return {"status": "healthy"}


# ─────────────────────────────────────────────
# ENDPOINTS — GABARITO
# ─────────────────────────────────────────────

@app.post(
    "/gabarito/ler",
    tags=["Gabarito"],
    summary="Ler gabarito",
    response_description="JSON com o total de questões e a letra marcada em cada uma",
    responses={
        200: {
            "description": "Gabarito lido com sucesso",
            "content": {
                "application/json": {
                    "example": {
                        "total": 5,
                        "answers": [
                        {"question_number": 1, "student_answer": "B"},
                        {"question_number": 2, "student_answer": "C"},
                        {"question_number": 3, "student_answer": None},
                        {"question_number": 4, "student_answer": "D"},
                        {"question_number": 5, "student_answer": "E"},
                    ],
                    }
                }
            },
        },
        408: {"description": f"Timeout — processamento excedeu {PROCESSING_TIMEOUT_SECONDS}s"},
        413: {"description": f"Imagem maior que {MAX_IMAGE_SIZE_MB} MB"},
        422: {"description": "Imagem inválida ou bolhas não detectadas"},
        500: {"description": "Erro interno de processamento"},
    },
)
async def ler_gabarito(
    file: UploadFile = File(..., description="Foto do gabarito preenchido (JPG ou PNG)"),
    num_questoes: int = Query(10, ge=1, le=100, description="Número de questões"),
    num_alternativas: int = Query(5, ge=2, le=5, description="Alternativas por questão (2–5)"),
):
    """
    Processa a imagem de um gabarito e retorna as respostas detectadas.

    - **file** — foto JPG ou PNG do gabarito (máx. 10 MB)
    - **num_questoes** — quantidade de questões no gabarito (padrão: 10)
    - **num_alternativas** — quantidade de alternativas por questão (padrão: 5 = A–E)

    Tempo máximo de processamento: **10 segundos**.
    """
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=422,
            detail="Formato inválido. Envie uma imagem JPG ou PNG.",
        )

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_IMAGE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"Imagem muito grande ({size_mb:.1f} MB). Limite: {MAX_IMAGE_SIZE_MB} MB.",
        )

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=422, detail="Não foi possível decodificar a imagem.")

    # Padroniza resolução → processamento consistente independente da câmera/dispositivo
    img = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)

    # Executar em thread separada com timeout (OpenCV/KMeans são bloqueantes)
    loop = asyncio.get_event_loop()
    fn = partial(processar_gabarito, img, num_questoes, num_alternativas)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            resultado = await asyncio.wait_for(
                loop.run_in_executor(executor, fn),
                timeout=PROCESSING_TIMEOUT_SECONDS,
            )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=408,
            detail=f"Tempo de processamento excedido ({PROCESSING_TIMEOUT_SECONDS}s). "
                   "Tente com uma imagem menor ou de melhor qualidade.",
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

    return JSONResponse(content=resultado)

@app.post("/gabarito/compressImage", tags=["compressImage"], summary="Comprimir imagem")
async def compress_image(
    file: UploadFile = File(..., description="Foto do gabarito preenchido (JPG ou PNG)"),
):
    """
    Endpoint auxiliar para comprimir imagens grandes antes de enviá-las para leitura.
    Retorna o arquivo JPEG comprimido.
    """
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=422,
            detail="Formato inválido. Envie uma imagem JPG ou PNG.",
        )

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=422, detail="Não foi possível decodificar a imagem.")

    # Comprimir imagem para reduzir tamanho (ajustar qualidade conforme necessário)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]  # qualidade entre 0-100
    ok, compressed = cv2.imencode('.jpg', img, encode_param)
    if not ok:
        raise HTTPException(status_code=500, detail="Falha ao comprimir a imagem.")

    compressed_bytes = compressed.tobytes()
    original_name = (file.filename or "imagem").rsplit(".", 1)[0]
    output_name = f"{original_name}_compressed.jpg"

    return Response(
        content=compressed_bytes,
        media_type="image/jpeg",
        headers={"Content-Disposition": f'attachment; filename="{output_name}"'},
    )
# ─────────────────────────────────────────────
# DOCS CUSTOMIZADAS
# ─────────────────────────────────────────────

@app.get("/docs", include_in_schema=False)
def swagger_ui() -> HTMLResponse:
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Leitor de Gabarito — Swagger UI",
        swagger_favicon_url="https://fastapi.tiangolo.com/img/favicon.png",
        swagger_ui_parameters={
            "defaultModelsExpandDepth": -1,
            "docExpansion": "list",
            "filter": True,
            "tryItOutEnabled": True,
            "persistAuthorization": True,
            "displayRequestDuration": True,
        },
    )


@app.get("/redoc", include_in_schema=False)
def redoc_ui() -> HTMLResponse:
    return get_redoc_html(
        openapi_url="/openapi.json",
        title="Leitor de Gabarito — ReDoc",
        redoc_favicon_url="https://fastapi.tiangolo.com/img/favicon.png",
        with_google_fonts=True,
    )


@app.get("/openapi.json", include_in_schema=False)
def openapi_schema():
    return get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        tags=tags_metadata,
        routes=app.routes,
    )