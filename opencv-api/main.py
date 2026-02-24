from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, HTMLResponse
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

PROCESSING_TIMEOUT_SECONDS = 20   # tempo máximo de processamento por requisição
MAX_IMAGE_SIZE_MB = 20             # tamanho máximo do arquivo enviado

# ─────────────────────────────────────────────
# CONFIGURAÇÃO E DOCS
# ─────────────────────────────────────────────

description = """
## 📋 Leitor de Gabarito API

API para leitura automática de gabaritos de múltipla escolha via visão computacional.

### Como funciona

1. **Envie** uma foto do gabarito preenchido (JPG ou PNG)
2. A API **detecta** automaticamente as bolhas usando OpenCV + HoughCircles
3. O algoritmo **clusteriza** as bolhas em linhas (questões) e colunas (A–E) via KMeans
4. Retorna um **JSON** com a resposta de cada questão

### Requisitos da imagem

- Formato: **JPG ou PNG**
- Tamanho máximo: **20 MB**
- A folha deve estar **visível e enquadrada** na foto
- Bolhas preenchidas com **caneta azul ou preta**
- Iluminação razoável, sem sombras fortes sobre as bolhas

### Exemplo de resposta

```json
{
  "total": 10,
  "answers": {
    "1": "B",
    "2": "C",
    "3": "D",
    "4": "A",
    "5": "E"
  }
}
```

### Limites

| Recurso | Limite |
|---------|--------|
| Tamanho máximo da imagem | 20 MB |
| Tempo máximo de processamento | 20 segundos |
| Questões por gabarito | 1 – 100 |
| Alternativas por questão | 2 – 5 |

### Códigos de erro

| Código | Motivo |
|--------|--------|
| `408`  | Tempo de processamento excedido (> 20s) |
| `413`  | Imagem maior que 20 MB |
| `422`  | Imagem inválida, formato não suportado ou bolhas não detectadas |
| `500`  | Erro interno de processamento |
"""

tags_metadata = [
    {
        "name": "Health",
        "description": "Endpoints para verificar se a API está no ar.",
    },
    {
        "name": "Gabarito",
        "description": "Leitura e processamento de gabaritos preenchidos.",
    },
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
    blur = cv2.GaussianBlur(cinza, (5, 5), 0)
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=60,
        param1=50,
        param2=28,
        minRadius=25,
        maxRadius=65,
    )
    if circles is None:
        return None
    return np.round(circles[0]).astype("int")


def criar_mascara_azul(tabela: np.ndarray) -> np.ndarray:
    """Máscara HSV que isola a tinta azul de caneta."""
    hsv = cv2.cvtColor(tabela, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array([100, 80, 50]), np.array([140, 255, 255]))


def densidade_bolha(mask: np.ndarray, cx: int, cy: int, r: int) -> float:
    """Fração de pixels marcados dentro da bolha."""
    roi = mask[max(0, cy - r) : cy + r, max(0, cx - r) : cx + r]
    return cv2.countNonZero(roi) / roi.size if roi.size > 0 else 0.0


def processar_gabarito(img: np.ndarray, num_questoes: int, n_alternativas: int) -> dict:
    """Pipeline completo: recorte → detecção → clusterização → leitura."""
    tabela = recortar_tabela(img)

    circles = detectar_circulos(tabela)
    if circles is None:
        raise ValueError("Nenhuma bolha detectada na imagem.")

    # Remove coluna de números das questões (esquerda ~25%) e header (topo)
    x_min_resp = int(tabela.shape[1] * 0.25)
    body = [c for c in circles if c[1] > 150 and c[0] > x_min_resp]

    if len(body) < n_alternativas:
        raise ValueError(f"Bolhas insuficientes detectadas: {len(body)}")

    # Clusterizar colunas A–E
    xs = np.array([c[0] for c in body]).reshape(-1, 1)
    col_centers = sorted(
        KMeans(n_clusters=n_alternativas, random_state=0, n_init=10)
        .fit(xs)
        .cluster_centers_.flatten()
    )

    # Clusterizar linhas (questões)
    n_rows = num_questoes or (len(body) // n_alternativas)
    ys = np.array([c[1] for c in body]).reshape(-1, 1)
    row_centers = sorted(
        KMeans(n_clusters=n_rows, random_state=0, n_init=10)
        .fit(ys)
        .cluster_centers_.flatten()
    )

    def nearest(val, centers):
        return int(np.argmin([abs(val - c) for c in centers]))

    mask = criar_mascara_azul(tabela)
    LETRAS = ["A", "B", "C", "D", "E"]

    dados: dict[int, dict[int, float]] = {}
    for c in body:
        cx, cy, r = int(c[0]), int(c[1]), int(c[2])
        ri = nearest(cy, row_centers)
        ci = nearest(cx, col_centers)
        d = densidade_bolha(mask, cx, cy, r)
        dados.setdefault(ri, {})
        dados[ri][ci] = max(dados[ri].get(ci, 0), d)

    answers: dict[str, str] = {}
    for ri in sorted(dados.keys()):
        opcoes = dados[ri]
        melhor = max(opcoes, key=opcoes.get)
        answers[str(ri + 1)] = LETRAS[melhor] if melhor < len(LETRAS) else "?"

    return {"total": len(answers), "answers": answers}


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
                        "answers": {"1": "B", "2": "C", "3": "D", "4": "A", "5": "E"},
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
    # Validar content-type
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=422,
            detail="Formato inválido. Envie uma imagem JPG ou PNG.",
        )

    # Ler bytes e validar tamanho
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_IMAGE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"Imagem muito grande ({size_mb:.1f} MB). Limite: {MAX_IMAGE_SIZE_MB} MB.",
        )

    # Decodificar imagem
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=422, detail="Não foi possível decodificar a imagem.")

    # Executar processamento em thread separada com timeout
    # (OpenCV/KMeans são bloqueantes — rodar no executor evita travar o event loop)
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


# ─────────────────────────────────────────────
# DOCS CUSTOMIZADAS
# ─────────────────────────────────────────────

@app.get("/docs", include_in_schema=False)
def swagger_ui() -> HTMLResponse:
    """Swagger UI com Try it out habilitado e campo de busca."""
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
    """ReDoc — documentação legível e navegável."""
    return get_redoc_html(
        openapi_url="/openapi.json",
        title="Leitor de Gabarito — ReDoc",
        redoc_favicon_url="https://fastapi.tiangolo.com/img/favicon.png",
        with_google_fonts=True,
    )


@app.get("/openapi.json", include_in_schema=False)
def openapi_schema():
    """Schema OpenAPI 3.0 em JSON."""
    return get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        tags=tags_metadata,
        routes=app.routes,
    )