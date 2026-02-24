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

PROCESSING_TIMEOUT_SECONDS = 10
MAX_IMAGE_SIZE_MB = 10
TARGET_WIDTH = 1201
TARGET_HEIGHT = 1600

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
  "answers": {
    "1": "B",
    "2": "C",
    "3": "D",
    "4": null,
    "5": "E"
  }
}
```

> Questões sem nenhuma bolha marcada retornam `null`.

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

    # ── Passo 2: agrupar linhas por proximidade de Y ──
    # Tolerância dinâmica: 1.2× raio médio das bolhas detectadas
    # Isso adapta automaticamente ao tamanho das bolhas em cada imagem,
    # evitando que linhas próximas se fundam ou linhas distantes se separem.
    r_medio = float(np.median([c[2] for c in body]))
    TOLERANCIA_Y = int(r_medio * 1.2)

    body_sorted = sorted(body, key=lambda c: c[1])

    grupos: list[list] = []
    grupo_atual = [body_sorted[0]]
    for c in body_sorted[1:]:
        if abs(c[1] - grupo_atual[-1][1]) <= TOLERANCIA_Y:
            grupo_atual.append(c)
        else:
            grupos.append(grupo_atual)
            grupo_atual = [c]
    grupos.append(grupo_atual)

    # ── Passo 3: limpar cada grupo — 1 círculo por coluna (maior raio) ──
    linhas_limpas: list[dict] = []
    for grupo in grupos:
        colunas: dict[int, any] = {}
        for c in grupo:
            ci = col_idx(int(c[0]))
            if ci not in colunas or c[2] > colunas[ci][2]:
                colunas[ci] = c
        if len(colunas) >= max(2, n_alternativas - 2):  # linha válida com pelo menos 3 bolhas
            linhas_limpas.append(colunas)

    if not linhas_limpas:
        raise ValueError("Não foi possível identificar linhas de questões.")

    # Limitar ao número de questões solicitado
    linhas_limpas = linhas_limpas[:num_questoes]

    # ── Passo 4: calcular densidades e determinar resposta ──
    mask = criar_mascara_azul(tabela)
    LETRAS = ["A", "B", "C", "D", "E"]

    # Densidade mínima para considerar uma bolha como marcada.
    # Bolhas vazias ficam em 0.000; bolhas preenchidas ficam acima de 0.30.
    # Threshold de 0.05 garante margem segura contra ruído de iluminação.
    DENSIDADE_MINIMA = 0.05

    answers: dict[str, str | None] = {}
    for i, linha in enumerate(linhas_limpas):
        densidades = {
            ci: densidade_bolha(mask, int(c[0]), int(c[1]), int(c[2]))
            for ci, c in linha.items()
        }
        melhor_col = max(densidades, key=densidades.get)
        melhor_dens = densidades[melhor_col]

        # Se nenhuma bolha atingiu o threshold, a questão está em branco
        if melhor_dens < DENSIDADE_MINIMA:
            answers[str(i + 1)] = None
        else:
            answers[str(i + 1)] = LETRAS[melhor_col] if melhor_col < len(LETRAS) else None

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
                        "answers": {"1": "B", "2": "C", "3": "D", "4": None, "5": "E"},
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
