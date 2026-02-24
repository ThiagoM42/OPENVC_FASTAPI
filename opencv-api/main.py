from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from sklearn.cluster import KMeans
import io

app = FastAPI(
    title="Leitor de Gabarito API",
    description="API para leitura automática de gabaritos via visão computacional",
    version="1.0.0",
)


# ─────────────────────────────────────────────
# FUNÇÕES DE PROCESSAMENTO
# ─────────────────────────────────────────────

def recortar_tabela(img: np.ndarray) -> np.ndarray:
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
    hsv = cv2.cvtColor(tabela, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array([100, 80, 50]), np.array([140, 255, 255]))


def densidade_bolha(mask: np.ndarray, cx: int, cy: int, r: int) -> float:
    roi = mask[max(0, cy - r) : cy + r, max(0, cx - r) : cx + r]
    return cv2.countNonZero(roi) / roi.size if roi.size > 0 else 0.0


def processar_gabarito(img: np.ndarray, num_questoes: int, n_alternativas: int) -> dict:
    tabela = recortar_tabela(img)

    circles = detectar_circulos(tabela)
    if circles is None:
        raise ValueError("Nenhuma bolha detectada na imagem.")

    # Filtrar: remover coluna de números (esquerda) e header (topo)
    x_min_resp = int(tabela.shape[1] * 0.25)
    body = [c for c in circles if c[1] > 150 and c[0] > x_min_resp]

    if len(body) < n_alternativas:
        raise ValueError(f"Bolhas insuficientes detectadas: {len(body)}")

    # Clusterizar colunas (A-E)
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

    questoes: dict[str, str] = {}
    for ri in sorted(dados.keys()):
        opcoes = dados[ri]
        melhor = max(opcoes, key=opcoes.get)
        questoes[str(ri + 1)] = LETRAS[melhor] if melhor < len(LETRAS) else "?"

    return {"total": len(questoes), "questoes": questoes}


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Leitor de Gabarito API rodando!"}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}


@app.post("/gabarito/ler", tags=["Gabarito"])
async def ler_gabarito(
    file: UploadFile = File(..., description="Imagem do gabarito (JPG ou PNG)"),
    num_questoes: int = Query(10, ge=1, le=100, description="Número de questões"),
    num_alternativas: int = Query(5, ge=2, le=5, description="Número de alternativas (2-5)"),
):
    """
    Recebe a imagem de um gabarito e retorna as respostas detectadas.

    - **file**: imagem JPG ou PNG do gabarito preenchido
    - **num_questoes**: quantidade de questões (padrão: 10)
    - **num_alternativas**: quantidade de alternativas por questão (padrão: 5 = A-E)
    """
    # Validar tipo do arquivo
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=422,
            detail="Formato inválido. Envie uma imagem JPG ou PNG.",
        )

    # Ler imagem do upload
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=422, detail="Não foi possível decodificar a imagem.")

    try:
        resultado = processar_gabarito(img, num_questoes, num_alternativas)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

    return JSONResponse(content=resultado)