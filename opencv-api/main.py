from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import os

app = FastAPI()


# ──────────────────────────── helpers ────────────────────────────
def _decode_image(source) -> np.ndarray:
    """Aceita bytes (requisição HTTP) ou caminho de arquivo (str/Path)."""
    if isinstance(source, (bytes, bytearray)):
        arr = np.frombuffer(source, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Não foi possível decodificar os bytes da imagem.")
    else:
        if not os.path.exists(source):
            raise FileNotFoundError(f"Imagem não encontrada: {source}")
        img = cv2.imread(str(source))
        if img is None:
            raise ValueError(f"Não foi possível carregar: {source}")
    return img


def _cluster(values: list, tol: int) -> list:
    """Agrupa valores numéricos próximos e retorna o centro de cada grupo."""
    sv = sorted(values)
    if not sv:
        return []
    groups = [[sv[0]]]
    for v in sv[1:]:
        if v - groups[-1][-1] <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [int(np.mean(g)) for g in groups]


def _blue_fill(hsv: np.ndarray, cx: int, cy: int, r: int) -> float:
    """
    Proporção de pixels com cor azul/roxo escuro (tinta de caneta) dentro do círculo.
    Usa 50% do raio para ignorar a borda e focar no interior da bolha.
    """
    inner = max(int(r * 0.50), 5)
    mask_circle = np.zeros(hsv.shape[:2], dtype=np.uint8)
    cv2.circle(mask_circle, (int(cx), int(cy)), inner, 255, -1)
    area = float(np.sum(mask_circle > 0))
    if area == 0:
        return 0.0

    lower = np.array([85, 40, 20])
    upper = np.array([165, 255, 200])
    mask_blue = cv2.inRange(hsv, lower, upper)

    filled = float(np.sum((mask_blue > 0) & (mask_circle > 0)))
    return filled / area


def _find_circles(gray: np.ndarray, h: int, w: int, x_min: int,
                  num_options: int = 5, min_rows: int = 8):
    """
    Busca adaptativa de círculos: testa param2 de 50 até 25 e retorna
    o primeiro resultado com grid válido (5 colunas e >= min_rows linhas).
    Garante detecção mesmo em imagens com menor contraste ou nitidez.
    """
    min_r    = max(10, int(h * 0.010))
    max_r    = max(20, int(h * 0.032))
    min_dist = max(15, int(min_r * 1.2))

    best = None

    for param2 in [50, 45, 40, 35, 30, 25]:
        raw = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT,
            dp=1, minDist=min_dist,
            param1=50, param2=param2,
            minRadius=min_r, maxRadius=max_r
        )
        if raw is None:
            continue

        circles = [
            (int(cx), int(cy), int(r))
            for cx, cy, r in np.round(raw[0]).astype(int)
            if cx > x_min
        ]
        if not circles:
            continue

        # Clusteriza colunas e seleciona as num_options mais populadas
        col_all = _cluster([cx for cx, cy, r in circles], tol=int(w * 0.020))
        col_pop: dict = {i: 0 for i in range(len(col_all))}
        for cx, cy, r in circles:
            ci = min(range(len(col_all)), key=lambda i: abs(cx - col_all[i]))
            col_pop[ci] += 1
        top_cols    = sorted(sorted(col_pop, key=col_pop.get, reverse=True)[:num_options])
        col_centers = [col_all[i] for i in top_cols]

        row_centers = _cluster([cy for cx, cy, r in circles], tol=int(h * 0.020))

        best = (circles, row_centers, col_centers)

        # Grid válido: colunas corretas e linhas suficientes
        if len(col_centers) == num_options and len(row_centers) >= min_rows:
            return best

    return best  # melhor encontrado, mesmo que incompleto


# ──────────────────────────── core ────────────────────────────

def read_gabarito(source, num_questions: int = None) -> list:
    """
    Lê um gabarito de múltipla escolha e retorna as respostas do aluno.

    Parâmetros
    ----------
    source        : bytes da imagem (requisição HTTP) ou str com caminho do arquivo.
    num_questions : número de questões esperadas; se None, detecta automaticamente.

    Retorna
    -------
    Lista de dicts: [{"question_number": int, "student_answer": str | None}, ...]
    student_answer é None quando a célula não apresenta tinta azul de caneta.
    """

    NUM_OPTIONS      = 5
    LEFT_MARGIN_FRAC = 0.30
    BLUE_MIN         = 0.05   # mínimo de fill azul para considerar marcado (5%)

    # ── 1. Carrega a imagem ───────────────────────────────────────────────
    img = _decode_image(source)
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # ── 2. Detecta círculos com busca adaptativa de param2 ────────────────
    x_min  = int(w * LEFT_MARGIN_FRAC)
    result = _find_circles(gray, h, w, x_min, num_options=NUM_OPTIONS)

    if result is None:
        raise RuntimeError("Nenhum círculo detectado. Verifique a qualidade e o formato da imagem.")

    circles, row_centers, col_centers = result
    avg_r = int(np.mean([r for cx, cy, r in circles])) if circles else int(h * 0.020)

    # ── 3. Remove linhas de cabeçalho ─────────────────────────────────────
    # O cabeçalho (rótulos A B C D E) tem um gap vertical bem maior que as questões
    if len(row_centers) > 1:
        gaps       = [row_centers[i + 1] - row_centers[i] for i in range(len(row_centers) - 1)]
        median_gap = float(np.median(gaps))
        header_idx = {i for i, g in enumerate(gaps) if g > median_gap * 1.5}
    else:
        header_idx = set()

    question_rows = [ry for i, ry in enumerate(row_centers) if i not in header_idx]
    if num_questions is not None:
        question_rows = question_rows[:num_questions]

    # ── 4. Para cada célula do grid, mede presença de tinta azul ─────────
    options = ['A', 'B', 'C', 'D', 'E']
    results = []

    for q_num, ry in enumerate(question_rows, start=1):
        fills     = [_blue_fill(hsv, cx_col, ry, avg_r) for cx_col in col_centers]
        best_col  = int(np.argmax(fills))
        best_fill = fills[best_col]
        marked    = best_fill >= BLUE_MIN and best_col < len(options)

        results.append({
            "question_number": q_num,
            "student_answer": options[best_col] if marked else None
        })

    return results


# ──────────────────────────── endpoints ────────────────────────────

@app.get("/")
def health():
    return {"status": "OpenCV API running 🚀"}


@app.post("/process")
async def process_image(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        answers = read_gabarito(contents)
    except Exception as e:
        print(f"Erro: {e}")
        return {"error": str(e)}

    return JSONResponse(content=answers)
