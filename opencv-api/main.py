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


def _br_diff_mask(img: np.ndarray, threshold: int = 50) -> np.ndarray:
    """
    Máscara de pixels onde o canal Azul supera o Vermelho por `threshold`.
    Detecta tinta azul/roxa de caneta mesmo quando dessaturada.
    """
    b = img[:, :, 0].astype(np.int16)
    r = img[:, :, 2].astype(np.int16)
    diff = np.clip(b - r, 0, 255).astype(np.uint8)
    return (diff > threshold).astype(np.uint8) * 255


def _br_fill(img: np.ndarray, cx: int, cy: int, r: int, threshold: int = 50) -> float:
    """
    Proporção de pixels com B-R > threshold dentro do círculo.
    Robusto para canetas azuis vibrantes e dessaturadas.
    """
    inner = max(int(r * 0.55), 5)
    mask_circle = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.circle(mask_circle, (int(cx), int(cy)), inner, 255, -1)
    area = float(np.sum(mask_circle > 0))
    if area == 0:
        return 0.0
    diff_mask = _br_diff_mask(img, threshold)
    filled = float(np.sum((diff_mask > 0) & (mask_circle > 0)))
    return filled / area


def _find_marked_centers(img: np.ndarray, x_min: int, min_area: int = 300):
    """
    Encontra os centros dos círculos MARCADOS usando diferença B-R forte (>150).
    Retorna lista de (cx, cy, r_estimado).
    """
    mask = _br_diff_mask(img, threshold=150)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        if cx < x_min:
            continue
        r_est = max(int(np.sqrt(area / np.pi)), 10)
        centers.append((cx, cy, r_est))

    return centers


def _regularize_cols(raw_cols: list, n: int = 5) -> list:
    """
    Recebe colunas detectadas (possivelmente irregulares) e retorna n colunas
    regularmente espaçadas que melhor se encaixam.
    """
    if len(raw_cols) < 2:
        return raw_cols
    best_score = -1
    best_result = raw_cols[:n]
    for i in range(len(raw_cols)):
        for j in range(i + 1, len(raw_cols)):
            for idx_i in range(n):
                for idx_j in range(idx_i + 1, n):
                    if idx_j - idx_i != (j - i):
                        continue
                    spacing = (raw_cols[j] - raw_cols[i]) / (idx_j - idx_i)
                    if spacing < 40 or spacing > 400:
                        continue
                    start = raw_cols[i] - idx_i * spacing
                    grid = [start + k * spacing for k in range(n)]
                    score = sum(
                        1 for c in raw_cols
                        if min(abs(c - g) for g in grid) < spacing * 0.3
                    )
                    if score > best_score:
                        best_score = score
                        best_result = [int(round(g)) for g in grid]
    return best_result


def _find_circles(gray: np.ndarray, h: int, w: int, x_min: int,
                  num_options: int = 5, min_rows: int = 8):
    """
    Busca adaptativa de círculos (param2 50→25).
    Retorna (circles, row_centers, col_centers) com o melhor resultado.
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

        col_all = _cluster([cx for cx, cy, r in circles], tol=int(w * 0.020))
        col_pop: dict = {i: 0 for i in range(len(col_all))}
        for cx, cy, r in circles:
            ci = min(range(len(col_all)), key=lambda i: abs(cx - col_all[i]))
            col_pop[ci] += 1
        top_cols    = sorted(sorted(col_pop, key=col_pop.get, reverse=True)[:num_options])
        col_centers = [col_all[i] for i in top_cols]
        row_centers = _cluster([cy for cx, cy, r in circles], tol=int(h * 0.020))

        best = (circles, row_centers, col_centers)
        if len(col_centers) == num_options and len(row_centers) >= min_rows:
            return best

    return best


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
    """
    NUM_OPTIONS      = 5
    LEFT_MARGIN_FRAC = 0.25
    BLUE_MIN         = 0.05

    # ── 1. Carrega a imagem ───────────────────────────────────────────────
    img = _decode_image(source)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x_min = int(w * LEFT_MARGIN_FRAC)

    # ── 2. Detecta centros dos círculos MARCADOS via B-R diff ─────────────
    marked = _find_marked_centers(img, x_min)
    avg_r_marked = int(np.mean([r for cx, cy, r in marked])) if marked else int(h * 0.025)

    # ── 3. Infere colunas a partir dos marcados ───────────────────────────
    col_centers = None
    if len(marked) >= 2:
        xs_marked = sorted(set(_cluster([m[0] for m in marked], tol=int(w * 0.05))))
        if len(xs_marked) >= 2:
            spacings = [xs_marked[i+1] - xs_marked[i] for i in range(len(xs_marked)-1)]
            col_spacing = int(np.median(spacings))
            # Determina qual índice de coluna tem o X mínimo
            min_x = min(xs_marked)
            best_cols = None
            best_score = -1
            for start_idx in range(NUM_OPTIONS):
                start = min_x - start_idx * col_spacing
                cols = [start + k * col_spacing for k in range(NUM_OPTIONS)]
                if not all(x_min * 0.5 < c < w * 0.98 for c in cols):
                    continue
                score = sum(
                    1 for mx in [m[0] for m in marked]
                    if min(abs(mx - c) for c in cols) < col_spacing * 0.3
                )
                if score > best_score:
                    best_score = score
                    best_cols = [int(c) for c in cols]
            col_centers = best_cols

    # ── 4. Fallback: HoughCircles adaptativo para encontrar colunas ───────
    result = _find_circles(gray, h, w, x_min, num_options=NUM_OPTIONS)
    if result is None and col_centers is None:
        raise RuntimeError("Nenhum círculo detectado. Verifique a qualidade e o formato da imagem.")

    hough_circles, hough_rows, hough_cols = result if result else ([], [], [])

    # Se não conseguiu colunas pelos marcados, usa HoughCircles + regularização
    if col_centers is None:
        col_centers = _regularize_cols(hough_cols, n=NUM_OPTIONS)

    avg_r = avg_r_marked if marked else (
        int(np.mean([r for cx, cy, r in hough_circles])) if hough_circles else int(h * 0.025)
    )

    # ── 5. Detecta TODAS as linhas (marcadas + não marcadas) ──────────────
    all_rows = set(_cluster([m[1] for m in marked], tol=int(h * 0.025)))

    # Complementa com linhas do HoughCircles
    for ry in hough_rows:
        if not any(abs(ry - er) < avg_r for er in all_rows):
            all_rows.add(ry)

    # Se ainda faltam linhas, tenta param2 mais baixo
    if len(all_rows) < 8:
        min_r = max(10, int(h * 0.010))
        max_r = max(20, int(h * 0.032))
        for p2 in range(25, 10, -5):
            raw = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1,
                minDist=max(15, int(min_r * 1.2)),
                param1=50, param2=p2, minRadius=min_r, maxRadius=max_r)
            if raw is None:
                continue
            extra = [(int(cx), int(cy), int(r))
                     for cx, cy, r in np.round(raw[0]).astype(int) if cx > x_min]
            for ry in _cluster([cy for cx, cy, r in extra], tol=int(h * 0.020)):
                if not any(abs(ry - er) < avg_r for er in all_rows):
                    all_rows.add(ry)
            if len(all_rows) >= 8:
                break

    row_centers = sorted(all_rows)

    # ── 6. Remove linhas de cabeçalho ─────────────────────────────────────
    if len(row_centers) > 1:
        gaps       = [row_centers[i+1] - row_centers[i] for i in range(len(row_centers)-1)]
        median_gap = float(np.median(gaps))
        header_idx = {i for i, g in enumerate(gaps) if g > median_gap * 1.5}
    else:
        header_idx = set()

    question_rows = [ry for i, ry in enumerate(row_centers) if i not in header_idx]
    if num_questions is not None:
        question_rows = question_rows[:num_questions]

    # ── 7. Classifica cada célula do grid ─────────────────────────────────
    options = ['A', 'B', 'C', 'D', 'E']
    results = []

    for q_num, ry in enumerate(question_rows, start=1):
        fills     = [_br_fill(img, cx, ry, avg_r) for cx in col_centers]
        best_col  = int(np.argmax(fills))
        best_fill = fills[best_col]
        marked_q  = best_fill >= BLUE_MIN and best_col < len(options)

        results.append({
            "question_number": q_num,
            "student_answer": options[best_col] if marked_q else None
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
