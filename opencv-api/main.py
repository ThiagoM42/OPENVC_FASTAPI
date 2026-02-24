from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import os

app = FastAPI()


# ──────────────────────────── helpers ────────────────────────────

def _decode_image(source) -> np.ndarray:
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


def _is_blue_pen(img: np.ndarray, x_min: int) -> bool:
    """Detecta se a marcação foi feita com caneta azul/roxa."""
    b    = img[:, :, 0].astype(np.int16)
    r_ch = img[:, :, 2].astype(np.int16)
    diff = np.clip(b - r_ch, 0, 255)
    region = diff[:, x_min:]
    pct = float(np.sum(region > 50)) / region.size
    return pct > 0.005


def _adaptive_dark_threshold(gray: np.ndarray, x_min: int) -> int:
    """
    Calcula threshold adaptativo para dark_fill via Otsu.
    Funciona tanto para caneta preta quanto para lápis (cinza médio).
    """
    region = gray[:, x_min:]
    otsu_val, _ = cv2.threshold(region, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark_pixels = region[region < int(otsu_val)]
    if len(dark_pixels) > 100:
        # 90º percentil dos pixels escuros captura tanto lápis (100-130)
        # quanto caneta preta (40-80)
        return int(np.percentile(dark_pixels, 90))
    return 120  # fallback seguro


def _br_fill(img: np.ndarray, cx: int, cy: int, r: int,
             threshold: int = 50) -> float:
    """Fill de pixels onde B-R > threshold (caneta azul/roxa)."""
    inner = max(int(r * 0.55), 3)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (int(cx), int(cy)), inner, 255, -1)
    area = float(np.sum(mask > 0))
    if area == 0:
        return 0.0
    b    = img[:, :, 0].astype(np.int16)
    r_ch = img[:, :, 2].astype(np.int16)
    diff = np.clip(b - r_ch, 0, 255).astype(np.uint8)
    return float(np.sum((diff > threshold) & (mask > 0))) / area


def _dark_fill(gray: np.ndarray, cx: int, cy: int, r: int,
               threshold: int = 120) -> float:
    """Fill de pixels escuros (caneta preta ou lápis)."""
    inner = max(int(r * 0.55), 3)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(mask, (int(cx), int(cy)), inner, 255, -1)
    area = float(np.sum(mask > 0))
    if area == 0:
        return 0.0
    return float(np.sum((gray < threshold) & (mask > 0))) / area


def _detect_answers(fills: list,
                    fill_min: float = 0.05,
                    relative_factor: float = 2.0) -> list:
    """
    Detecta quais alternativas estão marcadas em uma linha.

    Retorna lista de índices (pode ter 0, 1 ou mais elementos).
    Suporta múltiplas marcações: todas as colunas com fill >= 45%
    do melhor fill são consideradas marcadas.
    """
    if not fills:
        return []
    best = max(fills)
    if best < fill_min:
        return []
    # O melhor deve se destacar dos outros
    best_idx = int(np.argmax(fills))
    others   = [f for i, f in enumerate(fills) if i != best_idx]
    avg_o    = float(np.mean(others)) if others else 0.0
    if avg_o > 0 and best < avg_o * relative_factor:
        return []
    # Inclui todas as alternativas com >= 45% do melhor
    multi_thresh = max(fill_min, best * 0.45)
    return [i for i, f in enumerate(fills) if f >= multi_thresh]


def _find_marked_br(img: np.ndarray, x_min: int, h: int) -> list:
    """Detecta centros de círculos marcados com caneta azul (B-R > 150)."""
    b    = img[:, :, 0].astype(np.int16)
    r_ch = img[:, :, 2].astype(np.int16)
    diff = np.clip(b - r_ch, 0, 255).astype(np.uint8)
    mask = (diff > 150).astype(np.uint8) * 255
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(100, int((h * 0.012) ** 2 * 0.3))
    centers = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        if cx < x_min:
            continue
        r_est = max(int(np.sqrt(cv2.contourArea(cnt) / np.pi)), 5)
        centers.append((cx, cy, r_est))
    return centers


def _regularize_cols(raw_cols: list, n: int = 5) -> list:
    """Ajusta n colunas regularmente espaçadas ao conjunto detectado."""
    if len(raw_cols) < 2:
        return raw_cols
    best_score, best_result = -1, raw_cols[:n]
    for i in range(len(raw_cols)):
        for j in range(i + 1, len(raw_cols)):
            for idx_i in range(n):
                for idx_j in range(idx_i + 1, n):
                    if idx_j - idx_i != (j - i):
                        continue
                    spacing = (raw_cols[j] - raw_cols[i]) / (idx_j - idx_i)
                    if not (20 <= spacing <= 400):
                        continue
                    start  = raw_cols[i] - idx_i * spacing
                    grid   = [start + k * spacing for k in range(n)]
                    score  = sum(1 for c in raw_cols
                                 if min(abs(c - g) for g in grid) < spacing * 0.3)
                    if score > best_score:
                        best_score = score
                        best_result = [int(round(g)) for g in grid]
    return best_result


def _infer_cols_from_marked(marked_xs: list, x_min: int,
                             w: int, n: int = 5) -> list:
    """Infere n colunas uniformes a partir dos X dos círculos marcados."""
    col_m = _cluster(marked_xs, tol=int(w * 0.05))
    if len(col_m) < 2:
        return []
    spacings    = [col_m[i+1] - col_m[i] for i in range(len(col_m) - 1)]
    col_spacing = int(np.median(spacings))
    best_cols, best_score = None, -1
    for si in range(n):
        start = min(col_m) - si * col_spacing
        cols  = [start + k * col_spacing for k in range(n)]
        if not all(x_min * 0.3 < c < w * 0.98 for c in cols):
            continue
        score = sum(1 for mx in marked_xs
                    if min(abs(mx - c) for c in cols) < col_spacing * 0.3)
        if score > best_score:
            best_score, best_cols = score, [int(c) for c in cols]
    return best_cols or []


def _find_circles_hough(gray: np.ndarray, h: int, w: int, x_min: int,
                        num_options: int = 5, min_rows: int = 8):
    """Busca adaptativa de círculos via HoughCircles (param2 50→15)."""
    min_r    = max(5,  int(h * 0.010))
    max_r    = max(15, int(h * 0.032))
    min_dist = max(8,  int(min_r * 1.2))
    best = None
    for param2 in [50, 45, 40, 35, 30, 25, 20, 15]:
        raw = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT,
                               dp=1, minDist=min_dist,
                               param1=50, param2=param2,
                               minRadius=min_r, maxRadius=max_r)
        if raw is None:
            continue
        circles = [(int(cx), int(cy), int(r))
                   for cx, cy, r in np.round(raw[0]).astype(int)
                   if cx > x_min]
        if not circles:
            continue
        col_all = _cluster([cx for cx, cy, r in circles], tol=int(w * 0.020))
        col_pop = {i: 0 for i in range(len(col_all))}
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
    source        : bytes (HTTP) ou str/Path com caminho do arquivo.
    num_questions : número de questões esperadas; None = detecta automaticamente.

    Retorna
    -------
    Lista de dicts:
      [{"question_number": int,
        "student_answer": str | list[str] | None}, ...]

      - str       → resposta única, ex: "B"
      - list[str] → múltiplas marcações, ex: ["B", "D"]
      - None      → questão em branco
    """
    NUM_OPTIONS      = 5
    LEFT_MARGIN_FRAC = 0.25
    FILL_MIN         = 0.05
    RELATIVE_FACTOR  = 2.0

    # ── 1. Carrega ────────────────────────────────────────────────────────
    img  = _decode_image(source)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x_min = int(w * LEFT_MARGIN_FRAC)

    # ── 2. Detecta tipo de marcação ───────────────────────────────────────
    blue_pen       = _is_blue_pen(img, x_min)
    dark_threshold = _adaptive_dark_threshold(gray, x_min)

    # ── 3. HoughCircles para detectar o grid ──────────────────────────────
    hough = _find_circles_hough(gray, h, w, x_min, num_options=NUM_OPTIONS)
    if hough is None:
        raise RuntimeError(
            "Nenhum círculo detectado. Verifique a qualidade da imagem.")

    hough_circles, hough_rows, hough_cols = hough
    avg_r = (int(np.mean([r for cx, cy, r in hough_circles]))
             if hough_circles else int(h * 0.022))

    # ── 4. Refina colunas com marcados azuis (se caneta azul) ─────────────
    col_centers = None
    blue_marked = []
    if blue_pen:
        blue_marked = _find_marked_br(img, x_min, h)
        if len(blue_marked) >= 2:
            col_centers = _infer_cols_from_marked(
                [m[0] for m in blue_marked], x_min, w, n=NUM_OPTIONS)

    if not col_centers:
        col_centers = _regularize_cols(hough_cols, n=NUM_OPTIONS) or hough_cols

    # ── 5. Combina linhas dos marcados azuis + HoughCircles ───────────────
    all_rows = (set(_cluster([m[1] for m in blue_marked], tol=int(h * 0.025)))
                if blue_marked else set())
    for ry in hough_rows:
        if not any(abs(ry - er) < avg_r * 1.5 for er in all_rows):
            all_rows.add(ry)

    # Complementa se ainda faltam linhas
    if len(all_rows) < 8:
        min_r = max(5, int(h * 0.010))
        max_r = max(15, int(h * 0.032))
        for p2 in range(20, 5, -5):
            raw = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1,
                                   minDist=max(8, int(min_r * 1.2)),
                                   param1=50, param2=p2,
                                   minRadius=min_r, maxRadius=max_r)
            if raw is None:
                continue
            extra = [(int(cx), int(cy), int(r))
                     for cx, cy, r in np.round(raw[0]).astype(int)
                     if cx > x_min]
            for ry in _cluster([cy for cx, cy, r in extra], tol=int(h * 0.020)):
                if not any(abs(ry - er) < avg_r * 1.5 for er in all_rows):
                    all_rows.add(ry)
            if len(all_rows) >= 8:
                break

    row_centers = sorted(all_rows)

    # ── 6. Remove cabeçalho ───────────────────────────────────────────────
    if len(row_centers) > 1:
        gaps       = [row_centers[i+1] - row_centers[i]
                      for i in range(len(row_centers) - 1)]
        median_gap = float(np.median(gaps))
        header_idx = {i for i, g in enumerate(gaps) if g > median_gap * 1.5}
    else:
        header_idx = set()

    question_rows = [ry for i, ry in enumerate(row_centers)
                     if i not in header_idx]
    if num_questions is not None:
        question_rows = question_rows[:num_questions]

    # ── 7. Classifica cada célula ─────────────────────────────────────────
    options = ['A', 'B', 'C', 'D', 'E']
    results = []

    for q_num, ry in enumerate(question_rows, start=1):
        if not col_centers:
            results.append({"question_number": q_num, "student_answer": None})
            continue

        # Calcula fills com o sinal correto para o tipo de caneta
        if blue_pen:
            fills = [_br_fill(img, cx, ry, avg_r) for cx in col_centers]
        else:
            fills = [_dark_fill(gray, cx, ry, avg_r, dark_threshold)
                     for cx in col_centers]

        # Detecta quantas alternativas foram marcadas
        marked_idx = _detect_answers(fills,
                                     fill_min=FILL_MIN,
                                     relative_factor=RELATIVE_FACTOR)

        if not marked_idx:
            answer = None
        elif len(marked_idx) == 1:
            answer = options[marked_idx[0]]
        else:
            answer = [options[i] for i in sorted(marked_idx)]

        results.append({
            "question_number": q_num,
            "student_answer":  answer
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
