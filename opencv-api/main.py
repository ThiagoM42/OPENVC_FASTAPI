from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import io
import os

app = FastAPI()

@app.get("/")
def health():
    return {"status": "OpenCV API running 🚀"}

@app.post("/process")
async def process_image(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        answers = read_gabarito(contents)

        
        

        # out_json = "gabarito_resultado.json"
        # with open(out_json, "w", encoding="utf-8") as f:
        #     json.dump(answers, f, indent=2, ensure_ascii=False)
        # print(f"\n✓ Resultado salvo em: {out_json}")

    except Exception as e:
        print(f"Erro: {e}")
        return {"error": str(e)}
        
    
    
    return JSONResponse(content=answers)

def load_image(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Imagem não encontrada: {path}")
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Não foi possível carregar: {path}")
    return img


def preprocess(img: np.ndarray):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    return gray, thresh


def cluster_values(values, tolerance=30):
    """Agrupa valores próximos e retorna os centros dos grupos."""
    sv = sorted(values)
    if not sv:
        return []
    groups = [[sv[0]]]
    for v in sv[1:]:
        if v - groups[-1][-1] <= tolerance:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [int(np.mean(g)) for g in groups]


def fill_ratio(thresh: np.ndarray, cx: int, cy: int, r: int) -> float:
    """Proporção de pixels escuros (preenchidos) dentro do círculo."""
    mask = np.zeros(thresh.shape, dtype=np.uint8)
    inner = max(r - 5, 5)
    cv2.circle(mask, (cx, cy), inner, 255, -1)
    area = np.sum(mask > 0)
    filled = np.sum((thresh > 0) & (mask > 0))
    return filled / area if area else 0.0


# ─────────────────────── core ───────────────────────

def read_gabarito(source, debug: bool = False,
                  num_questions: int = 10) -> list:
    """
    Lê o gabarito e retorna lista de dicts com question_number e student_answer.

    Parâmetros
    ----------
    source        : Caminho (str) OU bytes da imagem recebida via requisição
    debug         : Se True, salva gabarito_debug.jpg com anotações visuais
    num_questions : Quantidade de questões esperadas (padrão 10)
    """
    FILL_THRESHOLD = 0.30   # fill ratio mínimo para considerar marcado (abaixo = null)
    MIN_RADIUS     = 18     # raio mínimo dos círculos (px)
    MAX_RADIUS     = 50     # raio máximo

    if isinstance(source, (bytes, bytearray)):
        arr = np.frombuffer(source, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Nao foi possivel decodificar os bytes da imagem.")
    else:
        img = load_image(source)
    h, w = img.shape[:2]
    gray, thresh = preprocess(img)

    # ── 1. Detecta todos os círculos ──────────────────────────────────────
    raw = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT,
        dp=1, minDist=25, param1=50, param2=28,
        minRadius=MIN_RADIUS, maxRadius=MAX_RADIUS
    )
    if raw is None:
        print("⚠  Nenhum círculo detectado. Verifique a qualidade da imagem.")
        return [{"question_number": i, "student_answer": None}
                for i in range(1, num_questions + 1)]

    all_circles = np.round(raw[0]).astype(int)

    # Descarta círculos muito à esquerda (números das questões escritos na lateral)
    # As alternativas ficam na região direita da imagem (x > 30% da largura)
    x_min_answer = w * 0.30
    circles = [(int(cx), int(cy), int(r), fill_ratio(thresh, int(cx), int(cy), int(r)))
               for cx, cy, r in all_circles if cx > x_min_answer]

    # ── 2. Agrupa por linhas (Y) e colunas (X) ───────────────────────────
    row_centers = cluster_values([c[1] for c in circles], tolerance=30)
    col_centers_raw = cluster_values([c[0] for c in circles], tolerance=30)

    # Mantém apenas as 5 colunas mais populadas (as alternativas A-E)
    if len(col_centers_raw) > 5:
        col_pop = {}
        for cx, cy, r, f in circles:
            ci = min(range(len(col_centers_raw)),
                     key=lambda i: abs(cx - col_centers_raw[i]))
            col_pop[ci] = col_pop.get(ci, 0) + 1
        top5 = sorted(sorted(col_pop, key=col_pop.get, reverse=True)[:5])
        col_centers = [col_centers_raw[i] for i in top5]
    else:
        col_centers = col_centers_raw

    # ── 3. Mapeia cada círculo para (linha, coluna) ───────────────────────
    grid: dict[tuple, float] = {}
    for cx, cy, r, f in circles:
        ri = min(range(len(row_centers)), key=lambda i: abs(cy - row_centers[i]))
        ci = min(range(len(col_centers)), key=lambda i: abs(cx - col_centers[i]))
        key = (ri, ci)
        if key not in grid or f > grid[key]:
            grid[key] = f

    # ── 4. Identifica linhas de questão (≥ 3 colunas detectadas) ─────────
    row_pop = {}
    for ri, ci in grid:
        row_pop[ri] = row_pop.get(ri, 0) + 1
    question_rows = sorted([r for r, cnt in row_pop.items() if cnt >= 3])
    question_rows = question_rows[:num_questions]

    # ── 5. Para cada questão, determina a alternativa marcada ─────────────
    options = ['A', 'B', 'C', 'D', 'E']
    results = []

    for q_num, ri in enumerate(question_rows, start=1):
        fills = [grid.get((ri, ci), 0.0) for ci in range(len(col_centers))]

        best_col = int(np.argmax(fills))
        best_fill = fills[best_col]

        # Média dos outros círculos da mesma linha
        others = [f for i, f in enumerate(fills) if i != best_col]
        avg_others = np.mean(others) if others else 0.0

        # Considera marcado apenas se:
        # 1. fill absoluto >= FILL_THRESHOLD
        # 2. destaca-se pelo menos 30% acima da média dos demais
        marked = (best_fill >= FILL_THRESHOLD and
                  best_fill >= avg_others * 1.3 and
                  best_col < len(options))

        answer = options[best_col] if marked else None
        results.append({"question_number": q_num, "student_answer": answer})

    # Completa até num_questions
    while len(results) < num_questions:
        results.append({"question_number": len(results) + 1, "student_answer": None})

    # ── 6. Debug: imagem anotada ──────────────────────────────────────────
    if debug:
        dbg = img.copy()
        for cx, cy, r, f in circles:
            marked = f >= FILL_THRESHOLD
            color = (0, 200, 0) if marked else (180, 180, 180)
            cv2.circle(dbg, (cx, cy), r, color, -1 if marked else 2)
            cv2.putText(dbg, f"{f:.2f}", (cx - 18, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 0, 0) if marked else (80, 80, 80), 1)
        for ry in row_centers:
            cv2.line(dbg, (0, ry), (w, ry), (255, 100, 0), 1)
        for cx_c in col_centers:
            cv2.line(dbg, (cx_c, 0), (cx_c, h), (0, 100, 255), 1)
        if isinstance(source, str):
            debug_out = os.path.splitext(source)[0] + '_debug.jpg'
        else:
            debug_out = 'gabarito_debug.jpg'
        cv2.imwrite(debug_out, dbg)
        print(f"✓ Debug salvo em: {debug_out}")

    return results
