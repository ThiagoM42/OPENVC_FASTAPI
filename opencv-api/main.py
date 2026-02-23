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


def _fill_ratio(thresh: np.ndarray, cx: int, cy: int, r: int) -> float:
    """
    Proporção de pixels escuros (preenchidos) dentro do círculo.
    Usa 50% do raio para ignorar a borda do círculo e capturar só o interior.
    """
    inner = max(int(r * 0.50), 5)
    mask = np.zeros(thresh.shape, dtype=np.uint8)
    cv2.circle(mask, (cx, cy), inner, 255, -1)
    area = np.sum(mask > 0)
    return float(np.sum((thresh > 0) & (mask > 0))) / area if area else 0.0


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
    student_answer é None quando o círculo não está preenchido ou está fraco demais.
    """

    # ── Parâmetros de detecção ────────────────────────────────────────────
    NUM_OPTIONS      = 5      # colunas de alternativas (A-E)
    LEFT_MARGIN_FRAC = 0.30   # ignora círculos à esquerda de 30% da largura
                              # (elimina ruídos dos números das questões na lateral)

    # Um círculo é considerado MARCADO quando:
    # 1. Seu fill ratio absoluto >= FILL_THRESHOLD  (elimina não-marcados mesmo com ratio alto)
    # 2. Seu fill ratio >= RELATIVE_FACTOR × média dos demais da mesma linha
    #    (exige que o marcado se destaque dos outros)
    FILL_THRESHOLD  = 0.15
    RELATIVE_FACTOR = 1.6

    # ── 1. Carrega e pré-processa ─────────────────────────────────────────
    img = _decode_image(source)
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # ── 2. Detecta círculos com parâmetros adaptativos à resolução ────────
    min_r    = max(10, int(h * 0.010))   # ~1.0% da altura
    max_r    = max(20, int(h * 0.032))   # ~3.2% da altura
    min_dist = max(15, int(min_r * 1.2))

    raw = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT,
        dp=1, minDist=min_dist,
        param1=50, param2=28,
        minRadius=min_r, maxRadius=max_r
    )
    if raw is None:
        raise RuntimeError(
            "Nenhum círculo detectado. Verifique a qualidade e o formato da imagem."
        )

    # Remove círculos da margem esquerda (ruídos dos números das questões)
    x_min = int(w * LEFT_MARGIN_FRAC)
    circles = [
        (int(cx), int(cy), int(r))
        for cx, cy, r in np.round(raw[0]).astype(int)
        if cx > x_min
    ]

    # ── 3. Clusteriza linhas (Y) ──────────────────────────────────────────
    tol_row = int(h * 0.020)
    row_centers = _cluster([cy for cx, cy, r in circles], tol=tol_row)

    # ── 4. Clusteriza colunas (X) e seleciona as NUM_OPTIONS mais populadas
    col_centers_all = _cluster([cx for cx, cy, r in circles], tol=int(w * 0.020))

    col_pop: dict = {i: 0 for i in range(len(col_centers_all))}
    for cx, cy, r in circles:
        ci = min(range(len(col_centers_all)), key=lambda i: abs(cx - col_centers_all[i]))
        col_pop[ci] += 1

    top_cols = sorted(
        sorted(col_pop, key=col_pop.get, reverse=True)[:NUM_OPTIONS]
    )
    col_centers = [col_centers_all[i] for i in top_cols]

    # Tolerância de coluna = metade do espaçamento entre colunas
    # Garante que círculos levemente deslocados ainda sejam capturados
    if len(col_centers) > 1:
        col_spacing = np.mean([
            col_centers[i + 1] - col_centers[i]
            for i in range(len(col_centers) - 1)
        ])
    else:
        col_spacing = w * 0.10
    tol_col = int(col_spacing * 0.50)

    # ── 5. Descarta linhas de cabeçalho ───────────────────────────────────
    # O cabeçalho (rótulos A B C D E) fica separado das questões por um gap
    # bem maior que o espaçamento uniforme entre as questões.
    if len(row_centers) > 1:
        gaps = [row_centers[i + 1] - row_centers[i] for i in range(len(row_centers) - 1)]
        median_gap = float(np.median(gaps))
        header_indices = {i for i, g in enumerate(gaps) if g > median_gap * 1.5}
    else:
        header_indices = set()

    question_rows = [ry for i, ry in enumerate(row_centers) if i not in header_indices]
    if num_questions is not None:
        question_rows = question_rows[:num_questions]

    # ── 6. Determina a resposta de cada questão ───────────────────────────
    options = ['A', 'B', 'C', 'D', 'E']
    results = []

    for q_num, ry in enumerate(question_rows, start=1):
        fills = []
        for cx_col in col_centers:
            # Candidatos na célula (linha ry, coluna cx_col)
            candidates = [
                (cx, cy, r) for cx, cy, r in circles
                if abs(cy - ry) <= tol_row and abs(cx - cx_col) <= tol_col
            ]
            if candidates:
                # Usa o círculo de MAIOR RAIO: as bolhas do gabarito são as maiores;
                # detecções menores dentro delas são ruídos do HoughCircles.
                cx, cy, r = max(candidates, key=lambda c: c[2])
                f = _fill_ratio(thresh, cx, cy, r)
            else:
                f = 0.0
            fills.append(f)

        best_col  = int(np.argmax(fills))
        best_fill = fills[best_col]
        others    = [f for i, f in enumerate(fills) if i != best_col]
        avg_others = float(np.mean(others)) if others else 0.0

        # Círculo marcado: fill absoluto suficiente E destaque relativo
        marked = (
            best_fill >= FILL_THRESHOLD
            and (avg_others == 0 or best_fill >= avg_others * RELATIVE_FACTOR)
            and best_col < len(options)
        )

        results.append({
            "question_number": q_num,
            "student_answer": options[best_col] if marked else None
        })

    return results
