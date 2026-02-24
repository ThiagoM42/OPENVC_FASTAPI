"""
main.py — Gabarito Reader API
Arquivo único compatível com a stack Docker existente no Portainer.

Endpoint principal:
    POST /process   → recebe imagem, retorna respostas em JSON

Recursos:
    • Caneta azul e preta
    • Correção automática de perspectiva e rotação (até ~20°)
    • Detecção de múltiplas marcações
    • Endpoint de debug com imagem anotada em base64

Uso local:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import base64
import logging
from itertools import combinations

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ════════════════════════════════════════════════════════════════════════════════
# Logging
# ════════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("gabarito_api")

# ════════════════════════════════════════════════════════════════════════════════
# Configuração
# ════════════════════════════════════════════════════════════════════════════════

FILL_THRESHOLD       = 0.65   # dark_ratio mínimo para considerar bolha preenchida
DARK_PIXEL           = 128    # limiar de pixel escuro
TARGET_HEIGHT        = 1200   # altura de trabalho (px)
OPTIONS              = ["A", "B", "C", "D", "E"]
PERSP_SKEW_THRESHOLD = 2.5    # graus — abaixo disso perspectiva é ignorada
ROTATE_THRESHOLD     = 0.5    # graus — abaixo disso não rotaciona
MAX_FILE_SIZE        = 10 * 1024 * 1024  # 10 MB

HOUGH_NORMAL    = dict(dp=1, minDist=22, param1=50, param2=17,
                       minRadius=10, maxRadius=38)
HOUGH_SENSITIVE = dict(dp=1, minDist=20, param1=40, param2=13,
                       minRadius=9,  maxRadius=40)

ALLOWED_TYPES = {
    "image/jpeg", "image/jpg", "image/png",
    "image/bmp", "image/webp", "image/tiff",
}

# ════════════════════════════════════════════════════════════════════════════════
# Alinhamento — correção de perspectiva e rotação
# ════════════════════════════════════════════════════════════════════════════════

def _order_points(pts: np.ndarray) -> np.ndarray:
    """Ordena 4 cantos: TL, TR, BR, BL."""
    pts = pts.astype(np.float32)
    s, diff = pts.sum(axis=1), np.diff(pts, axis=1).flatten()
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(diff)],
         pts[np.argmax(s)], pts[np.argmax(diff)]],
        dtype=np.float32,
    )


def _perspective_transform(img: np.ndarray, pts: np.ndarray) -> np.ndarray | None:
    rect = _order_points(pts)
    tl, tr, br, bl = rect
    maxW = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
    maxH = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))
    if maxW < 100 or maxH < 100:
        return None
    dst = np.array(
        [[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (maxW, maxH), borderValue=(245, 245, 245))


def _is_significant_perspective(pts: np.ndarray) -> bool:
    rect = _order_points(pts)
    tl, tr, br, bl = rect
    top_angle = abs(np.degrees(np.arctan2(tr[1] - tl[1], tr[0] - tl[0])))
    bot_angle = abs(np.degrees(np.arctan2(br[1] - bl[1], br[0] - bl[0])))
    lh, rh = np.linalg.norm(bl - tl), np.linalg.norm(br - tr)
    tw, bw = np.linalg.norm(tr - tl), np.linalg.norm(br - bl)
    h_ratio = abs(lh - rh) / max(lh, rh, 1)
    w_ratio = abs(tw - bw) / max(tw, bw, 1)
    return (max(top_angle, bot_angle) > PERSP_SKEW_THRESHOLD
            or h_ratio > 0.03 or w_ratio > 0.03)


def _find_document_corners(gray: np.ndarray) -> np.ndarray | None:
    blurred   = cv2.GaussianBlur(gray, (7, 7), 0)
    img_area  = gray.shape[0] * gray.shape[1]

    for morph_k, morph_i, eps in [
        (5, 2, 0.020), (3, 1, 0.025), (5, 3, 0.015), (7, 2, 0.030)
    ]:
        _, binary = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_k, morph_k))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel,
                                  iterations=morph_i)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            if cv2.contourArea(cnt) < img_area * 0.08:
                continue
            peri   = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, eps * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2).astype(np.float32)
    return None


def _estimate_skew_lines(gray: np.ndarray) -> float | None:
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60,
                             minLineLength=50, maxLineGap=30)
    if lines is None:
        return None
    angles, weights = [], []
    for x1, y1, x2, y2 in lines[:, 0]:
        if abs(x2 - x1) < 1:
            continue
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if -20 < angle < 20:
            angles.append(angle)
            weights.append(np.hypot(x2 - x1, y2 - y1))
    return float(np.average(angles, weights=weights)) if angles else None


def _estimate_skew_projection(gray: np.ndarray, steps: int = 81) -> float:
    h, w = gray.shape
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    best_angle, best_score = 0.0, -1.0
    for angle in np.linspace(-20, 20, steps):
        M   = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        rot = cv2.warpAffine(binary, M, (w, h))
        score = float(rot.sum(axis=1).astype(float).var())
        if score > best_score:
            best_score, best_angle = score, angle
    return best_angle


def _rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < ROTATE_THRESHOLD:
        return img
    h, w = img.shape[:2]
    M    = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    return cv2.warpAffine(img, M, (new_w, new_h),
                          flags=cv2.INTER_LINEAR,
                          borderValue=(245, 245, 245))


def align_image(img: np.ndarray) -> tuple[np.ndarray, str]:
    """
    Alinha a imagem em cascata:
      1. Perspectiva (4 cantos detectados)
      2. Rotação via HoughLinesP
      3. Rotação via projeção de pixels (fallback robusto)

    Retorna (img_alinhada, metodo_descricao).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]

    # ── Método 1: perspectiva ────────────────────────────────────────────
    corners = _find_document_corners(gray)
    if corners is not None:
        area_ratio = cv2.contourArea(corners) / (h * w)
        if area_ratio > 0.08 and _is_significant_perspective(corners):
            warped = _perspective_transform(img, corners)
            if warped is not None and warped.shape[0] > 200 and warped.shape[1] > 200:
                return warped, f"perspectiva ({area_ratio:.0%})"

    # ── Método 2: rotação via Hough ──────────────────────────────────────
    angle_hough = _estimate_skew_lines(gray)
    if angle_hough is not None and abs(angle_hough) > ROTATE_THRESHOLD:
        return _rotate_image(img, angle_hough), f"rotação {angle_hough:+.1f}° (Hough)"

    # ── Método 3: rotação via projeção ───────────────────────────────────
    angle_proj = _estimate_skew_projection(gray)
    if abs(angle_proj) > ROTATE_THRESHOLD:
        return _rotate_image(img, angle_proj), f"rotação {angle_proj:+.1f}° (projeção)"

    return img, "sem ajuste"


# ════════════════════════════════════════════════════════════════════════════════
# Análise de pixel — detecção de tinta
# ════════════════════════════════════════════════════════════════════════════════

def _dark_ratio(gray: np.ndarray, cx: int, cy: int, r: int) -> float:
    """Proporção de pixels escuros dentro do raio interno da bolha."""
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(mask, (cx, cy), max(4, int(r * 0.75)), 255, -1)
    pix = gray[mask > 0]
    return float(np.sum(pix < DARK_PIXEL)) / len(pix) if len(pix) > 0 else 0.0


def _classify_ink(img_bgr: np.ndarray, cx: int, cy: int, r: int) -> str:
    """
    Classifica a cor da tinta na bolha.
    Retorna: 'blue' | 'black' | 'none'
    """
    mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (cx, cy), max(4, int(r * 0.75)), 255, -1)
    pix = img_bgr[mask > 0].astype(float)
    if len(pix) == 0:
        return "none"
    b, g, r_ch = pix[:, 0].mean(), pix[:, 1].mean(), pix[:, 2].mean()
    brightness = (b + g + r_ch) / 3
    if brightness > 160:
        return "none"
    # Caneta azul: canal B dominante
    if b - max(g, r_ch) > 18:
        return "blue"
    # Caneta preta: todos os canais escuros e equilibrados
    if brightness < 150 and (max(b, g, r_ch) - min(b, g, r_ch)) < 50:
        return "black"
    return "none"


# ════════════════════════════════════════════════════════════════════════════════
# Detecção de círculos (HoughCircles com dois níveis de sensibilidade)
# ════════════════════════════════════════════════════════════════════════════════

def _hough(gray: np.ndarray, params: dict) -> list:
    blurred = cv2.medianBlur(gray, 5)
    cs = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, **params)
    if cs is None:
        return []
    return [
        (int(cx), int(cy), int(r), _dark_ratio(gray, int(cx), int(cy), int(r)))
        for cx, cy, r in np.round(cs[0]).astype(int)
    ]


def find_circles(gray: np.ndarray) -> list:
    """
    Detecta círculos em dois níveis de sensibilidade.
    Se o nível normal encontrar menos de 20, combina com o sensível.
    """
    circles = _hough(gray, HOUGH_NORMAL)
    if len(circles) < 20:
        for c in _hough(gray, HOUGH_SENSITIVE):
            cx, cy, r, d = c
            if not any(abs(cx - e[0]) < 15 and abs(cy - e[1]) < 15
                       for e in circles):
                circles.append(c)
    return circles


# ════════════════════════════════════════════════════════════════════════════════
# Inferência da grade (colunas A-E e linhas de questões)
# ════════════════════════════════════════════════════════════════════════════════

def _cluster_1d(vals: list, gap: int = 18) -> list[tuple[int, int]]:
    """
    Agrupa valores por proximidade (preserva multiplicidade).
    Retorna [(centro, contagem), ...].
    """
    vals = sorted(int(v) for v in vals)
    if not vals:
        return []
    groups = [[vals[0]]]
    for v in vals[1:]:
        if v - groups[-1][-1] <= gap:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [(int(np.mean(g)), len(g)) for g in groups]


def _infer_option_columns(circles: list, n: int = 5) -> list[int]:
    """Seleciona as n colunas X das opções com espaçamento mais uniforme."""
    xs = _cluster_1d([cx for cx, cy, r, d in circles], gap=18)
    frequent = [(c, cnt) for c, cnt in xs if cnt >= 4]
    if not frequent:
        frequent = sorted(xs, key=lambda x: x[1], reverse=True)[:n + 2]

    centers = sorted(c for c, _ in frequent)
    if len(centers) == n:
        return centers

    best, best_score = None, float("inf")
    for combo in combinations(centers, n):
        combo = sorted(combo)
        if combo[-1] - combo[0] < 80:
            continue
        score = float(np.std([combo[i + 1] - combo[i] for i in range(n - 1)]))
        if score < best_score:
            best_score, best = score, list(combo)
    return best or centers[:n]


def _infer_question_rows(circles: list) -> list[int]:
    """
    Retorna os Y centrais das linhas de questões.
    Filtra cabeçalho e ruído via consistência do espaçamento.
    """
    ys_all     = _cluster_1d([cy for cx, cy, r, d in circles], gap=18)
    candidates = sorted(yc for yc, cnt in ys_all if cnt >= 3)
    if len(candidates) <= 3:
        return candidates

    spacings  = [candidates[i + 1] - candidates[i]
                 for i in range(len(candidates) - 1)]
    median_sp = float(np.median(spacings))
    tol       = median_sp * 0.40

    best_start, best_len = 0, 1
    cur_start,  cur_len  = 0, 1
    for i, sp in enumerate(spacings):
        if abs(sp - median_sp) <= tol:
            cur_len += 1
            if cur_len > best_len:
                best_len, best_start = cur_len, cur_start
        else:
            cur_start, cur_len = i + 1, 1

    return candidates[best_start: best_start + best_len]


# ════════════════════════════════════════════════════════════════════════════════
# Pipeline principal de leitura
# ════════════════════════════════════════════════════════════════════════════════

def read_gabarito(image_bytes: bytes, debug: bool = False) -> dict:
    """
    Processa os bytes de uma imagem e retorna o resultado do gabarito.

    Parâmetros
    ----------
    image_bytes : bytes brutos da imagem (JPEG, PNG, BMP, WebP…).
    debug       : se True, inclui PNG anotado em base64 no retorno.

    Retorna
    -------
    dict com as chaves:
      - alinhamento        : str — método aplicado
      - total_questoes     : int
      - total_respondidas  : int
      - total_nao_resp     : int
      - total_multiplas    : int
      - questoes           : list[dict] com campos:
            question_number  : int
            student_answer   : str | list[str] | None
            multipla_marcacao: bool
            marcacoes        : list[str]
      - debug_image        : str | None — PNG base64 (somente se debug=True)

    Raises
    ------
    ValueError : imagem inválida ou grade não detectada.
    """
    # ── Decodifica ────────────────────────────────────────────────────────
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Formato de imagem inválido ou arquivo corrompido.")

    # ── Redimensiona para altura padrão ───────────────────────────────────
    scale = TARGET_HEIGHT / img.shape[0]
    img   = cv2.resize(img, (int(img.shape[1] * scale), TARGET_HEIGHT))

    # ── Alinha ────────────────────────────────────────────────────────────
    img, align_method = align_image(img)

    # Re-escala após alinhamento (perspectiva pode alterar dimensões)
    scale2 = TARGET_HEIGHT / img.shape[0]
    if abs(scale2 - 1.0) > 0.01:
        img = cv2.resize(img, (int(img.shape[1] * scale2), TARGET_HEIGHT))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── Detecta círculos e grade ──────────────────────────────────────────
    circles = find_circles(gray)
    if not circles:
        raise ValueError("Nenhuma bolha detectada na imagem.")

    option_xs   = _infer_option_columns(circles)
    question_ys = _infer_question_rows(circles)

    if not option_xs or not question_ys:
        raise ValueError("Não foi possível identificar a grade do gabarito.")

    tol_x, tol_y = 25, 28

    # ── Classifica cada célula ────────────────────────────────────────────
    questoes = []
    for q_idx, qy in enumerate(question_ys, start=1):
        marcacoes: list[str] = []

        for opt_idx, ox in enumerate(option_xs):
            candidates = [
                (cx, cy, r, d) for cx, cy, r, d in circles
                if abs(cx - ox) <= tol_x and abs(cy - qy) <= tol_y
            ]
            if not candidates:
                continue

            bx, by, br, bd = max(candidates, key=lambda c: c[3])
            if bd >= FILL_THRESHOLD and _classify_ink(img, bx, by, br) in ("blue", "black"):
                marcacoes.append(OPTIONS[opt_idx])

        multipla = len(marcacoes) > 1
        resposta = marcacoes[0] if len(marcacoes) == 1 else None

        questoes.append({
            "question_number":   q_idx,
            "student_answer":    marcacoes if multipla else resposta,
            "multipla_marcacao": multipla,
            "marcacoes":         marcacoes,
        })

        if q_idx >= 25:
            break

    # Remove questões em branco do final
    while questoes and not questoes[-1]["marcacoes"]:
        questoes.pop()

    total_resp     = sum(1 for q in questoes if len(q["marcacoes"]) == 1)
    total_nao_resp = sum(1 for q in questoes if not q["marcacoes"])
    total_mult     = sum(1 for q in questoes if q["multipla_marcacao"])

    # ── Debug image (opcional) ────────────────────────────────────────────
    debug_b64 = None
    if debug:
        dbg = img.copy()
        for i, x in enumerate(option_xs):
            cv2.line(dbg, (x, 0), (x, dbg.shape[0]), (0, 220, 220), 1)
            cv2.putText(dbg, OPTIONS[i], (x - 10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 140, 200), 2)
        for i, y in enumerate(question_ys):
            cv2.line(dbg, (0, y), (dbg.shape[1], y), (0, 200, 100), 1)
            cv2.putText(dbg, f"Q{i+1}", (5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 160, 80), 1)
        for cx, cy, r, dr in circles:
            ink    = _classify_ink(dbg, cx, cy, r)
            filled = dr >= FILL_THRESHOLD and ink in ("blue", "black")
            color  = (0, 0, 220) if filled else (60, 180, 60)
            cv2.circle(dbg, (cx, cy), r, color, 3 if filled else 1)
            cv2.putText(dbg, f"{dr:.2f}", (cx - 14, cy + r + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1)
        ok, buf = cv2.imencode(".png", dbg)
        if ok:
            debug_b64 = base64.b64encode(buf.tobytes()).decode()

    logger.info(
        "Alinhamento: %s | %d questão(ões) | %d respondida(s)",
        align_method, len(questoes), total_resp,
    )

    return {
        "alinhamento":       align_method,
        "total_questoes":    len(questoes),
        "total_respondidas": total_resp,
        "total_nao_resp":    total_nao_resp,
        "total_multiplas":   total_mult,
        "questoes":          questoes,
        **({"debug_image": debug_b64} if debug_b64 else {}),
    }


# ════════════════════════════════════════════════════════════════════════════════
# FastAPI
# ════════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Gabarito Reader API",
    description=(
        "Leitura automática de gabaritos de múltipla escolha.\n\n"
        "• Caneta azul e preta\n"
        "• Correção de perspectiva e rotação\n"
        "• Múltiplas marcações detectadas\n"
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Healthcheck ───────────────────────────────────────────────────────────────

@app.get("/", summary="Healthcheck")
def health():
    """Verifica se a API está operacional."""
    return {"status": "Gabarito API running 🚀", "version": "2.0.0"}


# ── Endpoint principal ────────────────────────────────────────────────────────

@app.post("/process", summary="Analisa gabarito")
async def process_image(
    file: UploadFile = File(..., description="Imagem do gabarito (JPEG, PNG, BMP, WebP)"),
):
    """
    Recebe uma imagem de gabarito e retorna as respostas detectadas.

    Resposta:
    ```json
    {
      "alinhamento": "perspectiva (68%)",
      "total_questoes": 10,
      "total_respondidas": 9,
      "total_nao_resp": 1,
      "total_multiplas": 0,
      "questoes": [
        {
          "question_number": 1,
          "student_answer": "B",
          "multipla_marcacao": false,
          "marcacoes": ["B"]
        }
      ]
    }
    ```
    """
    # Valida content-type
    content_type = (file.content_type or "").lower().split(";")[0].strip()
    if content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Tipo não suportado: '{content_type}'. Use JPEG, PNG, BMP ou WebP.",
        )

    contents = await file.read()

    # Valida tamanho
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Arquivo vazio.")
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Arquivo muito grande ({len(contents)/1024/1024:.1f} MB). Máximo: 10 MB.",
        )

    try:
        result = read_gabarito(contents, debug=False)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Erro inesperado: %s", e)
        raise HTTPException(status_code=500, detail="Erro interno ao processar a imagem.")

    return JSONResponse(content=result)


# ── Endpoint de debug ─────────────────────────────────────────────────────────

@app.post("/process/debug", summary="Analisa gabarito com imagem de debug")
async def process_image_debug(
    file: UploadFile = File(..., description="Imagem do gabarito"),
):
    """
    Igual a `POST /process`, mas inclui o campo `debug_image`:
    um **PNG em base64** com os círculos e a grade anotados.

    Para exibir a imagem de debug no frontend:
    ```html
    <img src="data:image/png;base64,{{ debug_image }}" />
    ```
    """
    content_type = (file.content_type or "").lower().split(";")[0].strip()
    if content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail=f"Tipo não suportado: '{content_type}'.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Arquivo vazio.")

    try:
        result = read_gabarito(contents, debug=True)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Erro inesperado: %s", e)
        raise HTTPException(status_code=500, detail="Erro interno ao processar a imagem.")

    return JSONResponse(content=result)