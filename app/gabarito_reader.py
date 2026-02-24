"""
gabarito_reader.py — Lógica central de leitura de gabaritos.

Responsabilidades:
  • Receber bytes de imagem (sem depender de disco)
  • Alinhar automaticamente (perspectiva / rotação)
  • Detectar bolhas preenchidas (caneta azul ou preta)
  • Retornar resultado estruturado como dict Python

Sem dependências de framework — pode ser usada em FastAPI, Flask, CLI, etc.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional

import cv2
import numpy as np

# ════════════════════════════════════════════════════════════════════════════════
# Configuração
# ════════════════════════════════════════════════════════════════════════════════

FILL_THRESHOLD       = 0.65   # dark_ratio ≥ → bolha preenchida
DARK_PIXEL           = 128    # limiar "pixel escuro"
TARGET_HEIGHT        = 1200   # altura de trabalho (px)
OPTIONS              = ["A", "B", "C", "D", "E"]
PERSP_SKEW_THRESHOLD = 2.5    # graus — abaixo disso perspectiva é ignorada
ROTATE_THRESHOLD     = 0.5    # graus — abaixo disso não rotaciona

HOUGH_NORMAL = dict(dp=1, minDist=22, param1=50, param2=17, minRadius=10, maxRadius=38)
HOUGH_SENSITIVE = dict(dp=1, minDist=20, param1=40, param2=13, minRadius=9, maxRadius=40)


# ════════════════════════════════════════════════════════════════════════════════
# Modelos de resultado
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class Questao:
    numero: int
    resposta: Optional[str]          # "A" | "B" | ... | None
    multipla_marcacao: bool = False   # True se mais de uma bolha preenchida
    marcacoes: list[str] = field(default_factory=list)  # todas as marcações


@dataclass
class GabaritoResult:
    questoes: list[Questao]
    total_questoes: int
    total_respondidas: int
    total_nao_respondidas: int
    total_multiplas: int
    alinhamento: str          # descrição do método de alinhamento aplicado
    debug_image_b64: Optional[str] = None  # PNG em base64 (somente se solicitado)

    def to_dict(self) -> dict:
        return {
            "alinhamento": self.alinhamento,
            "total_questoes": self.total_questoes,
            "total_respondidas": self.total_respondidas,
            "total_nao_respondidas": self.total_nao_respondidas,
            "total_multiplas": self.total_multiplas,
            "questoes": [
                {
                    "numero": q.numero,
                    "resposta": q.resposta,
                    "multipla_marcacao": q.multipla_marcacao,
                    "marcacoes": q.marcacoes,
                }
                for q in self.questoes
            ],
            **({"debug_image": self.debug_image_b64} if self.debug_image_b64 else {}),
        }


# ════════════════════════════════════════════════════════════════════════════════
# Alinhamento
# ════════════════════════════════════════════════════════════════════════════════

def _order_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.astype(np.float32)
    s, diff = pts.sum(axis=1), np.diff(pts, axis=1).flatten()
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(diff)],
         pts[np.argmax(s)], pts[np.argmax(diff)]],
        dtype=np.float32,
    )


def _perspective_transform(img: np.ndarray, pts: np.ndarray) -> Optional[np.ndarray]:
    rect = _order_points(pts)
    tl, tr, br, bl = rect
    maxW = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
    maxH = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))
    if maxW < 100 or maxH < 100:
        return None
    dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype=np.float32)
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
    return max(top_angle, bot_angle) > PERSP_SKEW_THRESHOLD or h_ratio > 0.03 or w_ratio > 0.03


def _find_document_corners(gray: np.ndarray) -> Optional[np.ndarray]:
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    h, w = gray.shape
    img_area = h * w
    for morph_k, morph_i, eps in [(5, 2, 0.02), (3, 1, 0.025), (5, 3, 0.015), (7, 2, 0.03)]:
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_k, morph_k))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=morph_i)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            if cv2.contourArea(cnt) < img_area * 0.08:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, eps * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2).astype(np.float32)
    return None


def _estimate_skew_lines(gray: np.ndarray) -> Optional[float]:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
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
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    best_angle, best_score = 0.0, -1.0
    for angle in np.linspace(-20, 20, steps):
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        rot = cv2.warpAffine(binary, M, (w, h))
        score = float(rot.sum(axis=1).astype(float).var())
        if score > best_score:
            best_score, best_angle = score, angle
    return best_angle


def _rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < ROTATE_THRESHOLD:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
    new_w, new_h = int(h * sin_a + w * cos_a), int(h * cos_a + w * sin_a)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    return cv2.warpAffine(img, M, (new_w, new_h),
                          flags=cv2.INTER_LINEAR, borderValue=(245, 245, 245))


def align_gabarito(img: np.ndarray) -> tuple[np.ndarray, str]:
    """
    Alinha o gabarito em cascata:
      1. Correção de perspectiva (4 cantos)
      2. Rotação via HoughLinesP
      3. Rotação via projeção de pixels (fallback robusto)

    Retorna (img_alinhada, descricao_metodo).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]

    corners = _find_document_corners(gray)
    if corners is not None:
        area_ratio = cv2.contourArea(corners) / (h * w)
        if area_ratio > 0.08 and _is_significant_perspective(corners):
            warped = _perspective_transform(img, corners)
            if warped is not None and warped.shape[0] > 200 and warped.shape[1] > 200:
                return warped, f"perspectiva ({area_ratio:.0%})"

    angle_hough = _estimate_skew_lines(gray)
    if angle_hough is not None and abs(angle_hough) > ROTATE_THRESHOLD:
        return _rotate_image(img, angle_hough), f"rotação {angle_hough:+.1f}° (Hough)"

    angle_proj = _estimate_skew_projection(gray)
    if abs(angle_proj) > ROTATE_THRESHOLD:
        return _rotate_image(img, angle_proj), f"rotação {angle_proj:+.1f}° (projeção)"

    return img, "sem ajuste"


# ════════════════════════════════════════════════════════════════════════════════
# Carregamento a partir de bytes (sem disco)
# ════════════════════════════════════════════════════════════════════════════════

def load_image_from_bytes(data: bytes) -> np.ndarray:
    """
    Decodifica bytes de imagem → ndarray BGR, alinhado e redimensionado.
    Aceita JPEG, PNG, BMP e qualquer formato suportado pelo OpenCV.

    Raises:
        ValueError: se os bytes não formam uma imagem válida.
    """
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Não foi possível decodificar a imagem. Verifique o formato.")

    # Redimensiona para altura padrão
    scale = TARGET_HEIGHT / img.shape[0]
    img = cv2.resize(img, (int(img.shape[1] * scale), TARGET_HEIGHT))

    # Alinha
    img, _ = align_gabarito(img)

    # Re-escala após alinhamento (perspectiva pode alterar dimensões)
    scale2 = TARGET_HEIGHT / img.shape[0]
    if abs(scale2 - 1.0) > 0.01:
        img = cv2.resize(img, (int(img.shape[1] * scale2), TARGET_HEIGHT))

    return img


# ════════════════════════════════════════════════════════════════════════════════
# Análise de pixel
# ════════════════════════════════════════════════════════════════════════════════

def _dark_ratio(gray: np.ndarray, cx: int, cy: int, r: int) -> float:
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(mask, (cx, cy), max(4, int(r * 0.75)), 255, -1)
    pix = gray[mask > 0]
    return float(np.sum(pix < DARK_PIXEL)) / len(pix) if len(pix) > 0 else 0.0


def _classify_ink(img_bgr: np.ndarray, cx: int, cy: int, r: int) -> str:
    """Retorna 'blue', 'black' ou 'none'."""
    mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (cx, cy), max(4, int(r * 0.75)), 255, -1)
    pix = img_bgr[mask > 0].astype(float)
    if len(pix) == 0:
        return "none"
    b, g, r_ch = pix[:, 0].mean(), pix[:, 1].mean(), pix[:, 2].mean()
    brightness = (b + g + r_ch) / 3
    if brightness > 160:
        return "none"
    if b - max(g, r_ch) > 18:
        return "blue"
    if brightness < 150 and (max(b, g, r_ch) - min(b, g, r_ch)) < 50:
        return "black"
    return "none"


# ════════════════════════════════════════════════════════════════════════════════
# Detecção de círculos
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
    circles = _hough(gray, HOUGH_NORMAL)
    if len(circles) < 20:
        for c in _hough(gray, HOUGH_SENSITIVE):
            cx, cy, r, d = c
            if not any(abs(cx - e[0]) < 15 and abs(cy - e[1]) < 15 for e in circles):
                circles.append(c)
    return circles


# ════════════════════════════════════════════════════════════════════════════════
# Inferência de grade (colunas A-E e linhas de questões)
# ════════════════════════════════════════════════════════════════════════════════

def _cluster_1d(vals: list, gap: int = 18) -> list[tuple[int, int]]:
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
    xs = _cluster_1d([cx for cx, cy, r, d in circles], gap=18)
    frequent = [(c, cnt) for c, cnt in xs if cnt >= 4] or sorted(xs, key=lambda x: x[1], reverse=True)[:n + 2]
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
    ys_all = _cluster_1d([cy for cx, cy, r, d in circles], gap=18)
    candidates = sorted(yc for yc, cnt in ys_all if cnt >= 3)
    if len(candidates) <= 3:
        return candidates
    spacings = [candidates[i + 1] - candidates[i] for i in range(len(candidates) - 1)]
    median_sp = float(np.median(spacings))
    tol = median_sp * 0.40
    best_start, best_len, cur_start, cur_len = 0, 1, 0, 1
    for i, sp in enumerate(spacings):
        if abs(sp - median_sp) <= tol:
            cur_len += 1
            if cur_len > best_len:
                best_len, best_start = cur_len, cur_start
        else:
            cur_start, cur_len = i + 1, 1
    return candidates[best_start: best_start + best_len]


# ════════════════════════════════════════════════════════════════════════════════
# Pipeline principal
# ════════════════════════════════════════════════════════════════════════════════

def process_image_bytes(
    image_bytes: bytes,
    return_debug_image: bool = False,
) -> GabaritoResult:
    """
    Processa bytes de uma imagem de gabarito e retorna um GabaritoResult.

    Args:
        image_bytes:        Conteúdo binário da imagem (JPEG, PNG…).
        return_debug_image: Se True, inclui PNG anotado em base64 no resultado.

    Returns:
        GabaritoResult com todas as questões e metadados.

    Raises:
        ValueError: imagem inválida ou sem bolhas detectadas.
    """
    # ── Carrega e alinha ────────────────────────────────────────────────────
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    raw = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if raw is None:
        raise ValueError("Formato de imagem inválido.")

    scale = TARGET_HEIGHT / raw.shape[0]
    raw = cv2.resize(raw, (int(raw.shape[1] * scale), TARGET_HEIGHT))
    img, align_method = align_gabarito(raw)
    scale2 = TARGET_HEIGHT / img.shape[0]
    if abs(scale2 - 1.0) > 0.01:
        img = cv2.resize(img, (int(img.shape[1] * scale2), TARGET_HEIGHT))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── Detecta círculos e infere grade ────────────────────────────────────
    circles = find_circles(gray)
    if not circles:
        raise ValueError("Nenhuma bolha detectada na imagem.")

    option_xs = _infer_option_columns(circles)
    question_ys = _infer_question_rows(circles)

    if not option_xs or not question_ys:
        raise ValueError("Não foi possível identificar a grade do gabarito.")

    tol_x, tol_y = 25, 28

    # ── Lê cada questão ─────────────────────────────────────────────────────
    questoes: list[Questao] = []

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
        resposta = marcacoes[0] if len(marcacoes) == 1 else (None if not marcacoes else None)

        questoes.append(Questao(
            numero=q_idx,
            resposta=resposta,
            multipla_marcacao=multipla,
            marcacoes=marcacoes,
        ))

        if q_idx >= 25:
            break

    # Remove questões em branco do final
    while questoes and not questoes[-1].marcacoes:
        questoes.pop()

    total_respondidas   = sum(1 for q in questoes if len(q.marcacoes) == 1)
    total_nao_resp      = sum(1 for q in questoes if not q.marcacoes)
    total_multiplas     = sum(1 for q in questoes if q.multipla_marcacao)

    # ── Debug image (opcional) ──────────────────────────────────────────────
    debug_b64 = None
    if return_debug_image:
        debug_img = img.copy()
        for i, x in enumerate(option_xs):
            cv2.line(debug_img, (x, 0), (x, debug_img.shape[0]), (0, 220, 220), 1)
            cv2.putText(debug_img, OPTIONS[i], (x - 10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 140, 200), 2)
        for i, y in enumerate(question_ys):
            cv2.line(debug_img, (0, y), (debug_img.shape[1], y), (0, 200, 100), 1)
            cv2.putText(debug_img, f"Q{i+1}", (5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 160, 80), 1)
        for cx, cy, r, dr in circles:
            ink = _classify_ink(debug_img, cx, cy, r)
            filled = dr >= FILL_THRESHOLD and ink in ("blue", "black")
            color = (0, 0, 220) if filled else (60, 180, 60)
            cv2.circle(debug_img, (cx, cy), r, color, 3 if filled else 1)
            cv2.putText(debug_img, f"{dr:.2f}", (cx - 14, cy + r + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1)
        ok, buf = cv2.imencode(".png", debug_img)
        if ok:
            debug_b64 = base64.b64encode(buf.tobytes()).decode()

    return GabaritoResult(
        questoes=questoes,
        total_questoes=len(questoes),
        total_respondidas=total_respondidas,
        total_nao_respondidas=total_nao_resp,
        total_multiplas=total_multiplas,
        alinhamento=align_method,
        debug_image_b64=debug_b64,
    )
