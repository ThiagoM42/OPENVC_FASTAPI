"""
tests/test_gabarito_reader.py
Testes unitários para a lógica de leitura de gabarito (sem FastAPI).

Execute com: pytest tests/ -v
"""

import io
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# Adiciona o diretório raiz ao path para importar app.*
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.gabarito_reader import (
    GabaritoResult,
    _classify_ink,
    _cluster_1d,
    _dark_ratio,
    _infer_option_columns,
    _infer_question_rows,
    align_gabarito,
    find_circles,
    process_image_bytes,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_gabarito_image(
    answers: list[int],  # índice 0-4 (A-E) por questão
    n_questions: int = 10,
    ink_color: tuple = (5, 5, 5),  # BGR preto
    rotation_deg: float = 0.0,
) -> bytes:
    """Gera uma imagem sintética de gabarito e retorna como bytes JPEG."""
    H, W = 900, 700
    img = np.ones((H, W, 3), dtype=np.uint8) * 240

    col_xs = [200, 290, 380, 470, 560]
    row_ys = [100 + i * 70 for i in range(n_questions)]

    # Círculos vazios impressos
    for qy in row_ys:
        for cx in col_xs:
            cv2.circle(img, (cx, qy), 22, (40, 40, 40), 2)

    # Preenche respostas
    for q, (qy, ai) in enumerate(zip(row_ys, answers)):
        cx = col_xs[ai]
        cv2.circle(img, (cx, qy), 20, ink_color, -1)
        cv2.circle(img, (cx, qy), 22, (0, 0, 0), 2)

    # Aplica rotação se necessário
    if abs(rotation_deg) > 0.1:
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), rotation_deg, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderValue=(240, 240, 240))

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    return buf.tobytes()


ANSWER_INDICES = [1, 2, 3, 0, 4, 1, 2, 0, 3, 2]  # B,C,D,A,E,B,C,A,D,C
EXPECTED       = ["B", "C", "D", "A", "E", "B", "C", "A", "D", "C"]


# ── Testes de utilitários ─────────────────────────────────────────────────────

class TestCluster1d:
    def test_empty(self):
        assert _cluster_1d([]) == []

    def test_single(self):
        result = _cluster_1d([42])
        assert result == [(42, 1)]

    def test_two_groups(self):
        vals = [10, 11, 12, 50, 51]
        result = _cluster_1d(vals, gap=5)
        assert len(result) == 2
        centers = [c for c, _ in result]
        assert centers[0] < 20
        assert centers[1] > 40

    def test_counts_are_correct(self):
        vals = [100, 101, 102, 200, 201]
        result = _cluster_1d(vals, gap=5)
        counts = [cnt for _, cnt in result]
        assert counts == [3, 2]


class TestDarkRatio:
    def _gray_with_circle(self, value: int) -> tuple[np.ndarray, int, int, int]:
        gray = np.ones((200, 200), dtype=np.uint8) * 200
        cx, cy, r = 100, 100, 30
        cv2.circle(gray, (cx, cy), r, value, -1)
        return gray, cx, cy, r

    def test_filled_circle_high_ratio(self):
        gray, cx, cy, r = self._gray_with_circle(10)  # muito escuro
        ratio = _dark_ratio(gray, cx, cy, r)
        assert ratio > 0.9

    def test_empty_circle_low_ratio(self):
        gray, cx, cy, r = self._gray_with_circle(220)  # claro
        ratio = _dark_ratio(gray, cx, cy, r)
        assert ratio < 0.1


class TestClassifyInk:
    def _bgr_img(self, b: int, g: int, r: int) -> tuple[np.ndarray, int, int, int]:
        img = np.ones((200, 200, 3), dtype=np.uint8) * 220
        cx, cy, rad = 100, 100, 30
        cv2.circle(img, (cx, cy), rad, (b, g, r), -1)
        return img, cx, cy, rad

    def test_blue_ink(self):
        img, cx, cy, r = self._bgr_img(130, 60, 60)
        assert _classify_ink(img, cx, cy, r) == "blue"

    def test_black_ink(self):
        img, cx, cy, r = self._bgr_img(10, 10, 10)
        assert _classify_ink(img, cx, cy, r) == "black"

    def test_empty_white(self):
        img, cx, cy, r = self._bgr_img(220, 220, 220)
        assert _classify_ink(img, cx, cy, r) == "none"


# ── Testes de alinhamento ─────────────────────────────────────────────────────

class TestAlignment:
    def test_no_rotation_needed(self):
        img_bytes = _make_gabarito_image(ANSWER_INDICES)
        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        _, method = align_gabarito(img)
        # Imagem já alinhada não deve rodar significativamente
        assert "sem ajuste" in method or abs(float(method.split("°")[0].split()[-1])) < 2

    def test_rotated_image_corrected(self):
        img_bytes = _make_gabarito_image(ANSWER_INDICES, rotation_deg=7.0)
        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        _, method = align_gabarito(img)
        assert method != "sem ajuste"


# ── Testes de integração (process_image_bytes) ────────────────────────────────

class TestProcessImageBytes:
    def test_black_pen_all_correct(self):
        img_bytes = _make_gabarito_image(ANSWER_INDICES, ink_color=(5, 5, 5))
        result = process_image_bytes(img_bytes)
        assert isinstance(result, GabaritoResult)
        assert result.total_questoes == len(ANSWER_INDICES)
        detected = {q.numero: q.resposta for q in result.questoes}
        for i, exp in enumerate(EXPECTED, start=1):
            assert detected.get(i) == exp, f"Q{i}: esperado {exp}, obtido {detected.get(i)}"

    def test_blue_pen_all_correct(self):
        img_bytes = _make_gabarito_image(ANSWER_INDICES, ink_color=(150, 60, 60))
        result = process_image_bytes(img_bytes)
        detected = {q.numero: q.resposta for q in result.questoes}
        for i, exp in enumerate(EXPECTED, start=1):
            assert detected.get(i) == exp, f"Q{i}: esperado {exp}, obtido {detected.get(i)}"

    def test_rotated_7deg_correct(self):
        img_bytes = _make_gabarito_image(ANSWER_INDICES, rotation_deg=7.0)
        result = process_image_bytes(img_bytes)
        detected = {q.numero: q.resposta for q in result.questoes}
        correct = sum(1 for i, exp in enumerate(EXPECTED, 1) if detected.get(i) == exp)
        assert correct >= 8, f"Esperado ≥8/10 com rotação, obteve {correct}/10"

    def test_invalid_bytes_raises(self):
        with pytest.raises(ValueError, match="inválido"):
            process_image_bytes(b"isso nao e uma imagem")

    def test_empty_bytes_raises(self):
        with pytest.raises((ValueError, Exception)):
            process_image_bytes(b"")

    def test_debug_image_returned(self):
        img_bytes = _make_gabarito_image(ANSWER_INDICES)
        result = process_image_bytes(img_bytes, return_debug_image=True)
        assert result.debug_image_b64 is not None
        assert len(result.debug_image_b64) > 100

    def test_no_debug_by_default(self):
        img_bytes = _make_gabarito_image(ANSWER_INDICES)
        result = process_image_bytes(img_bytes, return_debug_image=False)
        assert result.debug_image_b64 is None

    def test_result_to_dict_structure(self):
        img_bytes = _make_gabarito_image(ANSWER_INDICES)
        result = process_image_bytes(img_bytes)
        d = result.to_dict()
        assert "questoes" in d
        assert "total_questoes" in d
        assert "alinhamento" in d
        for q in d["questoes"]:
            assert "numero" in q
            assert "resposta" in q
            assert "multipla_marcacao" in q
            assert "marcacoes" in q
