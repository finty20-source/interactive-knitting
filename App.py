import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict

# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---

def split_total_into_steps(total, steps):
    """Разбить число total на steps частей (разница между частями ≤1)."""
    base = total // steps
    rem = total % steps
    parts = [base + (1 if i < rem else 0) for i in range(steps)]
    return parts

def allowed_even_rows(start_row, end_row, rows_total):
    """Вернуть список чётных рядов в пределах (не раньше 6-го и не позже rows_total-2)."""
    rows = []
    for r in range(max(6, start_row), min(end_row, rows_total - 1) + 1):
        if r % 2 == 0:
            rows.append(r)
    return rows

# --- ГОРЛОВИНА ---
def calc_round_neckline(total_stitches: int, total_rows: int, start_row: int, rows_total: int, straight_percent: float = 0.05):
    """
    Горловина:
    - первый шаг: 60% петель,
    - верхние straight_percent*глубины (но ≥2 ряда) — прямо,
    - остальные петли равномерно по чётным рядам,
    - последние 2 убавочных ряда ≤1 петли,
    - лишние петли раскидываются вниз (кроме первого шага).
    """
    if total_stitches <= 0 or total_rows <= 0:
        return []

    # первый шаг (60%)
    first_dec = int(round(total_stitches * 0.60))
    rest = total_stitches - first_dec

    # верхняя прямая часть
    straight_rows = max(2, int(round(total_rows * straight_percent)))
    neck_end_by_depth = start_row + total_rows - 1 - straight_rows
    effective_end = min(neck_end_by_depth, rows_total - 2)

    # допустимые ряды
    rows = allowed_even_rows(start_row, effective_end, rows_total)
    if not rows:
        return []

    actions = []
    # первый шаг
    actions.append((rows[0], f"-{first_dec} п. горловина (середина, разделение на плечи)"))

    # если нечего распределять
    if rest <= 0 or len(rows) == 1:
        return actions

    rest_rows = rows[1:]
    steps = min(len(rest_rows), rest)

    if steps == 2 and rest > 2 and len(rest_rows) >= 3:
        steps = 3

    idxs = np.linspace(0, len(rest_rows) - 1, num=steps, dtype=int)
    chosen = [rest_rows[i] for i in idxs]
    parts = split_total_into_steps(rest, steps)

    # сглаживание последних двух шагов
    if steps >= 2:
        over = 0
        last_idxs = [steps - 2, steps - 1]
        for i in last_idxs:
            if parts[i] > 1:
                over += parts[i] - 1
                parts[i] = 1
        if over > 0:
            if steps > 2:
                k = steps - 2
                j = 0
                while over > 0:
                    parts[j % k] += 1
                    over -= 1
                    j += 1
            else:
                parts[-2] += over
                over = 0

    for r, v in zip(chosen, parts):
        actions.append((r, f"-{v} п. горловина (каждое плечо отдельно)"))

    return actions

# --- СКОС ПЛЕЧА ---
def slope_shoulder(total_stitches: int, start_row: int, end_row: int, rows_total: int):
    """Скос плеча равномерно по чётным рядам, остаток в начале."""
    if total_stitches <= 0:
        return []

    rows = allowed_even_rows(start_row, end_row, rows_total)
    if not rows:
        return []

    steps = len(rows)
    base = total_stitches // steps
    rem = total_stitches % steps

    actions = []
    for i, r in enumerate(rows):
        dec = base + (1 if i < rem else 0)
        actions.append((r, f"-{dec} п. плечо (каждое плечо)"))
    return actions

# --- РЕНДЕР В ТАБЛИЦУ ---
def render_table(rows_total, sections):
    data = []
    for label, actions in sections:
        if actions:
            # вставляем подзаголовок
            data.append(("", f"--- {label} ---"))
            for r, note in actions:
                data.append((r, note))
    df = pd.DataFrame(data, columns=["Ряд", "Действие"])
    st.table(df)

# --- STREAMLIT UI ---
st.title("🧶 Калькулятор вязания")

with st.form("inputs"):
    st.subheader("Параметры")
    rows_total = st.number_input("Высота детали (ряды)", 10, 500, 120)
    neck_stitches = st.number_input("Ширина горловины (петли)", 0, 200, 30)
    neck_rows = st.number_input("Глубина горловины (ряды)", 0, 200, 20)
    shoulder_stitches = st.number_input("Ширина плеча (петли)", 0, 200, 15)
    shoulder_rows = st.number_input("Высота скоса плеча (ряды)", 0, 200, 10)
    submitted = st.form_submit_button("Рассчитать")

if submitted:
    sections = []
    # Горловина
    neck = calc_round_neckline(
        neck_stitches,
        neck_rows,
        rows_total - neck_rows + 1,
        rows_total
    )
    sections.append(("Горловина", neck))

    # Плечо
    shoulder = slope_shoulder(
        shoulder_stitches,
        rows_total - shoulder_rows + 1,
        rows_total,
        rows_total
    )
    sections.append(("Плечо", shoulder))

    st.subheader("📋 Инструкция")
    render_table(rows_total, sections)
