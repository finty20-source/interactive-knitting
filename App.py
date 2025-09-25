import streamlit as st
import numpy as np
import pandas as pd
from collections import defaultdict

st.title("🧶 Интерактивное вязание — калькулятор выкроек")

# -----------------------------
# Общие утилиты
# -----------------------------
def cm_to_st(cm, dens_st):   
    return int(round((cm/10.0)*dens_st))

def cm_to_rows(cm, dens_row):
    return int(round((cm/10.0)*dens_row))

def to_even(row: int) -> int:
    """Сдвигаем ряд до чётного, минимум 6-го."""
    row = max(6, row)
    return row if row % 2 == 0 else row + 1

def allowed_even_rows(start_row: int, end_row: int, rows_total: int):
    """
    Разрешённые чётные ряды:
    - не раньше 6,
    - не позже end_row,
    - не позже rows_total-2.
    """
    high = min(end_row, rows_total - 2)
    if high < 6:
        return []
    start = to_even(start_row)
    high = high if high % 2 == 0 else high - 1
    return list(range(start, high + 1, 2)) if start <= high else []

def split_total_into_steps(total: int, steps: int):
    """Делим total на steps положительных частей (сумма = total)."""
    if total <= 0:
        return []
    steps = max(1, min(steps, total))
    base = total // steps
    rem = total % steps
    parts = [base] * steps
    for i in range(rem):
        parts[i] += 1
    return parts

# -----------------------------
# Распределения манипуляций
# -----------------------------
def sym_increases(total_add: int, desired_steps: int, start_row: int, end_row: int, rows_total: int, label: str):
    """Симметричные прибавки (слева+справа)."""
    if total_add <= 0:
        return []
    if total_add % 2 == 1:
        total_add += 1
    rows = allowed_even_rows(start_row, end_row, rows_total)
    if not rows:
        return []
    steps = min(desired_steps, len(rows))
    per_side_total = total_add // 2
    parts = split_total_into_steps(per_side_total, steps)
    idxs = np.linspace(0, len(rows) - 1, num=steps, dtype=int)
    chosen = [rows[i] for i in idxs]
    return [(r, f"+{v} п. {label} слева и +{v} п. {label} справа") for r, v in zip(chosen, parts)]

def solo_decreases(total_dec: int, desired_steps: int, start_row: int, end_row: int, rows_total: int, label: str):
    """Несимметричные убавки (например, плечо)."""
    if total_dec <= 0:
        return []
    rows = allowed_even_rows(start_row, end_row, rows_total)
    if not rows:
        return []
    steps = min(desired_steps, len(rows))
    parts = split_total_into_steps(total_dec, steps)
    idxs = np.linspace(0, len(rows) - 1, num=steps, dtype=int)
    chosen = [rows[i] for i in idxs]
    return [(r, f"-{v} п. {label}") for r, v in zip(chosen, parts)]

def calc_round_neckline(total_stitches: int, total_rows: int, start_row: int, rows_total: int):
    """Горловина: 60% сразу, остальные по шагам. Шаги укрупняются если мало рядов."""
    if total_stitches <= 0 or total_rows <= 0:
        return []
    parts = [int(round(total_stitches * p / 100)) for p in [60, 20, 10, 5, 5]]
    diff = total_stitches - sum(parts)
    if diff != 0:
        parts[0] += diff
    rows = allowed_even_rows(start_row, start_row + total_rows - 1, rows_total)
    if not rows:
        return []
    actions = []
    # первый шаг (центральное закрытие)
    if parts[0] > 0:
        actions.append((rows[0], f"-{parts[0]} п. горловина (середина, разделение на плечи)"))
    # остальные шаги
    rest_total = sum(parts[1:])
    if rest_total > 0 and len(rows) > 1:
        steps = min(len(rows) - 1, rest_total)
        rest_parts = split_total_into_steps(rest_total, steps)
        idxs = np.linspace(1, len(rows) - 1, num=steps, dtype=int)
        chosen = [rows[i] for i in idxs]
        for r, v in zip(chosen, rest_parts):
            actions.append((r, f"-{v} п. горловина (каждое плечо отдельно)"))
    return actions

def slope_shoulder(total_stitches: int, start_row: int, end_row: int, rows_total: int, steps: int = 3):
    """Скос плеча с укрупнением шага."""
    return solo_decreases(total_stitches, steps, start_row, end_row, rows_total, "плечо (каждое плечо)")

def section_label(row, rows_to_armhole_end, armhole_start_row, armhole_end_row, neck_start_row, shoulder_start_row):
    tags = []
    if row <= rows_to_armhole_end:
        tags.append("Низ изделия")
    if armhole_start_row <= row <= armhole_end_row:
        tags.append("Пройма")
    if row >= neck_start_row:
        tags.append("Горловина")
    if row >= shoulder_start_row:
        tags.append("Скос плеча")
    return ", ".join(tags) if tags else "—"

def render_table(actions, rows_total, rows_to_armhole_end=None, armhole_start_row=None, armhole_end_row=None, neck_start_row=None, shoulder_start_row=None):
    merged = defaultdict(list)
    for row, note in actions:
        merged[row].append(note)  # ряды уже чётные (кроме финального закрытия)

    rows_sorted = sorted(merged.keys())
    data = {
        "Ряд": rows_sorted,
        "Действия": [", ".join(merged[r]) for r in rows_sorted],
    }
    if armhole_start_row is not None:
        data["Сегмент"] = [
            section_label(r, rows_to_armhole_end, armhole_start_row, armhole_end_row, neck_start_row, shoulder_start_row)
            for r in rows_sorted
        ]
    else:
        data["Сегмент"] = ["Рукав" if r < rows_total else "Окат (прямой)" for r in rows_sorted]
    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

# -----------------------------
# Вкладки
# -----------------------------
tab1, tab2, tab3 = st.tabs(["Перед", "Спинка", "Рукав"])

# -----------------------------
# Перед
# -----------------------------
with tab1:
    st.header("Перед")
    density_st = st.number_input("Плотность (петли в 10 см)", min_value=1, value=23, key="dst1")
    density_row = st.number_input("Плотность (ряды в 10 см)", min_value=1, value=40, key="drw1")
    chest_cm = st.number_input("Обхват груди (см)", min_value=50, value=90, key="ch1")
    hip_cm   = st.number_input("Обхват низа (см)",   min_value=50, value=80, key="hip1")
    length_cm= st.number_input("Длина изделия (см)", min_value=30, value=55, key="len1")
    armhole_depth_cm = st.number_input("Глубина проймы (см)", min_value=10, value=23, key="arm1")
    shoulders_width_cm = st.number_input("Ширина по плечам (см)", min_value=20, value=100, key="shw1")
    neck_width_cm = st.number_input("Ширина горловины (см)", min_value=5, value=18, key="nw1")
    neck_depth_cm = st.number_input("Глубина горловины (см)", min_value=2, value=6, key="nd1")
    shoulder_len_cm   = st.number_input("Длина плеча (см)", min_value=5, value=12, key="sl1")
    shoulder_slope_cm = st.number_input("Высота скоса плеча (см)", min_value=1, value=4, key="ss1")

    if st.button("🔄 Рассчитать перед"):
        st_chest, st_hip, st_sh = cm_to_st(chest_cm, density_st), cm_to_st(hip_cm, density_st), cm_to_st(shoulders_width_cm, density_st)
        rows_total, rows_armhole = cm_to_rows(length_cm, density_row), cm_to_rows(armhole_depth_cm, density_row)
        neck_st, neck_rows = cm_to_st(neck_width_cm, density_st), cm_to_rows(neck_depth_cm, density_row)
        st_shoulder, rows_sh_slope = cm_to_st(shoulder_len_cm, density_st), cm_to_rows(shoulder_slope_cm, density_row)
        rows_to_armhole_end = rows_total - rows_armhole
        shoulder_start_row, neck_start_row = rows_total - rows_sh_slope + 1, rows_total - neck_rows + 1
        armhole_start_row, armhole_end_row = rows_to_armhole_end + 1, min(rows_total, shoulder_start_row - 1)
        actions = []
        if st_chest > st_hip:
            actions += sym_increases(st_chest - st_hip, (st_chest - st_hip)//2, 6, rows_to_armhole_end, rows_total, "бок")
        if st_sh > st_chest:
            actions += sym_increases(st_sh - st_chest, (st_sh - st_chest)//2, armhole_start_row, armhole_end_row, rows_total, "пройма")
        actions += calc_round_neckline(neck_st, neck_rows, neck_start_row, rows_total)
        actions += slope_shoulder(st_shoulder, shoulder_start_row, rows_total, rows_total, steps=3)
        st.subheader("Пошаговый план")
        render_table(actions, rows_total, rows_to_armhole_end, armhole_start_row, armhole_end_row, neck_start_row, shoulder_start_row)

# -----------------------------
# Спинка
# -----------------------------
with tab2:
    st.header("Спинка")
    density_st = st.number_input("Плотность (петли в 10 см)", min_value=1, value=23, key="dst2")
    density_row = st.number_input("Плотность (ряды в 10 см)", min_value=1, value=40, key="drw2")
    chest_cm = st.number_input("Обхват груди (см)", min_value=50, value=90, key="ch2")
    hip_cm   = st.number_input("Обхват низа (см)",   min_value=50, value=80, key="hip2")
    length_cm= st.number_input("Длина изделия (см)", min_value=30, value=55, key="len2")
    armhole_depth_cm = st.number_input("Глубина проймы (см)", min_value=10, value=23, key="arm2")
    shoulders_width_cm = st.number_input("Ширина по плечам (см)", min_value=20, value=100, key="shw2")
    neck_width_cm = st.number_input("Ширина горловины (см)", min_value=5, value=18, key="nw2")
    neck_depth_cm = st.number_input("Глубина горловины (см)", min_value=1, value=3, key="nd2")
    shoulder_len_cm   = st.number_input("Длина плеча (см)", min_value=5, value=12, key="sl2")
    shoulder_slope_cm = st.number_input("Высота скоса плеча (см)", min_value=1, value=4, key="ss2")

    if st.button("🔄 Рассчитать спинку"):
        st_chest, st_hip, st_sh = cm_to_st(chest_cm, density_st), cm_to_st(hip_cm, density_st), cm_to_st(shoulders_width_cm, density_st)
        rows_total, rows_armhole = cm_to_rows(length_cm, density_row), cm_to_rows(armhole_depth_cm, density_row)
        neck_st, neck_rows = cm_to_st(neck_width_cm, density_st), cm_to_rows(neck_depth_cm, density_row)
        st_shoulder, rows_sh_slope = cm_to_st(shoulder_len_cm, density_st), cm_to_rows(shoulder_slope_cm, density_row)
        rows_to_armhole_end = rows_total - rows_armhole
        shoulder_start_row, neck_start_row = rows_total - rows_sh_slope + 1, rows_total - neck_rows + 1
        armhole_start_row, armhole_end_row = rows_to_armhole_end + 1, min(rows_total, shoulder_start_row - 1)
        actions = []
        if st_chest > st_hip:
            actions += sym_increases(st_chest - st_hip, (st_chest - st_hip)//2, 6, rows_to_armhole_end, rows_total, "бок")
        if st_sh > st_chest:
            actions += sym_increases(st_sh - st_chest, (st_sh - st_chest)//2, armhole_start_row, armhole_end_row, rows_total, "пройма")
        actions += calc_round_neckline(neck_st, neck_rows, neck_start_row, rows_total)
        actions += slope_shoulder(st_shoulder, shoulder_start_row, rows_total, rows_total, steps=3)
        st.subheader("Пошаговый план")
        render_table(actions, rows_total, rows_to_armhole_end, armhole_start_row, armhole_end_row, neck_start_row, shoulder_start_row)

# -----------------------------
# Рукав
# -----------------------------
with tab3:
    st.header("Рукав (оверсайз, прямой окат)")
    density_st = st.number_input("Плотность (петли в 10 см)", min_value=1, value=23, key="dst3")
    density_row = st.number_input("Плотность (ряды в 10 см)", min_value=1, value=40, key="drw3")
    length_cm   = st.number_input("Длина рукава (см)", min_value=20, value=60, key="len_r")
    wrist_cm    = st.number_input("Ширина манжеты (см)", min_value=10, value=18, key="wr")
    top_cm      = st.number_input("Ширина верха рукава (см)", min_value=20, value=36, key="top_r")

    if st.button("🔄 Рассчитать рукав"):
        st_wrist, st_top = cm_to_st(wrist_cm, density_st), cm_to_st(top_cm, density_st)
        rows_total = cm_to_rows(length_cm, density_row)
        delta = st_top - st_wrist
        actions = []
        if delta > 0:
            actions += sym_increases(delta, delta//2, 6, rows_total - 1, rows_total, "рукав")
        actions.append((rows_total, "Закрыть все петли (прямой окат)"))
        st.subheader("Пошаговый план")
        render_table(actions, rows_total)
