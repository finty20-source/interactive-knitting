import streamlit as st
import numpy as np
import pandas as pd
from collections import defaultdict

st.title("🧶 Интерактивное вязание — калькулятор выкроек")

# -----------------------------
# Общие функции
# -----------------------------
def cm_to_st(cm, dens_st):   
    return int(round((cm/10.0)*dens_st))

def cm_to_rows(cm, dens_row):
    return int(round((cm/10.0)*dens_row))

def to_even(row: int) -> int:
    """Сдвигаем номер ряда до чётного, минимум 6-го."""
    row = max(6, row)
    if row % 2 == 1:
        row += 1
    return row

def distribute_steps(total_change, steps, start_row, end_row, label, action_type="±"):
    """
    Распределяет прибавки/убавки по шагам.
    Если рядов мало → увеличиваем размер шага.
    """
    results = []
    if total_change == 0 or steps <= 0:
        return results

    # максимум доступных чётных рядов
    safe_end = max(6, end_row - 2)
    rows_available = list(range(to_even(start_row), safe_end + 1, 2))
    max_steps = len(rows_available)

    if max_steps == 0:
        return results

    # если шагов больше чем доступно → сжимаем
    steps = min(steps, max_steps)

    # делим изменение на шаги
    base = total_change // steps
    rem = total_change % steps
    parts = [base + (1 if i < rem else 0) for i in range(steps)]

    # выбираем ряды равномерно
    chosen_rows = np.linspace(0, max_steps - 1, num=steps, dtype=int)
    rows = [rows_available[i] for i in chosen_rows]

    for r, val in zip(rows, parts):
        if val > 0:
            if action_type == "+":
                results.append((r, f"+{val} п. {label} слева и +{val} п. {label} справа"))
            elif action_type == "-":
                results.append((r, f"-{val} п. {label} (каждое плечо отдельно)" if "плечо" in label else f"-{val} п. {label}"))
    return results

def calc_round_neckline(total_stitches, total_rows, start_row, rows_total):
    """Горловина: распределение по процентам, укрупняем шаги если мало рядов."""
    if total_stitches <= 0 or total_rows <= 0:
        return []
    percentages = [60, 20, 10, 5, 5]
    parts = [int(round(total_stitches * p / 100)) for p in percentages]
    diff = total_stitches - sum(parts)
    if diff != 0:
        parts[0] += diff

    safe_end = rows_total - 2
    start_row = to_even(start_row)
    rows_available = list(range(start_row, safe_end + 1, 2))
    max_steps = len(rows_available)

    steps = min(len(parts), max_steps)
    parts = parts[:steps]  # если рядов меньше → урезаем шаги
    chosen_rows = np.linspace(0, max_steps - 1, num=steps, dtype=int)
    rows = [rows_available[i] for i in chosen_rows]

    results = []
    for i, (r, dec) in enumerate(zip(rows, parts)):
        if dec > 0:
            if i == 0:
                results.append((r, f"-{dec} п. горловина (середина, разделение на плечи)"))
            else:
                results.append((r, f"-{dec} п. горловина (каждое плечо отдельно)"))
    return results

def slope_shoulder_steps(total_stitches, start_row, end_row, steps=3):
    """Скос плеча с гарантированным влезанием."""
    return distribute_steps(total_stitches, steps, start_row, end_row, "плечо", action_type="-")

def get_section(row, rows_to_armhole_end, armhole_start_row, armhole_end_row, neck_start_row, shoulder_start_row):
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

def show_table(actions, rows_total, rows_to_armhole_end=None, armhole_start_row=None, armhole_end_row=None, neck_start_row=None, shoulder_start_row=None):
    merged = defaultdict(list)
    for row, note in actions:
        merged[to_even(row)].append(note)

    if not merged:
        st.info("Нет действий для выбранных параметров.")
    else:
        rows_sorted = sorted(merged.keys())
        data = {
            "Ряд": rows_sorted,
            "Действия": [", ".join(merged[r]) for r in rows_sorted],
        }
        if armhole_start_row is not None:  # перед/спинка
            data["Сегмент"] = [
                get_section(r, rows_to_armhole_end, armhole_start_row, armhole_end_row, neck_start_row, shoulder_start_row)
                for r in rows_sorted
            ]
        else:  # рукав
            data["Сегмент"] = ["Рукав" if r < rows_total else "Окат (прямой)" for r in rows_sorted]

        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)

# -----------------------------
# Вкладки
# -----------------------------
tab1, tab2, tab3 = st.tabs(["Перед", "Спинка", "Рукав"])

# -----------------------------
# Перед
# -----------------------------
with tab1:
    st.header("Перед")
    density_st = st.number_input("Плотность (петли в 10 см)", min_value=1, value=23, key="density_st1")
    density_row = st.number_input("Плотность (ряды в 10 см)", min_value=1, value=40, key="density_row1")

    chest_cm = st.number_input("Обхват груди (см)", min_value=50, value=90, key="chest1")
    hip_cm   = st.number_input("Обхват низа (см)",   min_value=50, value=80, key="hip1")
    length_cm= st.number_input("Длина изделия (см)", min_value=30, value=55, key="length1")

    armhole_depth_cm = st.number_input("Глубина проймы (см)", min_value=10, value=23, key="armhole1")
    shoulders_width_cm = st.number_input("Ширина изделия по плечам (см)", min_value=20, value=100, key="shoulders1")

    neck_width_cm = st.number_input("Ширина горловины (см)", min_value=5, value=18, key="neck_w1")
    neck_depth_cm = st.number_input("Глубина горловины (см)", min_value=2, value=6, key="neck_d1")

    shoulder_len_cm   = st.number_input("Длина одного плеча (см)", min_value=5, value=12, key="sh_len1")
    shoulder_slope_cm = st.number_input("Высота скоса плеча (см)", min_value=1, value=4, key="sh_slope1")

    if st.button("🔄 Рассчитать перед"):
        stitches_chest      = cm_to_st(chest_cm, density_st)
        stitches_hip        = cm_to_st(hip_cm,   density_st)
        stitches_shoulders  = cm_to_st(shoulders_width_cm, density_st)

        rows_total          = cm_to_rows(length_cm, density_row)
        rows_armhole        = cm_to_rows(armhole_depth_cm, density_row)

        neck_stitches       = cm_to_st(neck_width_cm, density_st)
        neck_rows           = cm_to_rows(neck_depth_cm, density_row)

        stitches_shoulder   = cm_to_st(shoulder_len_cm, density_st)
        rows_shoulder_slope = cm_to_rows(shoulder_slope_cm, density_row)

        rows_to_armhole_end = rows_total - rows_armhole
        shoulder_start_row  = to_even(rows_total - rows_shoulder_slope + 1)
        neck_start_row      = to_even(rows_total - neck_rows + 1)

        armhole_start_row = rows_to_armhole_end + 1
        armhole_end_row   = min(rows_total, shoulder_start_row - 1)

        # действия
        actions = []
        actions += distribute_steps(stitches_chest - stitches_hip, abs(stitches_chest - stitches_hip)//2, 6, rows_to_armhole_end, "бок", action_type="+")
        actions += distribute_steps(stitches_shoulders - stitches_chest, abs(stitches_shoulders - stitches_chest)//2, armhole_start_row, armhole_end_row, "пройма", action_type="+")
        actions += calc_round_neckline(neck_stitches, neck_rows, neck_start_row, rows_total)
        actions += slope_shoulder_steps(stitches_shoulder, shoulder_start_row, rows_total, steps=3)

        st.subheader("Пошаговый план")
        show_table(actions, rows_total, rows_to_armhole_end, armhole_start_row, armhole_end_row, neck_start_row, shoulder_start_row)

# -----------------------------
# Спинка
# -----------------------------
with tab2:
    st.header("Спинка")
    density_st = st.number_input("Плотность (петли в 10 см)", min_value=1, value=23, key="density_st2")
    density_row = st.number_input("Плотность (ряды в 10 см)", min_value=1, value=40, key="density_row2")

    chest_cm = st.number_input("Обхват груди (см)", min_value=50, value=90, key="chest2")
    hip_cm   = st.number_input("Обхват низа (см)",   min_value=50, value=80, key="hip2")
    length_cm= st.number_input("Длина изделия (см)", min_value=30, value=55, key="length2")

    armhole_depth_cm = st.number_input("Глубина проймы (см)", min_value=10, value=23, key="armhole2")
    shoulders_width_cm = st.number_input("Ширина изделия по плечам (см)", min_value=20, value=100, key="shoulders2")

    neck_width_cm = st.number_input("Ширина горловины (см)", min_value=5, value=18, key="neck_w2")
    neck_depth_cm = st.number_input("Глубина горловины (см)", min_value=1, value=3, key="neck_d2")

    shoulder_len_cm   = st.number_input("Длина одного плеча (см)", min_value=5, value=12, key="sh_len2")
    shoulder_slope_cm = st.number_input("Высота скоса плеча (см)", min_value=1, value=4, key="sh_slope2")

    if st.button("🔄 Рассчитать спинку"):
        stitches_chest      = cm_to_st(chest_cm, density_st)
        stitches_hip        = cm_to_st(hip_cm,   density_st)
        stitches_shoulders  = cm_to_st(shoulders_width_cm, density_st)

        rows_total          = cm_to_rows(length_cm, density_row)
        rows_armhole        = cm_to_rows(armhole_depth_cm, density_row)

        neck_stitches       = cm_to_st(neck_width_cm, density_st)
        neck_rows           = cm_to_rows(neck_depth_cm, density_row)

        stitches_shoulder   = cm_to_st(shoulder_len_cm, density_st)
        rows_shoulder_slope = cm_to_rows(shoulder_slope_cm, density_row)

        rows_to_armhole_end = rows_total - rows_armhole
        shoulder_start_row  = to_even(rows_total - rows_shoulder_slope + 1)
        neck_start_row      = to_even(rows_total - neck_rows + 1)

        armhole_start_row = rows_to_armhole_end + 1
        armhole_end_row   = min(rows_total, shoulder_start_row - 1)

        actions = []
        actions += distribute_steps(stitches_chest - stitches_hip, abs(stitches_chest - stitches_hip)//2, 6, rows_to_armhole_end, "бок", action_type="+")
        actions += distribute_steps(stitches_shoulders - stitches_chest, abs(stitches_shoulders - stitches_chest)//2, armhole_start_row, armhole_end_row, "пройма", action_type="+")
        actions += calc_round_neckline(neck_stitches, neck_rows, neck_start_row, rows_total)
        actions += slope_shoulder_steps(stitches_shoulder, shoulder_start_row, rows_total, steps=3)

        st.subheader("Пошаговый план")
        show_table(actions, rows_total, rows_to_armhole_end, armhole_start_row, armhole_end_row, neck_start_row, shoulder_start_row)

# -----------------------------
# Рукав
# -----------------------------
with tab3:
    st.header("Рукав (оверсайз, прямой окат)")
    density_st = st.number_input("Плотность (петли в 10 см)", min_value=1, value=23, key="density_st3")
    density_row = st.number_input("Плотность (ряды в 10 см)", min_value=1, value=40, key="density_row3")

    length_cm   = st.number_input("Длина рукава (см)", min_value=20, value=60, key="len_r")
    wrist_cm    = st.number_input("Ширина манжеты (см)", min_value=10, value=18, key="wrist")
    top_cm      = st.number_input("Ширина рукава вверху (см)", min_value=20, value=36, key="top_r")

    if st.button("🔄 Рассчитать рукав"):
        stitches_wrist = cm_to_st(wrist_cm, density_st)
        stitches_top   = cm_to_st(top_cm, density_st)
        rows_total     = cm_to_rows(length_cm, density_row)

        delta = stitches_top - stitches_wrist
        if delta % 2 == 1:
            delta += 1

        actions = []
        actions += distribute_steps(delta, delta//2, 6, rows_total, "рукав", action_type="+")
        actions.append((rows_total, "Закрыть все петли (прямой окат)"))

        st.subheader("Пошаговый план")
        show_table(actions, rows_total)
