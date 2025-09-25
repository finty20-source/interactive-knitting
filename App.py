import streamlit as st
import numpy as np
import pandas as pd
from collections import defaultdict

st.title("🧶 Интерактивное вязание — расчёт переда (единый план)")

# -----------------------------
# Ввод параметров
# -----------------------------
st.header("Ввод параметров")

density_st = st.number_input("Плотность (петли в 10 см)", min_value=1, value=23)
density_row = st.number_input("Плотность (ряды в 10 см)", min_value=1, value=40)

# Основные мерки
chest_cm = st.number_input("Обхват груди (см)", min_value=50, value=90)
hip_cm   = st.number_input("Обхват низа (см)",   min_value=50, value=80)
length_cm= st.number_input("Длина изделия (см)", min_value=30, value=55)

# Пройма
armhole_depth_cm = st.number_input("Глубина проймы (см)", min_value=10, value=23)

# Верхняя ширина (оверсайз)
shoulders_width_cm = st.number_input("Ширина изделия по плечам (см)", min_value=20, value=100)

# Горловина (круглая, по процентам)
neck_width_cm = st.number_input("Ширина горловины (см)", min_value=5, value=18)
neck_depth_cm = st.number_input("Глубина горловины (см)", min_value=2, value=6)

# Плечо
shoulder_len_cm   = st.number_input("Длина одного плеча (см)", min_value=5, value=12)
shoulder_slope_cm = st.number_input("Высота скоса плеча (см)", min_value=1, value=4)

st.write("---")

# -----------------------------
# Пересчёт см → петли/ряды
# -----------------------------
def cm_to_st(cm, dens_st):   return int(round((cm/10.0)*dens_st))
def cm_to_rows(cm, dens_row):return int(round((cm/10.0)*dens_row))

stitches_chest      = cm_to_st(chest_cm, density_st)
stitches_hip        = cm_to_st(hip_cm,   density_st)
stitches_shoulders  = cm_to_st(shoulders_width_cm, density_st)

rows_total          = cm_to_rows(length_cm, density_row)
rows_armhole        = cm_to_rows(armhole_depth_cm, density_row)

neck_stitches       = cm_to_st(neck_width_cm, density_st)
neck_rows           = cm_to_rows(neck_depth_cm, density_row)

stitches_shoulder   = cm_to_st(shoulder_len_cm, density_st)
rows_shoulder_slope = cm_to_rows(shoulder_slope_cm, density_row)

# служебные границы сегментов
rows_to_armhole_end   = max(0, rows_total - rows_armhole)
shoulder_start_row    = max(2, rows_total - rows_shoulder_slope + 1)
neck_start_row        = max(2, rows_total - neck_rows + 1)

# пройма должна закончиться ДО скоса плеча
armhole_start_row = rows_to_armhole_end + 1
armhole_end_row   = max(armhole_start_row-1, min(rows_total, shoulder_start_row - 1))

# расширение по пройме (оверсайз)
armhole_extra_st_total = max(0, stitches_shoulders - stitches_chest)

# -----------------------------
# Утилиты распределения
# -----------------------------
def spread_rows(start_row: int, end_row: int, count: int):
    """Равномерное распределение по рядам (без дублей, начиная с >=2)."""
    if count <= 0 or end_row < start_row:
        return []
    start_row = max(2, start_row)
    if start_row == end_row:
        return [start_row]*count
    xs = np.linspace(start_row, end_row, num=count, endpoint=True)
    rows = [int(round(x)) for x in xs]
    used = set()
    for i in range(len(rows)):
        r = rows[i]
        while r in used and r < end_row:
            r += 1
        while r in used and r > start_row:
            r -= 1
        rows[i] = max(2, r)
        used.add(rows[i])
    rows.sort()
    return rows

def distribute_side_increases(start_row, end_row, total_delta, label):
    """
    Прибавки симметричные → выполняются парой в одном ряду.
    """
    if total_delta <= 0 or end_row < start_row:
        return []
    pairs = total_delta // 2
    rows = spread_rows(start_row, end_row, pairs)
    out = []
    for r in rows:
        out.append((r, f"+1 п. {label} слева и +1 п. {label} справа"))
    if total_delta % 2 == 1 and rows:
        out.append((rows[-1] + 2, f"+1 п. {label} (доп.)"))
    return out

def calc_round_neckline_by_percent(total_stitches, total_rows, start_row, percentages=(60,20,10,5,5)):
    """
    Горловина: симметричные убавки → разные ряды.
    """
    if total_stitches <= 0 or total_rows <= 0:
        return []
    parts = [int(round(total_stitches * p / 100.0)) for p in percentages]
    diff = total_stitches - sum(parts)
    if diff != 0:
        parts[0] += diff
    actions = []
    row = max(2, start_row)
    for dec in parts:
        if dec > 0 and row <= start_row + total_rows - 1:
            actions.append((row,   f"-{dec} п. горловина (левая)"))
            actions.append((row+1, f"-{dec} п. горловина (правая)"))
        row += 2
    return actions

def slope_shoulder_steps(total_stitches, start_row, end_row, steps=3):
    """
    Скос плеча: убавки симметричные → разные ряды.
    """
    if total_stitches <= 0 or end_row < start_row or steps <= 0:
        return []
    base = total_stitches // steps
    parts = [base]*steps
    parts[-1] += (total_stitches - base*steps)
    rows = spread_rows(start_row, end_row, steps)
    out = []
    for r, p in zip(rows, parts):
        out.append((r,   f"закрыть {p} п. плечо (левое)"))
        out.append((r+1, f"закрыть {p} п. плечо (правое)"))
    return out

# -----------------------------
# План действий
# -----------------------------
actions = []

# 1) Низ → грудь
delta_bottom = max(0, stitches_chest - stitches_hip)
actions += distribute_side_increases(
    2, rows_to_armhole_end,
    delta_bottom,
    label="бок"
)

# 2) Пройма (оверсайз)
if armhole_start_row <= armhole_end_row:
    actions += distribute_side_increases(
        armhole_start_row, armhole_end_row,
        armhole_extra_st_total,
        label="пройма"
    )

# 3) Горловина
actions += calc_round_neckline_by_percent(
    neck_stitches, neck_rows, neck_start_row
)

# 4) Скос плеча
if shoulder_start_row <= rows_total:
    actions += slope_shoulder_steps(
        stitches_shoulder, shoulder_start_row, rows_total, steps=3
    )

# -----------------------------
# Схлопываем по рядам
# -----------------------------
merged = defaultdict(list)
for row, note in actions:
    merged[row].append(note)

# -----------------------------
# Вывод таблицей
# -----------------------------
st.header("Единый пошаговый план")

if not merged:
    st.info("Нет действий для выбранных параметров.")
else:
    rows_sorted = sorted(merged.keys())
    data = {"Ряд": rows_sorted,
            "Действия": [", ".join(merged[r]) for r in rows_sorted]}
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)

st.write("---")
st.success(
    f"Итого рядов: {rows_total}. "
    f"Горловина начинается с {neck_start_row}-го ряда, "
    f"скос плеча — с {shoulder_start_row}-го ряда."
)
