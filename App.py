import streamlit as st
import numpy as np
from collections import defaultdict

st.title("🧶 Интерактивное вязание — единый план переда")

# -----------------------------
# 🧾 Ввод параметров
# -----------------------------
st.header("Ввод параметров")

density_stitches = st.number_input("Плотность (петли в 10 см)", min_value=1, value=23)
density_rows = st.number_input("Плотность (ряды в 10 см)", min_value=1, value=40)

# Основные мерки
chest = st.number_input("Обхват груди (см)", min_value=50, value=90)
hip = st.number_input("Обхват низа изделия (см)", min_value=50, value=80)
length = st.number_input("Длина изделия (см)", min_value=30, value=55)

# Пройма
armhole_depth_cm = st.number_input("Глубина проймы (см)", min_value=10, value=23)

# Верхняя ширина
shoulders_width_cm = st.number_input("Ширина изделия по плечам (см)", min_value=20, value=100)

# Горловина
neck_width_cm = st.number_input("Ширина горловины (см)", min_value=5, value=18)
neck_depth_cm = st.number_input("Глубина горловины (см)", min_value=2, value=6)

# Плечи
shoulder_length_cm = st.number_input("Длина одного плеча (см)", min_value=5, value=12)
shoulder_slope_cm = st.number_input("Высота скоса плеча (см)", min_value=1, value=4)

st.write("---")

# -----------------------------
# 🔄 Пересчёт см → петли/ряды
# -----------------------------
stitches_chest = int((chest / 10) * density_stitches)
stitches_hip = int((hip / 10) * density_stitches)
stitches_shoulders = int((shoulders_width_cm / 10) * density_stitches)

rows_total = int((length / 10) * density_rows)
rows_armhole = int((armhole_depth_cm / 10) * density_rows)

neck_stitches = int((neck_width_cm / 10) * density_stitches)
neck_rows = int((neck_depth_cm / 10) * density_rows)

stitches_shoulder = int((shoulder_length_cm / 10) * density_stitches)
rows_shoulder_slope = int((shoulder_slope_cm / 10) * density_rows)

# автоматическая ширина по пройме
armhole_extra_stitches = stitches_shoulders - stitches_chest

# -----------------------------
# 📊 Логика расчётов
# -----------------------------
def distribute_increases(start_st, end_st, total_rows, label):
    diff = end_st - start_st
    if diff <= 0:
        return []
    step = total_rows / diff
    return [(round(i * step), f"+1 п. ({label})") for i in range(1, diff + 1)]

def calc_round_neckline(total_stitches, total_rows, start_row):
    percentages = [60, 20, 10, 5, 5]
    decreases = [round(total_stitches * p / 100) for p in percentages]
    plan = []
    current_row = 0
    for d in decreases:
        if d > 0:
            plan.append((start_row + current_row, f"-{d} п. (горловина)"))
        current_row += 2
    return plan

def slope_shoulder(stitches, rows, start_row):
    if stitches <= 0:
        return []
    step = stitches // 3
    parts = [step, step, stitches - 2*step]
    return [(start_row + i, f"закрыть {val} п. (плечо)") for i, val in enumerate(parts, 1)]

# -----------------------------
# 📌 Генерация общего плана
# -----------------------------
actions = []

rows_to_armhole = rows_total - rows_armhole

# 1. От низа до груди
actions += distribute_increases(stitches_hip, stitches_chest, rows_to_armhole, "боковые прибавки")

# 2. Пройма (оверсайз → прибавки)
actions += distribute_increases(stitches_chest, stitches_shoulders, rows_armhole, "пройма")

# 3. Горловина (стартует сверху)
neck_start_row = rows_total - neck_rows
actions += calc_round_neckline(neck_stitches, neck_rows, neck_start_row)

# 4. Плечи (тоже сверху)
shoulder_start_row = rows_total - rows_shoulder_slope
actions += slope_shoulder(stitches_shoulder, rows_shoulder_slope, shoulder_start_row)

# -----------------------------
# 🗂 Объединение по рядам
# -----------------------------
merged = defaultdict(list)
for row, action in actions:
    merged[row].append(action)

# -----------------------------
# 📋 Вывод единого плана
# -----------------------------
st.header("Единый пошаговый план переда")

for row in sorted(merged.keys()):
    st.write(f"➡️ Ряд {row}: " + ", ".join(merged[row]))

st.write("---")
st.success(f"Всего рядов: {rows_total}. Горловина с {neck_start_row}-го ряда, плечо со {shoulder_start_row}-го ряда.")
