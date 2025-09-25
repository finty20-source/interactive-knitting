import streamlit as st
import numpy as np

st.title("🧶 Интерактивное вязание — расчёт переда (оверсайз)")

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
st.header("Пересчёт в петли и ряды")

stitches_chest = int((chest / 10) * density_stitches)
stitches_hip = int((hip / 10) * density_stitches)
stitches_shoulders = int((shoulders_width_cm / 10) * density_stitches)

rows_total = int((length / 10) * density_rows)
rows_armhole = int((armhole_depth_cm / 10) * density_rows)

neck_stitches = int((neck_width_cm / 10) * density_stitches)
neck_rows = int((neck_depth_cm / 10) * density_rows)

stitches_shoulder = int((shoulder_length_cm / 10) * density_stitches)
rows_shoulder_slope = int((shoulder_slope_cm / 10) * density_rows)

# автоматически считаем расширение по пройме
armhole_extra_stitches = stitches_shoulders - stitches_chest

st.write(f"🔹 Петли по груди: **{stitches_chest}**")
st.write(f"🔹 Петли по низу: **{stitches_hip}**")
st.write(f"🔹 Высота изделия: **{rows_total} рядов**")
st.write(f"🔹 Высота проймы: **{rows_armhole} рядов**")
st.write(f"🔹 Ширина по плечам: {shoulders_width_cm} см → {stitches_shoulders} п.")
st.write(f"🔹 Прибавки по пройме: {armhole_extra_stitches} п. (равномерно)")
st.write(f"🔹 Горловина: {neck_stitches} п. за {neck_rows} р.")
st.write(f"🔹 Плечо: {shoulder_length_cm} см → {stitches_shoulder} п., скос {rows_shoulder_slope} р.")

# -----------------------------
# 📊 Логика расчётов
# -----------------------------
def distribute_increases(start_st, end_st, total_rows):
    """Равномерное распределение прибавок"""
    diff = end_st - start_st
    if diff <= 0:
        return []
    step = total_rows / diff
    return [round(i * step) for i in range(1, diff + 1)]

def calc_round_neckline(total_stitches, total_rows):
    """Округлая горловина по процентам"""
    percentages = [60, 20, 10, 5, 5]
    decreases = [round(total_stitches * p / 100) for p in percentages]
    plan = []
    current_row = 1
    for d in decreases:
        if d > 0:
            plan.append((current_row, d))
        current_row += 2
    while current_row <= total_rows:
        plan.append((current_row, 0))
        current_row += 1
    return plan

def slope_shoulder(stitches):
    """Скос плеча: 3 ступени"""
    if stitches <= 0:
        return []
    step = stitches // 3
    return [step, step, stitches - 2*step]

# -----------------------------
# 📌 Итоговый план переда
# -----------------------------
st.header("План вязания переда")

rows_to_armhole = rows_total - rows_armhole

# 1. От низа до проймы
st.subheader("1️⃣ От низа до проймы")
hip_to_chest_increases = distribute_increases(stitches_hip, stitches_chest, rows_to_armhole)
st.write(f"Нужно прибавить {stitches_chest - stitches_hip} петель за {rows_to_armhole} рядов.")
for r in hip_to_chest_increases:
    st.write(f"➡️ Ряд {r}: прибавить 1 п.")

# 2. Пройма (оверсайз → прибавки)
st.subheader("2️⃣ Пройма (оверсайз)")
armhole_increases = distribute_increases(stitches_chest, stitches_shoulders, rows_armhole)
st.write(f"Пройма: прибавить {armhole_extra_stitches} п. за {rows_armhole} рядов.")
for r in armhole_increases:
    st.write(f"➡️ Ряд {rows_to_armhole + r}: прибавить 1 п.")

# 3. Горловина
st.subheader("3️⃣ Горловина (округлая)")
neck_plan = calc_round_neckline(neck_stitches, neck_rows)
for row, dec in neck_plan:
    if dec > 0:
        st.write(f"➡️ Ряд {rows_total - rows_armhole - neck_rows + row}: убавить {dec} п.")
    else:
        st.write(f"➡️ Ряд {rows_total - rows_armhole - neck_rows + row}: прямо")

# 4. Плечо
st.subheader("4️⃣ Скос плеча")
shoulder_parts = slope_shoulder(stitches_shoulder)
st.write(f"Плечо: закрыть {stitches_shoulder} петель за {rows_shoulder_slope} рядов (3 ступени).")
for i, val in enumerate(shoulder_parts, 1):
    st.write(f"➡️ Этап {i}: закрыть {val} п.")

# -----------------------------
# ✅ Резюме
# -----------------------------
st.write("---")
st.success(
    f"Перед рассчитан: {rows_total} рядов.\n"
    f"• От низа до проймы: {rows_to_armhole} рядов.\n"
    f"• Пройма (оверсайз): +{armhole_extra_stitches} п.\n"
    f"• Горловина: {neck_rows} рядов.\n"
    f"• Плечо: {stitches_shoulder} п., скос {rows_shoulder_slope} р."
)
