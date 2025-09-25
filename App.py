import streamlit as st
import numpy as np

st.title("🧶 Интерактивное вязание — калькулятор свитера")

# --- Ввод параметров ---
st.header("Ввод данных")

density_stitches = st.number_input("Плотность (петли в 10 см)", min_value=1, value=23)
density_rows = st.number_input("Плотность (ряды в 10 см)", min_value=1, value=40)

chest = st.number_input("Обхват груди (см)", min_value=50, value=90)
length = st.number_input("Длина изделия (см)", min_value=30, value=55)

st.write("---")

# --- Пересчёт ---
st.header("Базовые расчёты")

stitches_total = int((chest / 10) * density_stitches)
rows_total = int((length / 10) * density_rows)

st.write(f"🔹 Общее количество петель по груди: **{stitches_total}**")
st.write(f"🔹 Общее количество рядов по длине: **{rows_total}**")

# --- Функция для круглой горловины ---
def calc_round_neckline(total_stitches, total_rows):
    """
    Расчёт убавок для круглой горловины.
    total_stitches - сколько всего петель нужно убрать
    total_rows - высота горловины в рядах
    """

    # пропорции убавок (сумма = 100%)
    percentages = [60, 20, 10, 5, 5]

    # пересчёт в реальные петли
    decreases = [round(total_stitches * p / 100) for p in percentages]

    plan = []
    current_row = 1

    for d in decreases:
        plan.append((current_row, d))   # убавка в этом ряду
        current_row += 2                # делаем через ряд

    # оставшиеся ряды вяжем прямо
    while current_row <= total_rows:
        plan.append((current_row, 0))
        current_row += 1

    return plan

# --- Горловина ---
st.subheader("Расчёт круглой горловины")

neck_stitches = round(stitches_total * 0.25)  # допустим, 25% от груди
neck_rows = round(rows_total * 0.3)           # допустим, 30% от длины

st.write(f"Горловина: убавляем **{neck_stitches} петель** за **{neck_rows} рядов**")

neck_plan = calc_round_neckline(neck_stitches, neck_rows)

for row, dec in neck_plan:
    if dec > 0:
        st.write(f"Ряд {row}: убавить {dec} п.")
    else:
        st.write(f"Ряд {row}: без убавок")
