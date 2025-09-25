import streamlit as st
import numpy as np

st.title("🧶 Интерактивное вязание — калькулятор свитера")

# --- Ввод параметров ---
st.header("Ввод данных")

density_stitches = st.number_input("Плотность (петли в 10 см)", min_value=1, value=23)
density_rows = st.number_input("Плотность (ряды в 10 см)", min_value=1, value=40)

chest = st.number_input("Обхват груди (см)", min_value=50, value=90)
length = st.number_input("Длина изделия (см)", min_value=30, value=55)

# 🔹 Новые параметры для горловины
neck_width_cm = st.number_input("Ширина горловины (см)", min_value=5, value=18)
neck_depth_cm = st.number_input("Глубина горловины (см)", min_value=2, value=6)

st.write("---")

# --- Пересчёт ---
st.header("Базовые расчёты")

stitches_total = int((chest / 10) * density_stitches)
rows_total = int((length / 10) * density_rows)

st.write(f"🔹 Общее количество петель по груди: **{stitches_total}**")
st.write(f"🔹 Общее количество рядов по длине: **{rows_total}**")

# перевод горловины из см в петли/ряды
neck_stitches = int((neck_width_cm / 10) * density_stitches)
neck_rows = int((neck_depth_cm / 10) * density_rows)

st.write(f"🔹 Ширина горловины: {neck_width_cm} см = **{neck_stitches} п.**")
st.write(f"🔹 Глубина горловины: {neck_depth_cm} см = **{neck_rows} р.**")

# --- Функция для круглой горловины ---
def calc_round_neckline(total_stitches, total_rows):
    """
    Расчёт убавок для круглой горловины.
    total_stitches - сколько всего петель нужно убрать
    total_rows - высота горловины в рядах
    """

    # пропорции убавок (сумма = 100%)
    percentages = [60, 20, 10, 5, 5]

    # пересч
