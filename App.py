import streamlit as st
import numpy as np

st.title("🧶 Интерактивное вязание")

density_stitches = st.number_input("Плотность (петли в 10 см)", min_value=1, value=23)
density_rows = st.number_input("Плотность (ряды в 10 см)", min_value=1, value=40)
chest = st.number_input("Обхват груди (см)", min_value=50, value=90)
length = st.number_input("Длина изделия (см)", min_value=30, value=55)

stitches_total = int((chest / 10) * density_stitches)
rows_total = int((length / 10) * density_rows)

st.write(f"🔹 Общее количество петель: **{stitches_total}**")
st.write(f"🔹 Общее количество рядов: **{rows_total}**")

def calc_neckline(total_rows, total_decreases):
    x = np.linspace(0, total_rows, total_decreases)
    return [(int(r), 1) for r in np.round(x).astype(int)]

neck_calc = calc_neckline(40, 8)
st.subheader("Пример горловины (убавки)")
for row, dec in neck_calc:
    st.write(f"Ряд {row}: убавить {dec} петлю")
