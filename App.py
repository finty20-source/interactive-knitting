import streamlit as st
import numpy as np

st.title("🧶 Интерактивное вязание — калькулятор свитера")

# -----------------------------
# 🧾 Ввод данных
# -----------------------------
st.header("Ввод параметров")

density_stitches = st.number_input("Плотность (петли в 10 см)", min_value=1, value=23)
density_rows = st.number_input("Плотность (ряды в 10 см)", min_value=1, value=40)

neck_width_cm = st.number_input("Ширина горловины (см)", min_value=5, value=18)
neck_depth_cm = st.number_input("Глубина горловины (см)", min_value=2, value=6)

st.write("---")

# -----------------------------
# 🔄 Пересчёт см → петли/ряды
# -----------------------------
st.header("Расчёт горловины")

neck_stitches = int((neck_width_cm / 10) * density_stitches)
neck_rows = int((neck_depth_cm / 10) * density_rows)

st.write(f"📏 Ширина горловины: {neck_width_cm} см → **{neck_stitches} петель**")
st.write(f"📐 Глубина горловины: {neck_depth_cm} см → **{neck_rows} рядов**")

# -----------------------------
# 📊 Логика убавок
# -----------------------------
def calc_round_neckline(total_stitches, total_rows):
    """
    Расчёт убавок для круглой горловины.
    total_stitches - сколько всего петель нужно убрать
    total_rows - высота горловины в рядах
    """

    # Пропорции распределения убавок
    percentages = [60, 20, 10, 5, 5]

    # Переводим проценты в количество петель
    decreases = [round(total_stitches * p / 100) for p in percentages]

    # План по рядам
    plan = []
    current_row = 1

    for d in decreases:
        if d > 0:
            plan.append((current_row, d))   # убавка в этом ряду
        current_row += 2                    # убавки через ряд

    # Остаток рядов → без убавок
    while current_row <= total_rows:
        plan.append((current_row, 0))
        current_row += 1

    return plan

# -----------------------------
# 📌 Итоговый план вязания
# -----------------------------
st.subheader("Пошаговый план горловины")

neck_plan = calc_round_neckline(neck_stitches, neck_rows)

for row, dec in neck_plan:
    if dec > 0:
        st.write(f"➡️ Ряд {row}: убавить {dec} п.")
    else:
        st.write(f"➡️ Ряд {row}: вязать прямо (без убавок)")

# -----------------------------
# ✅ Финальное резюме
# -----------------------------
st.write("---")
st.success(
    f"Горловина рассчитана: нужно убавить **{neck_stitches} петель** за **{neck_rows} рядов**.\n"
    "Последние несколько рядов провязываем прямо для плавного завершения."
)
