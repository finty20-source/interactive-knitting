import streamlit as st
import numpy as np
import pandas as pd
from collections import defaultdict

st.title("🧶 Интерактивное вязание — расчёт рукава (оверсайз, прямой окат)")

# -----------------------------
# Ввод параметров
# -----------------------------
st.header("Ввод параметров")

density_st = st.number_input("Плотность (петли в 10 см)", min_value=1, value=23)
density_row = st.number_input("Плотность (ряды в 10 см)", min_value=1, value=40)

length_cm   = st.number_input("Длина рукава (см)", min_value=20, value=60)
wrist_cm    = st.number_input("Ширина манжеты (см)", min_value=10, value=18)
top_cm      = st.number_input("Ширина рукава вверху (см)", min_value=20, value=36)

st.write("---")

# -----------------------------
# Функции
# -----------------------------
def cm_to_st(cm, dens_st):   
    return int(round((cm/10.0)*dens_st))

def cm_to_rows(cm, dens_row):
    return int(round((cm/10.0)*dens_row))

def spread_rows(start_row: int, end_row: int, count: int):
    """Равномерное распределение по рядам (не раньше 5-го ряда)."""
    if count <= 0 or end_row < start_row:
        return []
    start_row = max(5, start_row)  # правило: не раньше 5-го ряда
    if start_row == end_row:
        return [start_row]*count
    xs = np.linspace(start_row, end_row, num=count, endpoint=True)
    rows = sorted(set(int(round(x)) for x in xs))
    while len(rows) < count and rows[-1] < end_row:
        rows.append(rows[-1]+1)
    return rows[:count]

def distribute_side_increases(start_row, end_row, total_delta, label):
    """Прибавки симметричные → пара в одном ряду."""
    if total_delta <= 0 or end_row < start_row:
        return []
    if total_delta % 2 == 1:
        total_delta += 1  # всегда чётное
    pairs = total_delta // 2
    rows = spread_rows(start_row, end_row, pairs)
    return [(r, f"+1 п. {label} слева и +1 п. {label} справа") for r in rows]

# -----------------------------
# Кнопка запуска
# -----------------------------
if st.button("🔄 Рассчитать рукав"):
    # Пересчёт см → петли/ряды
    stitches_wrist = cm_to_st(wrist_cm, density_st)
    stitches_top   = cm_to_st(top_cm, density_st)
    rows_total     = cm_to_rows(length_cm, density_row)

    delta = stitches_top - stitches_wrist
    if delta % 2 == 1:
        delta += 1

    # -----------------------------
    # Генерация действий
    # -----------------------------
    actions = []

    # Прибавки от манжеты к верху
    actions += distribute_side_increases(5, rows_total - 1, delta, "рукав")

    # Закрытие оката (прямое)
    actions.append((rows_total, "Закрыть все петли (прямой окат)"))

    # -----------------------------
    # Схлопывание
    # -----------------------------
    merged = defaultdict(list)
    for row, note in actions:
        merged[row].append(note)

    # -----------------------------
    # Таблица
    # -----------------------------
    st.header("Пошаговый план для рукава")

    if not merged:
        st.info("Нет действий для выбранных параметров.")
    else:
        rows_sorted = sorted(merged.keys())
        data = {
            "Ряд": rows_sorted,
            "Действия": [", ".join(merged[r]) for r in rows_sorted],
            "Сегмент": ["Рукав" if r < rows_total else "Окат (прямой)" for r in rows_sorted]
        }
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.write("---")
    st.success(
        f"Итого рядов: {rows_total}. "
        f"От манжеты ({stitches_wrist} п.) до верха ({stitches_top} п.)."
    )
