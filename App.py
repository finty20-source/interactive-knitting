import streamlit as st
import numpy as np
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
rows_to_armhole_start = 1
rows_to_armhole_end   = max(0, rows_total - rows_armhole)  # от низа до старта проймы (включительно)
shoulder_start_row    = max(1, rows_total - rows_shoulder_slope + 1)  # первый ряд скоса плеча
neck_start_row        = max(1, rows_total - neck_rows + 1)            # первый ряд горловины

# пройма должна закончиться ДО скоса плеча
armhole_start_row = rows_to_armhole_end + 1
armhole_end_row   = max(armhole_start_row-1, min(rows_total, shoulder_start_row - 1))  # если скос начинается сразу, проймы не будет

# Расширение на пройме (оверсайз): считаем как разницу между верхней шириной и грудью
armhole_extra_st_total = max(0, stitches_shoulders - stitches_chest)

# -----------------------------
# Утилиты распределения по рядам
# -----------------------------
def spread_rows(start_row: int, end_row: int, count: int):
    """
    Возвращает строго неубывающую последовательность row-ов длиной count
    равномерно распределённую внутри [start_row, end_row].
    Гарантирует уникальность, где возможно.
    """
    if count <= 0 or end_row < start_row:
        return []
    if start_row == end_row:
        return [start_row]*count
    xs = np.linspace(start_row, end_row, num=count, endpoint=True)
    rows = [int(round(x)) for x in xs]
    # устраняем дубликаты, сдвигая вверх в пределах end_row
    used = set()
    for i in range(len(rows)):
        r = rows[i]
        while r in used and r < end_row:
            r += 1
        # если упёрлись в конец и дубликат — попробуем сдвинуть назад
        while r in used and r > start_row:
            r -= 1
        rows[i] = r
        used.add(r)
    rows.sort()
    return rows

def distribute_side_increases(start_row, end_row, total_delta, label_left, label_right):
    """
    Раскладываем прибавки по ЛЕВОЙ/ПРАВОЙ стороне отдельно,
    чтобы суммарно получить total_delta петель.
    """
    if total_delta <= 0 or end_row < start_row:
        return []
    left_cnt  = total_delta // 2
    right_cnt = total_delta - left_cnt
    rows_left  = spread_rows(start_row, end_row, left_cnt)
    rows_right = spread_rows(start_row, end_row, right_cnt)
    out = []
    for r in rows_left:
        out.append((r, f"+1 п. {label_left}"))
    for r in rows_right:
        out.append((r, f"+1 п. {label_right}"))
    return out

def calc_round_neckline_by_percent(total_stitches, total_rows, start_row, percentages=(60,20,10,5,5)):
    """
    Горловина: крупные убавки в первый приём, затем реже, через ряд.
    Возвращает пары (row, '-X п. горловина').
    """
    if total_stitches <= 0 or total_rows <= 0:
        return []
    # пересчёт процентов в целые петли
    parts = [int(round(total_stitches * p / 100.0)) for p in percentages]
    # подправим сумму на всякий случай
    diff = total_stitches - sum(parts)
    if diff != 0:
        # распределим остаток по крупнейшим порциям
        sign = 1 if diff > 0 else -1
        for i in range(abs(diff)):
            idx = i % len(parts)
            parts[idx] += sign
    # размещаем убавки "через ряд" в пределах блока горловины
    actions = []
    row = start_row
    for dec in parts:
        if dec > 0 and row <= start_row + total_rows - 1:
            actions.append((row, f"-{dec} п. горловина"))
        row += 2
    # если вылезли за пределы, больше не добавляем
    return actions

def slope_shoulder_steps(total_stitches, start_row, end_row, steps=3):
    """
    Классический скос плеча: закрываем в несколько ступеней равными долями.
    """
    if total_stitches <= 0 or end_row < start_row or steps <= 0:
        return []
    base = total_stitches // steps
    parts = [base]*steps
    parts[-1] += (total_stitches - base*steps)  # остаток в последнюю ступень
    rows = spread_rows(start_row, end_row, steps)
    return [(r, f"закрыть {p} п. плечо") for r, p in zip(rows, parts)]

# -----------------------------
# План действий (единый список)
# -----------------------------
actions = []

# 1) Низ → грудь: боковые прибавки (не пересекаются с проймой)
delta_bottom = max(0, stitches_chest - stitches_hip)
actions += distribute_side_increases(
    rows_to_armhole_start, rows_to_armhole_end,
    delta_bottom,
    label_left  = "(бок, левая) от низа к груди",
    label_right = "(бок, правая) от низа к груди",
)

# 2) Пройма (оверсайз): прибавки по бокам, но ТОЛЬКО до начала скоса плеча
if armhole_start_row <= armhole_end_row:
    actions += distribute_side_increases(
        armhole_start_row, armhole_end_row,
        armhole_extra_st_total,
        label_left  = "(пройма, левая)",
        label_right = "(пройма, правая)",
    )

# 3) Горловина: может пересекаться с проймой и/или со скосом плеча
actions += calc_round_neckline_by_percent(
    neck_stitches, neck_rows, neck_start_row, percentages=(60,20,10,5,5)
)

# 4) Скос плеча: начинается после окончания проймы
if shoulder_start_row <= rows_total:
    actions += slope_shoulder_steps(
        stitches_shoulder,
        start_row=shoulder_start_row,
        end_row=rows_total,
        steps=3
    )

# -----------------------------
# Схлопываем по рядам
# -----------------------------
merged = defaultdict(list)
for row, note in actions:
    merged[row].append(note)

# -----------------------------
# Вывод
# -----------------------------
st.header("Единый пошаговый план (ряд → действия)")

if not merged:
    st.info("Нет действий для выбранных параметров. Проверьте мерки.")
else:
    for row in sorted(merged.keys()):
        st.write(f"➡️ Ряд {row}: " + ", ".join(merged[row]))

st.write("---")
st.caption(
    f"Сегменты: низ→грудь: 1..{rows_to_armhole_end}; "
    f"пройма: {armhole_start_row}..{max(armhole_start_row, armhole_end_row)}; "
    f"горловина: {neck_start_row}..{rows_total}; "
    f"скос плеча: {shoulder_start_row}..{rows_total}."
)
