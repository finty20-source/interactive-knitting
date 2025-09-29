import streamlit as st
import numpy as np
import pandas as pd
from collections import defaultdict

st.title("🧶 Интерактивное вязание — полный расчёт модели")

# -----------------------------
# Сессия для хранения результатов
# -----------------------------
if "actions" not in st.session_state:
    st.session_state.actions = []
    st.session_state.actions_back = []
    st.session_state.st_hip = 0
    st.session_state.rows_total = 0
    st.session_state.rows_bottom = 0
    st.session_state.table_front = []
    st.session_state.table_back = []

# -----------------------------
# Конвертеры
# -----------------------------
def cm_to_st(cm, dens_st):
    return int(round((cm / 10.0) * dens_st))

def cm_to_rows(cm, dens_row):
    return int(round((cm / 10.0) * dens_row))

# -----------------------------
# Рядовые правила
# -----------------------------
def allowed_even_rows(start_row, end_row, rows_total):
    if end_row is None:
        end_row = rows_total
    if end_row < 6:
        return []
    start = max(6, start_row)
    return list(range(start, end_row + 1))

def split_total_into_steps(total, steps):
    if total <= 0 or steps <= 0:
        return []
    steps = min(steps, total)
    base = total // steps
    rem = total % steps
    return [base + (1 if i < rem else 0) for i in range(steps)]

# -----------------------------
# Симметричные прибавки/убавки
# -----------------------------
def sym_increases(total_add, start_row, end_row, rows_total, label):
    if total_add <= 0:
        return []
    if total_add % 2 == 1:
        total_add += 1
    rows = allowed_even_rows(start_row, end_row, rows_total)
    if not rows:
        return []
    per_side = total_add // 2
    steps = min(len(rows), per_side)
    parts = split_total_into_steps(per_side, steps)
    idxs = np.linspace(0, len(rows) - 1, num=steps, dtype=int)
    chosen = [rows[i] for i in idxs]
    out = []
    for r, v in zip(chosen, parts):
        out.append((r, f"+{v} п. {label} (справа)"))
        out.append((r, f"+{v} п. {label} (слева)"))
    return out

def sym_decreases(total_sub, start_row, end_row, rows_total, label):
    if total_sub <= 0:
        return []
    if total_sub % 2 == 1:
        total_sub += 1
    rows = allowed_even_rows(start_row, end_row, rows_total)
    if not rows:
        return []
    per_side = total_sub // 2
    steps = min(len(rows), per_side)
    parts = split_total_into_steps(per_side, steps)
    idxs = np.linspace(0, len(rows) - 1, num=steps, dtype=int)
    chosen = [rows[i] for i in idxs]
    out = []
    for r, v in zip(chosen, parts):
        out.append((r, f"-{v} п. {label} (справа)"))
        out.append((r, f"-{v} п. {label} (слева)"))
    return out

# -----------------------------
# Пройма
# -----------------------------
def calc_round_armhole(st_chest, st_shoulders, start_row, shoulder_start_row, rows_total):
    if shoulder_start_row <= start_row:
        return []
    end_row = shoulder_start_row - 1
    total_rows = end_row - start_row + 1
    if total_rows <= 0:
        return []
    depth_armhole_st = int(round(st_chest * 0.05))
    st_mid = st_chest - depth_armhole_st
    rows_smooth = int(total_rows * 0.4)
    rows_hold = int(total_rows * 0.1)
    actions = []
    delta1 = st_mid - st_chest
    if delta1 < 0:
        actions += sym_decreases(-delta1, start_row, start_row + rows_smooth, rows_total, "пройма")
    delta2 = st_shoulders - st_mid
    if delta2 > 0:
        actions += sym_increases(delta2, start_row + rows_smooth + rows_hold, end_row, rows_total, "пройма")
    return actions

# -----------------------------
# Горловина
# -----------------------------
def calc_round_neckline(total_stitches, total_rows, start_row, rows_total, straight_spec=0.20):
    if total_stitches <= 0 or total_rows <= 0:
        return []
    first_dec = int(round(total_stitches * 0.60))
    if first_dec % 2 == 1:
        first_dec += 1
    rest = total_stitches - first_dec
    straight_rows = max(2, int(round(total_rows * straight_spec)))
    neck_end_by_depth = start_row + total_rows - 1 - straight_rows
    effective_end = min(neck_end_by_depth, rows_total - 2)
    rows = allowed_even_rows(start_row, effective_end, rows_total)
    if not rows:
        return []
    actions = [(rows[0], f"-{first_dec} п. горловина (середина, разделение на плечи)")]
    if rest <= 0 or len(rows) == 1:
        return actions
    rest_rows = rows[1:]
    steps = min(len(rest_rows), rest)
    idxs = np.linspace(0, len(rest_rows) - 1, num=steps, dtype=int)
    chosen = [rest_rows[i] for i in idxs]
    parts = split_total_into_steps(rest, steps)
    for r, v in zip(chosen, parts):
        actions.append((r, f"-{v} п. горловина"))
    return actions

# -----------------------------
# Скос плеча
# -----------------------------
def slope_shoulders(total_stitches, start_row, end_row, rows_total):
    """
    Левое плечо → нечётные ряды,
    Правое плечо → чётные ряды.
    """
    if total_stitches <= 0:
        return [], []
    rows = list(range(start_row, end_row + 1))
    steps = len(rows)
    base = total_stitches // steps
    rem = total_stitches % steps
    left_actions, right_actions = [], []
    for i, r in enumerate(rows):
        dec = base + (1 if i < rem else 0)
        left_row = r if r % 2 == 1 else r + 1
        right_row = r if r % 2 == 0 else r + 1
        if left_row <= rows_total:
            left_actions.append((left_row, f"-{dec} п. скос плеча [L]"))
        if right_row <= rows_total:
            right_actions.append((right_row, f"-{dec} п. скос плеча [R]"))
    return left_actions, right_actions

# -----------------------------
# Горловина + плечи
# -----------------------------
def plan_neck_and_shoulders_split(neck_st, neck_rows, neck_start_row, st_shoulders, shoulder_start_row, rows_total, straight_percent=0.20):
    actions = []
    actions += calc_round_neckline(neck_st, neck_rows, neck_start_row, rows_total, straight_spec=straight_percent)
    actions_left, actions_right = slope_shoulders(st_shoulders // 2, shoulder_start_row, rows_total, rows_total)
    actions += actions_left + actions_right
    return actions

# -----------------------------
# Таблицы
# -----------------------------
def section_tags(row, rows_to_armhole_end, neck_start_row, shoulder_start_row):
    tags = []
    if row <= rows_to_armhole_end:
        tags.append("Низ изделия")
    if rows_to_armhole_end < row < shoulder_start_row:
        tags.append("Пройма")
    if neck_start_row and row >= neck_start_row:
        tags.append("Горловина")
    if shoulder_start_row and row >= shoulder_start_row:
        tags.append("Скос плеча")
    return " + ".join(tags) if tags else "—"

def make_table_front_split(actions, rows_count, rows_to_armhole_end, neck_start_row, shoulder_start_row, key=None):
    merged = defaultdict(list)
    for row, note in actions:
        if 1 <= row <= rows_count:
            merged[row].append(note)
    rows_sorted = sorted(merged.keys())
    table_rows = []
    for r in rows_sorted:
        table_rows.append((str(r), "; ".join(merged[r]), section_tags(r, rows_to_armhole_end, neck_start_row, shoulder_start_row)))
    df = pd.DataFrame(table_rows, columns=["Ряды", "Действия", "Сегмент"])
    st.dataframe(df, use_container_width=True, hide_index=True)
    if key:
        st.session_state[key] = table_rows

# -----------------------------
# Ввод параметров
# -----------------------------
st.header("Перед / Спинка")

density_st_str = st.text_input("Плотность: петли в 10 см", placeholder="введите плотность")
density_row_str = st.text_input("Плотность: ряды в 10 см", placeholder="введите плотность")
hip_cm_str = st.text_input("Ширина низа детали (см)", placeholder="введите ширину")
chest_cm_str = st.text_input("Ширина детали по груди (см)", placeholder="введите ширину")
length_cm_str = st.text_input("Длина изделия (см)", placeholder="введите длину")
armhole_depth_cm_str = st.text_input("Длина проймы (см)", placeholder="введите длину")
neck_width_cm_str = st.text_input("Ширина горловины (см)", placeholder="введите ширину")
neck_depth_cm_str = st.text_input("Глубина горловины спереди (см)", placeholder="введите глубину")
neck_depth_back_cm_str = st.text_input("Глубина горловины спинки (см)", placeholder="введите глубину")
shoulder_len_cm_str = st.text_input("Длина плеча (см)", placeholder="введите длину")
shoulder_slope_cm_str = st.text_input("Скос плеча (см)", placeholder="введите высоту")

# -----------------------------
# Кнопка
# -----------------------------
if st.button("🔄 Рассчитать"):
    st_hip = cm_to_st(float(hip_cm_str), float(density_st_str))
    if st_hip % 2:
        st_hip += 1
    st_chest = cm_to_st(float(chest_cm_str), float(density_st_str))
    rows_total = cm_to_rows(float(length_cm_str), float(density_row_str))
    rows_armh = cm_to_rows(float(armhole_depth_cm_str), float(density_row_str))
    neck_st = cm_to_st(float(neck_width_cm_str), float(density_st_str))
    neck_rows_front = cm_to_rows(float(neck_depth_cm_str), float(density_row_str))
    neck_rows_back = cm_to_rows(float(neck_depth_back_cm_str), float(density_row_str))
    st_shldr = cm_to_st(float(shoulder_len_cm_str), float(density_st_str))
    rows_slope = cm_to_rows(float(shoulder_slope_cm_str), float(density_row_str))
    st_shoulders = 2 * st_shldr + neck_st
    rows_bottom = rows_total - rows_armh - rows_slope
    armhole_start_row = rows_bottom + 1
    shoulder_start_row = rows_total - rows_slope + 1
    neck_start_row_front = rows_total - neck_rows_front + 1
    neck_start_row_back = rows_total - neck_rows_back + 1
    actions = []
    delta_bottom = st_chest - st_hip
    if delta_bottom > 0:
        actions += sym_increases(delta_bottom, 6, rows_bottom, rows_total, "бок")
    elif delta_bottom < 0:
        actions += sym_decreases(-delta_bottom, 6, rows_bottom, rows_total, "бок")
    actions += calc_round_armhole(st_chest, st_shoulders, armhole_start_row, shoulder_start_row, rows_total)
    actions += plan_neck_and_shoulders_split(neck_st, neck_rows_front, neck_start_row_front, st_shoulders, shoulder_start_row, rows_total)
    make_table_front_split(actions, rows_total, rows_bottom, neck_start_row_front, shoulder_start_row, key="table_front")
