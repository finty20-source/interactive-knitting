import streamlit as st
import numpy as np
import pandas as pd
from collections import defaultdict

st.title("🧶 Интерактивное вязание — полный расчёт модели")

# -----------------------------
# Конвертеры
# -----------------------------
def cm_to_st(cm, dens_st):   
    return int(round((cm/10.0)*dens_st))

def cm_to_rows(cm, dens_row):
    return int(round((cm/10.0)*dens_row))

# -----------------------------
# Рядовые правила
# -----------------------------
def allowed_even_rows(start_row: int, end_row: int, rows_total: int):
    """Разрешённые чётные ряды: ≥6 и ≤ rows_total-2."""
    if end_row is None:
        end_row = rows_total
    high = min(end_row, rows_total - 2)
    if high < 6:
        return []
    start = max(6, start_row)
    if start % 2 == 1: start += 1
    if high  % 2 == 1: high  -= 1
    return list(range(start, high + 1, 2)) if start <= high else []

def split_total_into_steps(total: int, steps: int):
    if total <= 0 or steps <= 0:
        return []
    steps = min(steps, total)
    base = total // steps
    rem  = total % steps
    return [base + (1 if i < rem else 0) for i in range(steps)]

# -----------------------------
# Симметричные прибавки / убавки
# -----------------------------
def sym_increases(total_add, start_row, end_row, rows_total, label):
    if total_add <= 0: return []
    if total_add % 2 == 1: total_add += 1
    rows = allowed_even_rows(start_row, end_row, rows_total)
    if not rows: return []
    per_side = total_add // 2
    steps = min(len(rows), per_side)
    parts = split_total_into_steps(per_side, steps)
    idxs = np.linspace(0, len(rows)-1, num=steps, dtype=int)
    chosen = [rows[i] for i in idxs]
    return [(r, f"+{v} п. {label} (с каждой стороны)") for r, v in zip(chosen, parts)]

def sym_decreases(total_sub, start_row, end_row, rows_total, label):
    if total_sub <= 0: return []
    if total_sub % 2 == 1: total_sub += 1
    rows = allowed_even_rows(start_row, end_row, rows_total)
    if not rows: return []
    per_side = total_sub // 2
    steps = min(len(rows), per_side)
    parts = split_total_into_steps(per_side, steps)
    idxs = np.linspace(0, len(rows)-1, num=steps, dtype=int)
    chosen = [rows[i] for i in idxs]
    return [(r, f"-{v} п. {label} (с каждой стороны)") for r, v in zip(chosen, parts)]

# -----------------------------
# Скос плеча
# -----------------------------
def slope_shoulder(total_stitches, start_row, end_row, rows_total):
    if total_stitches <= 0: return []
    rows = allowed_even_rows(start_row, end_row, rows_total)
    if not rows: return []
    steps = len(rows)
    base = total_stitches // steps
    rem  = total_stitches % steps
    actions = []
    for i, r in enumerate(rows):
        dec = base + (1 if i < rem else 0)
        actions.append((r, f"-{dec} п. скос плеча (одно плечо)"))
    return actions

# -----------------------------
# Горловина (круглая)
# -----------------------------
def calc_round_neckline(total_stitches, total_rows, start_row, rows_total, straight_percent=0.05):
    if total_stitches <= 0 or total_rows <= 0:
        return []

    # первый шаг = 60% и доводим до четного
    first_dec = int(round(total_stitches * 0.60))
    if first_dec % 2 == 1:
        first_dec += 1
    rest = total_stitches - first_dec

    straight_rows = max(2, int(round(total_rows * straight_percent)))
    neck_end_by_depth = start_row + total_rows - 1 - straight_rows
    effective_end = min(neck_end_by_depth, rows_total - 2)

    rows = allowed_even_rows(start_row, effective_end, rows_total)
    if not rows:
        return []

    actions = []
    actions.append((rows[0], f"-{first_dec} п. горловина (середина, разделение на плечи)"))

    if rest <= 0 or len(rows) == 1:
        return actions

    rest_rows = rows[1:]
    steps = min(len(rest_rows), rest)
    idxs   = np.linspace(0, len(rest_rows)-1, num=steps, dtype=int)
    chosen = [rest_rows[i] for i in idxs]
    parts  = split_total_into_steps(rest, steps)

    # последние 2 шага <= 1 петли
    if steps >= 2:
        over = 0
        for i in [steps-2, steps-1]:
            if parts[i] > 1:
                over += parts[i] - 1
                parts[i] = 1
        jmax = max(1, steps-2)
        j = 0
        while over > 0 and jmax > 0:
            parts[j % jmax] += 1
            over -= 1
            j += 1

    for r, v in zip(chosen, parts):
        actions.append((r, f"-{v} п. горловина (каждое плечо)"))
    return actions

# -----------------------------
# Таблица + сегменты
# -----------------------------
def section_tags(row, rows_to_armhole_end, neck_start_row, shoulder_start_row):
    tags = []
    # 1) низ до линии проймы
    if row <= rows_to_armhole_end:
        tags.append("Низ изделия")
    # 2) пройма между линией проймы и началом плеча (включая пересечения)
    if rows_to_armhole_end < row < shoulder_start_row:
        tags.append("Пройма")
    # 3) горловина может начаться до плеча и продолжаться на плечах
    if neck_start_row and row >= neck_start_row:
        tags.append("Горловина")
    # 4) скос плеча — самый верх, не пересекается с проймой
    if shoulder_start_row and row >= shoulder_start_row:
        tags.append("Скос плеча")
    return " + ".join(tags) if tags else "—"

def make_table_full(actions, rows_total, rows_to_armhole_end, neck_start_row, shoulder_start_row):
    merged = defaultdict(list)
    for row, note in actions:
        merged[row].append(note)

    rows_sorted = sorted(merged.keys())
    table_rows = []
    prev = 1

    for r in rows_sorted:
        # внутри make_table_full
if r > prev:
    seg = section_tags(prev, rows_to_armhole_end, neck_start_row, shoulder_start_row)
    table_rows.append((f"{prev}-{r-1}", "Прямо", seg))
    
                           section_tags(r, rows_to_armhole_end, neck_start_row, shoulder_start_row)))
        prev = r + 1

if prev <= rows_total:
    seg = section_tags(prev, rows_to_armhole_end, neck_start_row, shoulder_start_row)
    table_rows.append((f"{prev}-{rows_total}", "Прямо", seg))

    df = pd.DataFrame(table_rows, columns=["Ряды", "Действия", "Сегмент"])
    st.dataframe(df, use_container_width=True, hide_index=True)

# -----------------------------
# Ввод параметров
# -----------------------------
st.header("Перед / Спинка")

density_st  = st.number_input("Плотность: петли в 10 см", 1, 999, 23)
density_row = st.number_input("Плотность: ряды в 10 см",  1, 999, 40)

hip_cm      = st.number_input("Ширина низа детали (см)", 50, 200, 80)
length_cm   = st.number_input("Длина изделия (см)", 30, 120, 55)

armhole_depth_cm = st.number_input("Длина проймы (см)", 10, 40, 23)

neck_width_cm    = st.number_input("Ширина горловины (см)", 5, 40, 18)
neck_depth_cm    = st.number_input("Глубина горловины спереди (см)", 1, 40, 6)
neck_depth_back_cm = st.number_input("Глубина горловины спинки (см)", 1, 20, 3)

shoulder_len_cm  = st.number_input("Длина плеча (см)", 5, 30, 12)
shoulder_slope_cm= st.number_input("Скос плеча (см)", 1, 20, 4)

if st.button("🔄 Рассчитать"):
    # в петли/ряды
    st_hip     = cm_to_st(hip_cm, density_st)
    rows_total = cm_to_rows(length_cm, density_row)
    rows_armh  = cm_to_rows(armhole_depth_cm, density_row)

    neck_st    = cm_to_st(neck_width_cm, density_st)
    neck_rows_front  = cm_to_rows(neck_depth_cm, density_row)
    neck_rows_back   = cm_to_rows(neck_depth_back_cm, density_row)

    st_shldr   = cm_to_st(shoulder_len_cm, density_st)
    rows_slope = cm_to_rows(shoulder_slope_cm, density_row)

    st_shoulders = 2 * st_shldr + neck_st
    st_chest = st_shoulders  # скрытый параметр для расчётов

    rows_to_armhole_end = rows_total - rows_armh
    neck_start_row_front= rows_total - neck_rows_front + 1
    neck_start_row_back = rows_total - neck_rows_back + 1
    shoulder_start_row  = rows_total - rows_slope + 1

    st.subheader("📊 Сводка")
    st.write(f"- Набрать петель: **{st_hip}**")
    st.write(f"- Всего рядов: **{rows_total}**")
    st.write(f"- Авто ширина по плечам: **{st_shoulders} п.** (= 2×{st_shldr} + {neck_st})")

    # ----- ПЕРЕД -----
    st.subheader("📋 Инструкция для переда")
    actions = []
    delta_bottom = st_chest - st_hip
    if delta_bottom > 0:
        actions += sym_increases(delta_bottom, 6, rows_to_armhole_end, rows_total, "бок")
    elif delta_bottom < 0:
        actions += sym_decreases(-delta_bottom, 6, rows_to_armhole_end, rows_total, "бок")

    delta_armh = st_shoulders - st_chest
    armhole_start_row = rows_to_armhole_end + 1
    armhole_end_row   = shoulder_start_row - 1
    if delta_armh > 0:
        actions += sym_increases(delta_armh, armhole_start_row, armhole_end_row, rows_total, "пройма")
    elif delta_armh < 0:
        actions += sym_decreases(-delta_armh, armhole_start_row, armhole_end_row, rows_total, "пройма")

    actions += calc_round_neckline(neck_st, neck_rows_front, neck_start_row_front, rows_total)
    actions += slope_shoulder(st_shldr, shoulder_start_row, rows_total, rows_total)

    make_table_full(actions, rows_total, rows_to_armhole_end, neck_start_row_front, shoulder_start_row)

    # ----- СПИНКА -----
    st.subheader("📋 Инструкция для спинки")
    actions_back = []
    delta_bottom = st_chest - st_hip
    if delta_bottom > 0:
        actions_back += sym_increases(delta_bottom, 6, rows_to_armhole_end, rows_total, "бок")
    elif delta_bottom < 0:
        actions_back += sym_decreases(-delta_bottom, 6, rows_to_armhole_end, rows_total, "бок")

    if delta_armh > 0:
        actions_back += sym_increases(delta_armh, armhole_start_row, armhole_end_row, rows_total, "пройма")
    elif delta_armh < 0:
        actions_back += sym_decreases(-delta_armh, armhole_start_row, armhole_end_row, rows_total, "пройма")

    actions_back += calc_round_neckline(neck_st, neck_rows_back, neck_start_row_back, rows_total)
    actions_back += slope_shoulder(st_shldr, shoulder_start_row, rows_total, rows_total)

    make_table_full(actions_back, rows_total, rows_to_armhole_end, neck_start_row_back, shoulder_start_row)
