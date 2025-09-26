import streamlit as st
import numpy as np
import pandas as pd
from collections import defaultdict

st.title("🧶 Интерактивное вязание — полный расчёт")

# -----------------------------
# Конвертеры
# -----------------------------
def cm_to_st(cm, dens_st):   return int(round((cm/10.0)*dens_st))
def cm_to_rows(cm, dens_row):return int(round((cm/10.0)*dens_row))

# -----------------------------
# Рядовые правила
# -----------------------------
def allowed_even_rows(start_row: int, end_row: int, rows_total: int):
    """
    Разрешённые чётные ряды:
    - не раньше 6-го,
    - не позже rows_total-2 (последние 2 ряда — прямо).
    """
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
    """Разбить total на steps положительных целых частей (разница ≤1, сумма == total)."""
    if total <= 0 or steps <= 0:
        return []
    steps = min(steps, total)  # избегаем нулевых шагов
    base = total // steps
    rem  = total % steps
    return [base + (1 if i < rem else 0) for i in range(steps)]

# -----------------------------
# Прибавки / убавки симметрично
# -----------------------------
def sym_increases(total_add: int, start_row: int, end_row: int, rows_total: int, label: str):
    """
    Симметричные прибавки: total_add — суммарно по полотну.
    В строке пишем: +v слева и +v справа.
    """
    if total_add <= 0: return []
    if total_add % 2 == 1: total_add += 1  # деталь симметрична — чётное число петель
    rows = allowed_even_rows(start_row, end_row, rows_total)
    if not rows: return []
    per_side = total_add // 2
    steps = min(len(rows), per_side)
    parts = split_total_into_steps(per_side, steps)
    idxs = np.linspace(0, len(rows)-1, num=steps, dtype=int)
    chosen = [rows[i] for i in idxs]
    return [(r, f"+{v} п. {label} слева и +{v} п. {label} справа") for r, v in zip(chosen, parts)]

def sym_decreases(total_sub: int, start_row: int, end_row: int, rows_total: int, label: str):
    """
    Симметричные убавки: total_sub — суммарно по полотну.
    В строке пишем: -v слева и -v справа.
    """
    if total_sub <= 0: return []
    if total_sub % 2 == 1: total_sub += 1
    rows = allowed_even_rows(start_row, end_row, rows_total)
    if not rows: return []
    per_side = total_sub // 2
    steps = min(len(rows), per_side)
    parts = split_total_into_steps(per_side, steps)
    idxs = np.linspace(0, len(rows)-1, num=steps, dtype=int)
    chosen = [rows[i] for i in idxs]
    return [(r, f"-{v} п. {label} слева и -{v} п. {label} справа") for r, v in zip(chosen, parts)]

# -----------------------------
# Скос плеча
# -----------------------------
def slope_shoulder(total_stitches: int, start_row: int, end_row: int, rows_total: int):
    """
    Скос плеча равномерно по всем доступным чётным рядам; остаток — в начале.
    total_stitches — суммарно для одного плеча.
    """
    if total_stitches <= 0: return []
    rows = allowed_even_rows(start_row, end_row, rows_total)
    if not rows: return []
    steps = len(rows)
    base = total_stitches // steps
    rem  = total_stitches % steps
    actions = []
    for i, r in enumerate(rows):
        dec = base + (1 if i < rem else 0)
        actions.append((r, f"-{dec} п. плечо (каждое плечо)"))
    return actions

# -----------------------------
# Горловина (круглая)
# -----------------------------
def calc_round_neckline(total_stitches: int, total_rows: int, start_row: int, rows_total: int, straight_percent: float = 0.05):
    """
    Правила:
    - 1-й шаг = 60% петель горловины (центральное закрытие) — в первом доступном чётном ряду.
    - Верхние straight_percent*глубины (но ≥2 ряда) — вяжутся прямо (без убавок).
    - Остаток петель равномерно по чётным рядам между 1-м шагом и верхней «прямой» зоной.
    - Последние 2 убавочных ряда ≤ 1 п.; «лишнее» равномерно раздаётся вниз (к более ранним шагам).
    - Манипуляции не попадают в последние 2 ряда полотна.
    """
    if total_stitches <= 0 or total_rows <= 0: return []
    first_dec = int(round(total_stitches * 0.60))
    rest = total_stitches - first_dec

    straight_rows = max(2, int(round(total_rows * straight_percent)))
    neck_end_by_depth = start_row + total_rows - 1 - straight_rows
    effective_end = min(neck_end_by_depth, rows_total - 2)

    rows = allowed_even_rows(start_row, effective_end, rows_total)
    if not rows: return []

    actions = []
    # Первый шаг (центральное закрытие, разделение на плечи)
    actions.append((rows[0], f"-{first_dec} п. горловина (середина, разделение на плечи)"))

    if rest <= 0 or len(rows) == 1:
        return actions

    rest_rows = rows[1:]
    steps = min(len(rest_rows), rest)
    if steps == 2 and rest > 2 and len(rest_rows) >= 3:
        steps = 3

    idxs   = np.linspace(0, len(rest_rows)-1, num=steps, dtype=int)
    chosen = [rest_rows[i] for i in idxs]
    parts  = split_total_into_steps(rest, steps)

    # Сглаживание: последние 2 убавочных ряда ≤ 1 п.
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
        actions.append((r, f"-{v} п. горловина (каждое плечо отдельно)"))
    return actions

# -----------------------------
# Таблица + сегменты
# -----------------------------
def section_tags(row, rows_to_armhole_end, neck_start_row, shoulder_start_row):
    tags = []
    if row <= rows_to_armhole_end:
        tags.append("Низ изделия")
    if rows_to_armhole_end+1 <= row < shoulder_start_row:
        tags.append("Пройма")
    if neck_start_row and row >= neck_start_row:
        tags.append("Горловина")
    if shoulder_start_row and row >= shoulder_start_row:
        tags.append("Скос плеча")
    return ", ".join(tags) if tags else "—"

def make_table(actions, rows_total, rows_to_armhole_end, neck_start_row, shoulder_start_row):
    merged = defaultdict(list)
    for row, note in actions:
        merged[row].append(note)
    rows_sorted = sorted(merged.keys())
    data = {
        "Ряд": rows_sorted,
        "Действия": ["; ".join(merged[r]) for r in rows_sorted],
        "Сегмент": [
            section_tags(r, rows_to_armhole_end, neck_start_row, shoulder_start_row)
            for r in rows_sorted
        ]
    }
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)

# -----------------------------
# UI
# -----------------------------
st.header("Перед / Спинка")

density_st  = st.number_input("Плотность: петли в 10 см", 1, 999, 23)
density_row = st.number_input("Плотность: ряды в 10 см",  1, 999, 40)

hip_cm      = st.number_input("Ширина низа (см)", 50, 200, 80)
chest_cm    = st.number_input("Ширина груди (см)", 50, 200, 90)
length_cm   = st.number_input("Длина изделия (см)", 30, 120, 55)

armhole_depth_cm = st.number_input("Глубина проймы (см)", 10, 40, 23)

neck_width_cm    = st.number_input("Ширина горловины (см)", 5, 40, 18)
neck_depth_cm    = st.number_input("Глубина горловины (см)", 1, 40, 6)

shoulder_len_cm  = st.number_input("Длина одного плеча (см)", 5, 30, 12)
shoulder_slope_cm= st.number_input("Высота скоса плеча (см)", 1, 20, 4)

if st.button("🔄 Рассчитать"):
    # В петли/ряды
    st_hip     = cm_to_st(hip_cm, density_st)
    st_chest   = cm_to_st(chest_cm, density_st)
    rows_total = cm_to_rows(length_cm, density_row)
    rows_armh  = cm_to_rows(armhole_depth_cm, density_row)

    neck_st    = cm_to_st(neck_width_cm, density_st)
    neck_rows  = cm_to_rows(neck_depth_cm, density_row)

    st_shldr   = cm_to_st(shoulder_len_cm, density_st)     # ширина одного плеча (петли)
    rows_slope = cm_to_rows(shoulder_slope_cm, density_row)

    # Ширина по плечам (АВТО): 2 × плечо + горловина (в петлях)
    st_shoulders = 2 * st_shldr + neck_st

    # Ключевые ряды
    rows_to_armhole_end = rows_total - rows_armh
    neck_start_row      = rows_total - neck_rows + 1
    shoulder_start_row  = rows_total - rows_slope + 1

    actions = []

    # 1) Низ → грудь (разница по размерам)
    delta_bottom = st_chest - st_hip
    if delta_bottom > 0:
        actions += sym_increases(delta_bottom, 6, rows_to_armhole_end, rows_total, "бок")
    elif delta_bottom < 0:
        actions += sym_decreases(-delta_bottom, 6, rows_to_armhole_end, rows_total, "бок")

    # 2) Пройма (ВАЖНО: по твоему правилу)
    # Δпроймы = ширина по плечам - ширина груди
    delta_armh = st_shoulders - st_chest
    armhole_start_row = rows_to_armhole_end + 1
    armhole_end_row   = shoulder_start_row - 1   # пройма не пересекается со скосом плеча
    if delta_armh > 0:
        actions += sym_increases(delta_armh, armhole_start_row, armhole_end_row, rows_total, "пройма")
    elif delta_armh < 0:
        actions += sym_decreases(-delta_armh, armhole_start_row, armhole_end_row, rows_total, "пройма")

    # 3) Горловина (круглая)
    actions += calc_round_neckline(neck_st, neck_rows, neck_start_row, rows_total)

    # 4) Скос плеча (только после проймы)
    actions += slope_shoulder(st_shldr, shoulder_start_row, rows_total, rows_total)

    # Служебный вывод: показать рассчитанную ширину по плечам
    st.info(f"Авто ширина по плечам: {st_shoulders} п. ( = 2×{st_shldr} + {neck_st} )")

    st.subheader("📋 Инструкция")
    make_table(actions, rows_total,
               rows_to_armhole_end, neck_start_row, shoulder_start_row)
