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
def calc_round_neckline(total_stitches, total_rows, start_row, rows_total, shoulder_start_row):
    """Округлая горловина: первая убавка 60%, затем распределённые убавки.
       Последние 20% рядов идут прямо, но не позже начала плеча."""
    if total_stitches <= 0 or total_rows <= 0:
        return []

    # первый шаг ~60%
    first_dec = int(round(total_stitches * 0.60))
    if first_dec % 2 == 1:
        first_dec += 1
    rest = total_stitches - first_dec

    # последние прямые ряды = 20% глубины горловины
    straight_rows = max(2, int(round(total_rows * 0.20)))
    # конец горловины по глубине минус прямые ряды
    neck_end_by_depth = start_row + total_rows - 1 - straight_rows
    # но не позже, чем начало плеча
    neck_end_by_shoulder = shoulder_start_row - 1
    effective_end = min(neck_end_by_depth, neck_end_by_shoulder)

    rows = allowed_even_rows(start_row, effective_end, rows_total)
    if not rows:
        return []

    actions = []
    # первая крупная убавка в середине
    actions.append((rows[0], f"-{first_dec} п. горловина (середина, разделение на плечи)"))

    if rest <= 0 or len(rows) == 1:
        return actions

    # оставшиеся убавки
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
# Пройма (круглая)
# -----------------------------
def calc_round_armhole(st_chest, st_shoulders, start_row, shoulder_start_row, rows_total, depth_percent=0.05, hold_percent=0.1):
    """Скруглённая пройма: убавки внутрь, потом прямо, потом плавный выход к плечам.
       Пройма всегда заканчивается до начала плеча."""
    if shoulder_start_row <= start_row:
        return []

    end_row = shoulder_start_row - 1
    total_rows = end_row - start_row + 1
    if total_rows <= 0:
        return []

    depth_armhole_st = int(round(st_chest * depth_percent))
    st_mid = st_chest - depth_armhole_st

    rows_smooth = int(total_rows * 0.4)       # нижняя часть
    rows_hold   = int(total_rows * hold_percent)  # прямо
    rows_rest   = total_rows - rows_smooth - rows_hold

    actions = []

    # Этап 1: убавки внутрь (chest → mid)
    delta1 = st_mid - st_chest
    if delta1 < 0:
        actions += sym_decreases(-delta1, start_row, start_row+rows_smooth, rows_total, "пройма")

    # Этап 2: прямо (st_mid)

    # Этап 3: прибавки наружу (mid → плечи)
    delta2 = st_shoulders - st_mid
    if delta2 > 0:
        actions += sym_increases(delta2, start_row+rows_smooth+rows_hold, end_row, rows_total, "пройма")

    return actions

# -----------------------------
# Слияние действий (горловина + плечо)
# -----------------------------
def merge_actions(actions, rows_total):
    """Правила:
       - убавки горловины и плеча не могут быть в одном ряду
       - горловина остаётся на месте (чётные ряды)
       - скос плеча переносим на ряд выше (даже если он нечётный)"""
    merged = defaultdict(list)
    for row, note in actions:
        merged[row].append(note)

    fixed = []
    used_rows = set()

    for row in sorted(merged.keys()):
        notes = merged[row]

        if ("горловина" in " ".join(notes)) and ("скос плеча" in " ".join(notes)):
            shoulder_notes = [n for n in notes if "скос плеча" in n]
            neck_notes     = [n for n in notes if "горловина" in n]

            fixed.append((row, "; ".join(neck_notes)))
            used_rows.add(row)

            new_row = row - 1 if row > 1 else row + 1
            while new_row in used_rows and new_row < rows_total:
                new_row += 1

            for n in shoulder_notes:
                fixed.append((new_row, n))
                used_rows.add(new_row)

        else:
            fixed.append((row, "; ".join(notes)))
            used_rows.add(row)

    return fixed

# -----------------------------
# Таблица + сегменты
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

def make_table_full(actions, rows_total, rows_to_armhole_end, neck_start_row, shoulder_start_row):
    merged = defaultdict(list)
    for row, note in actions:
        merged[row].append(note)

    rows_sorted = sorted(merged.keys())
    table_rows = []
    prev = 1

    if not rows_sorted:
        seg = section_tags(1, rows_to_armhole_end, neck_start_row, shoulder_start_row)
        table_rows.append((f"1-{rows_total}", "Прямо", seg))
    else:
        for r in rows_sorted:
            if r > prev:
                seg = section_tags(prev, rows_to_armhole_end, neck_start_row, shoulder_start_row)
                if prev == r-1:
                    table_rows.append((str(prev), "Прямо", seg))
                else:
                    table_rows.append((f"{prev}-{r-1}", "Прямо", seg))
            table_rows.append((str(r), "; ".join(merged[r]),
                               section_tags(r, rows_to_armhole_end, neck_start_row, shoulder_start_row)))
            prev = r + 1

        if prev <= rows_total:
            seg = section_tags(prev, rows_to_armhole_end, neck_start_row, shoulder_start_row)
            if prev == rows_total:
                table_rows.append((str(prev), "Прямо", seg))
            else:
                table_rows.append((f"{prev}-{rows_total}", "Прямо", seg))

    df = pd.DataFrame(table_rows, columns=["Ряды", "Действия", "Сегмент"])
    st.dataframe(df, use_container_width=True, hide_index=True)

# -----------------------------
# Ввод параметров
# -----------------------------
st.header("Перед / Спинка")

density_st_str   = st.text_input("Плотность: петли в 10 см", placeholder="введите плотность")
density_row_str  = st.text_input("Плотность: ряды в 10 см",  placeholder="введите плотность")

hip_cm_str       = st.text_input("Ширина низа детали (см)", placeholder="введите ширину")
chest_cm_str     = st.text_input("Ширина детали по груди (см)", placeholder="введите ширину")
length_cm_str    = st.text_input("Длина изделия (см)", placeholder="введите длину")

armhole_depth_cm_str   = st.text_input("Длина проймы (см)", placeholder="введите длину")

neck_width_cm_str      = st.text_input("Ширина горловины (см)", placeholder="введите ширину")
neck_depth_cm_str      = st.text_input("Глубина горловины спереди (см)", placeholder="введите глубину")
neck_depth_back_cm_str = st.text_input("Глубина горловины спинки (см)", placeholder="введите глубину")

shoulder_len_cm_str    = st.text_input("Длина плеча (см)", placeholder="введите длину")
shoulder_slope_cm_str  = st.text_input("Скос плеча (см)", placeholder="введите высоту")

# -----------------------------
# Кнопка расчёта
# -----------------------------
if st.button("🔄 Рассчитать"):
    try:
        density_st         = float(density_st_str.replace(",", "."))
        density_row        = float(density_row_str.replace(",", "."))
        hip_cm             = float(hip_cm_str.replace(",", "."))
        chest_cm           = float(chest_cm_str.replace(",", "."))
        length_cm          = float(length_cm_str.replace(",", "."))
        armhole_depth_cm   = float(armhole_depth_cm_str.replace(",", "."))
        neck_width_cm      = float(neck_width_cm_str.replace(",", "."))
        neck_depth_cm      = float(neck_depth_cm_str.replace(",", "."))
        neck_depth_back_cm = float(neck_depth_back_cm_str.replace(",", "."))
        shoulder_len_cm    = float(shoulder_len_cm_str.replace(",", "."))
        shoulder_slope_cm  = float(shoulder_slope_cm_str.replace(",", "."))
    except:
        st.error("⚠️ Пожалуйста, заполните все поля числами (можно с точкой или запятой)")
        st.stop()

    # -----------------------------
    # Пересчёт в петли/ряды
    # -----------------------------
    st_hip     = cm_to_st(hip_cm, density_st)        # низ
    st_chest   = cm_to_st(chest_cm, density_st)      # грудь
    rows_total = cm_to_rows(length_cm, density_row)
    rows_armh  = cm_to_rows(armhole_depth_cm, density_row)

    neck_st    = cm_to_st(neck_width_cm, density_st)
    neck_rows_front  = cm_to_rows(neck_depth_cm, density_row)
    neck_rows_back   = cm_to_rows(neck_depth_back_cm, density_row)

    st_shldr   = cm_to_st(shoulder_len_cm, density_st)
    rows_slope = cm_to_rows(shoulder_slope_cm, density_row)

    st_shoulders = 2 * st_shldr + neck_st   # скрытая ширина по плечам

    rows_to_armhole_end = rows_total - rows_armh
    armhole_start_row   = rows_to_armhole_end + 1
    shoulder_start_row  = rows_total - rows_slope + 1
    armhole_end_row     = shoulder_start_row - 1

    neck_start_row_front = rows_total - neck_rows_front + 1
    neck_start_row_back  = rows_total - neck_rows_back + 1

    # -----------------------------
    # 📊 Сводка
    # -----------------------------
    st.subheader("📊 Сводка")
    st.write(f"- Набрать петель: **{st_hip}**")
    st.write(f"- Всего рядов: **{rows_total}**")

    # ----- ПЕРЕД -----
    st.subheader("📋 Инструкция для переда")
    actions = []

    delta_bottom = st_chest - st_hip
    if delta_bottom > 0:
        actions += sym_increases(delta_bottom, 6, rows_to_armhole_end, rows_total, "бок")
    elif delta_bottom < 0:
        actions += sym_decreases(-delta_bottom, 6, rows_to_armhole_end, rows_total, "бок")

    actions += calc_round_armhole(st_chest, st_shoulders, armhole_start_row, shoulder_start_row, rows_total)
    actions += calc_round_neckline(neck_st, neck_rows_front, neck_start_row_front, rows_total, shoulder_start_row)
    actions += slope_shoulder(st_shldr, shoulder_start_row, rows_total, rows_total)

    make_table_full(actions, rows_total, rows_to_armhole_end, neck_start_row_front, shoulder_start_row)

  # ----- СПИНКА -----
st.subheader("📋 Инструкция для спинки")
actions_back = []

# боковые прибавки/убавки
delta_bottom = st_chest - st_hip
if delta_bottom > 0:
    actions_back += sym_increases(delta_bottom, 6, rows_to_armhole_end, rows_total, "бок")
elif delta_bottom < 0:
    actions_back += sym_decreases(-delta_bottom, 6, rows_to_armhole_end, rows_total, "бок")

# пройма (округлая)
actions_back += calc_round_armhole(st_chest, st_shoulders, armhole_start_row, shoulder_start_row, rows_total)

# горловина (для спинки глубина меньше)
actions_back += calc_round_neckline(neck_st, neck_rows_back, neck_start_row_back, rows_total, shoulder_start_row)

# скос плеча
actions_back += slope_shoulder(st_shldr, shoulder_start_row, rows_total, rows_total)

# ⚡️ объединяем, чтобы горловина и плечо не совпадали
actions_back = merge_actions(actions_back, rows_total)

# итоговая таблица
make_table_full(actions_back, rows_total, rows_to_armhole_end, neck_start_row_back, shoulder_start_row)
