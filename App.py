import streamlit as st
import numpy as np
import pandas as pd
from collections import defaultdict

st.title("🧶 Интерактивное вязание — полный расчёт (перед/спинка/рукав)")

# -----------------------------
# Вспомогательные конвертеры
# -----------------------------
def cm_to_st(cm, dens_st):   return int(round((cm/10.0)*dens_st))
def cm_to_rows(cm, dens_row):return int(round((cm/10.0)*dens_row))

# -----------------------------
# Рядовые правила
# -----------------------------
def allowed_even_rows(start_row: int, end_row: int, rows_total: int):
    """
    Разрешённые чётные ряды:
    - не раньше 6,
    - не позже end_row,
    - не позже rows_total-2 (никаких манипуляций в последние 2 ряда).
    """
    if end_row is None:
        end_row = rows_total
    high = min(end_row, rows_total - 2)
    if high < 6:
        return []
    start = max(6, start_row)
    # привести к чётным
    if start % 2 == 1: start += 1
    if high  % 2 == 1: high  -= 1
    return list(range(start, high + 1, 2)) if start <= high else []

def split_total_into_steps(total: int, steps: int):
    """Равномерно разбить total на steps положительных целых частей (сумма == total)."""
    if total <= 0 or steps <= 0:
        return []
    steps = min(steps, total)  # чтобы не было нулевых шагов
    base = total // steps
    rem  = total % steps
    return [base + (1 if i < rem else 0) for i in range(steps)]

# -----------------------------
# Генераторы действий
# -----------------------------
def sym_increases(total_add: int, start_row: int, end_row: int, rows_total: int, label: str):
    """
    Симметричные прибавки: в одном ряду одинаково слева и справа.
    total_add — суммарно по полотну (всего петель).
    """
    if total_add <= 0:
        return []
    # симметрия => общее число прибавляемых петель по полотну должно быть чётным
    if total_add % 2 == 1:
        total_add += 1
    rows = allowed_even_rows(start_row, end_row, rows_total)
    if not rows:
        return []
    per_side = total_add // 2
    steps = min(len(rows), per_side)  # минимум 1 петля на шаг с каждой стороны
    parts = split_total_into_steps(per_side, steps)
    # равномерно выбираем ряды по высоте
    idxs = np.linspace(0, len(rows)-1, num=steps, dtype=int)
    chosen = [rows[i] for i in idxs]
    return [(r, f"+{v} п. {label} слева и +{v} п. {label} справа") for r, v in zip(chosen, parts)]

def slope_shoulder(total_stitches: int, start_row: int, end_row: int, rows_total: int):
    """
    Скос плеча: равномерно по ВСЕМ доступным чётным рядам; если остаток — кладём его в начало.
    Действие формулируется как закрытие на КАЖДОМ плече.
    """
    if total_stitches <= 0:
        return []
    rows = allowed_even_rows(start_row, end_row, rows_total)
    if not rows:
        return []
    steps = len(rows)
    base = total_stitches // steps
    rem  = total_stitches % steps
    actions = []
    for i, r in enumerate(rows):
        dec = base + (1 if i < rem else 0)  # "лишние" петли в начале
        actions.append((r, f"-{dec} п. плечо (каждое плечо)"))
    return actions

def calc_round_neckline(total_stitches: int, total_rows: int, start_row: int, rows_total: int, straight_percent: float = 0.05):
    """
    КРУГЛАЯ горловина (без выбора):
    - 1-й шаг: 60% петель горловины, в первом доступном чётном ряду (деление на плечи).
    - Верхние straight_percent*глубины (но ≥2 ряда) — ВЯЗАТЬ ПРЯМО (без убавок).
    - Остаток петель равномерно распределяется по чётным рядам между первым шагом и верхней «прямой» зоной.
    - Последние 2 убавочных ряда ≤ 1 петли; "лишнее" равномерно раздаётся вниз (первые шаги после 60%).
    - Манипуляции не попадают в последние 2 ряда полотна.
    """
    if total_stitches <= 0 or total_rows <= 0:
        return []

    # 1) 60% сразу
    first_dec = int(round(total_stitches * 0.60))
    rest = total_stitches - first_dec

    # 2) Верхняя прямая часть: 5% глубины (но минимум 2 ряда)
    straight_rows = max(2, int(round(total_rows * straight_percent)))
    neck_end_by_depth = start_row + total_rows - 1 - straight_rows
    effective_end = min(neck_end_by_depth, rows_total - 2)

    # 3) Доступные ряды
    rows = allowed_even_rows(start_row, effective_end, rows_total)
    if not rows:
        return []
    actions = []

    # Первый шаг (60%) в самом нижнем доступном чётном ряду
    actions.append((rows[0], f"-{first_dec} п. горловина (середина, разделение на плечи)"))

    # Если нечего распределять
    if rest <= 0 or len(rows) == 1:
        return actions

    # 4) Остаток равномерно по оставшимся рядам (между 60%-шагом и верхней «прямой» зоной)
    rest_rows = rows[1:]
    steps = min(len(rest_rows), rest)  # чтобы каждая часть ≥1
    # Чтобы было куда "сгладить" последние два, иногда выгодно steps >= 3
    if steps == 2 and rest > 2 and len(rest_rows) >= 3:
        steps = 3

    idxs   = np.linspace(0, len(rest_rows)-1, num=steps, dtype=int)
    chosen = [rest_rows[i] for i in idxs]
    parts  = split_total_into_steps(rest, steps)

    # 5) Сглаживание: последние две убавочные строки ≤ 1 петли
    if steps >= 2:
        over = 0
        for i in [steps-2, steps-1]:
            if parts[i] > 1:
                over += parts[i] - 1
                parts[i] = 1
        # Раздаём over равномерно вниз (к шагам 0..steps-3)
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
# Сегменты/склейка
# -----------------------------
def section_tags(row, rows_to_armhole_end, armhole_start_row, armhole_end_row, neck_start_row, shoulder_start_row):
    tags = []
    if row <= rows_to_armhole_end:
        tags.append("Низ изделия")
    if armhole_start_row is not None and armhole_end_row is not None:
        if armhole_start_row <= row <= armhole_end_row:
            tags.append("Пройма")
    if neck_start_row is not None and row >= neck_start_row:
        tags.append("Горловина")
    if shoulder_start_row is not None and row >= shoulder_start_row:
        tags.append("Скос плеча")
    return ", ".join(tags) if tags else "—"

def make_table(actions, rows_total,
               rows_to_armhole_end=None,
               armhole_start_row=None, armhole_end_row=None,
               neck_start_row=None, shoulder_start_row=None):
    """
    Объединяем все действия, совпавшие по ряду, в одну строку.
    """
    merged = defaultdict(list)
    for row, note in actions:
        merged[row].append(note)

    if not merged:
        st.info("Нет действий для выбранных параметров.")
        return

    rows_sorted = sorted(merged.keys())
    data = {
        "Ряд": rows_sorted,
        "Действия": ["; ".join(merged[r]) for r in rows_sorted],
        "Сегмент": [
            section_tags(r, rows_to_armhole_end, armhole_start_row, armhole_end_row, neck_start_row, shoulder_start_row)
            for r in rows_sorted
        ] if rows_to_armhole_end is not None else
        ["Рукав" if r < rows_total else "Окат (прямой)" for r in rows_sorted]
    }
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)

# -----------------------------
# Вкладки UI
# -----------------------------
tab1, tab2, tab3 = st.tabs(["Перед", "Спинка", "Рукав"])

# -----------------------------
# ПЕРЕД
# -----------------------------
with tab1:
    st.header("Перед")
    density_st  = st.number_input("Плотность: петли в 10 см", 1, 999, 23, key="dst1")
    density_row = st.number_input("Плотность: ряды в 10 см",  1, 999, 40, key="drw1")

    chest_cm   = st.number_input("Обхват груди (см)", 50, 200, 90, key="ch1")
    hip_cm     = st.number_input("Обхват низа (см)",  50, 200, 80, key="hip1")
    length_cm  = st.number_input("Длина изделия (см)", 30, 120, 55, key="len1")

    armhole_depth_cm     = st.number_input("Глубина проймы (см)", 10, 40, 23, key="arm1")
    shoulders_width_cm   = st.number_input("Ширина изделия по плечам (см)", 60, 160, 100, key="shw1")

    neck_width_cm = st.number_input("Ширина горловины (см)", 5, 40, 18, key="nw1")
    neck_depth_cm = st.number_input("Глубина горловины (см)", 2, 40, 6, key="nd1")

    shoulder_len_cm   = st.number_input("Длина одного плеча (см)", 5, 30, 12, key="sl1")
    shoulder_slope_cm = st.number_input("Высота скоса плеча (см)", 1, 20, 4, key="ss1")

 if st.button("🔄 Рассчитать перед"):
    # Пересчёты
    st_chest     = cm_to_st(chest_cm, density_st)
    st_hip       = cm_to_st(hip_cm,   density_st)
    st_shoulders = cm_to_st(shoulders_width_cm, density_st)

    rows_total   = cm_to_rows(length_cm, density_row)
    rows_armhole = cm_to_rows(armhole_depth_cm, density_row)

    neck_st      = cm_to_st(neck_width_cm, density_st)
    neck_rows    = cm_to_rows(neck_depth_cm, density_row)

    st_shoulder  = cm_to_st(shoulder_len_cm, density_st)
    rows_sh_slope= cm_to_rows(shoulder_slope_cm, density_row)

    # Ключевые границы по рядам
    rows_to_armhole_end = rows_total - rows_armhole
    shoulder_start_row  = rows_total - rows_sh_slope + 1
    neck_start_row      = rows_total - neck_rows + 1
    armhole_start_row   = rows_to_armhole_end + 1
    armhole_end_row     = rows_total   # ⚡️ теперь до самого верха, не обрезаем горловиной

    actions = []

    # 1) Низ → грудь (оверсайз: растём кверху)
    delta_bottom = st_chest - st_hip
    if delta_bottom > 0:
        actions += sym_increases(delta_bottom, 6, rows_to_armhole_end, rows_total, "бок")

    # 2) Пройма (от груди до ширины по плечам)
    delta_armhole = st_shoulders - st_chest
    if delta_armhole > 0 and armhole_start_row <= armhole_end_row:
        actions += sym_increases(delta_armhole, armhole_start_row, armhole_end_row, rows_total, "пройма")

    # 3) Горловина
    actions += calc_round_neckline(neck_st, neck_rows, neck_start_row, rows_total, straight_percent=0.05)

    # 4) Скос плеча
    actions += slope_shoulder(st_shoulder, shoulder_start_row, rows_total, rows_total)

    st.subheader("📋 Единый план — ПЕРЕД")
    make_table(actions, rows_total,
               rows_to_armhole_end, armhole_start_row, armhole_end_row,
               neck_start_row, shoulder_start_row)

        # 1) Низ → грудь (оверсайз: растём кверху)
        delta_bottom = st_chest - st_hip
        actions = []
        if delta_bottom > 0:
            actions += sym_increases(delta_bottom, 6, rows_to_armhole_end, rows_total, "бок")

        # 2) Пройма (добавка от груди до ширины по плечам)
        delta_armhole = st_shoulders - st_chest
        if delta_armhole > 0 and armhole_start_row <= armhole_end_row:
            actions += sym_increases(delta_armhole, armhole_start_row, armhole_end_row, rows_total, "пройма")

        # 3) Горловина (60% + гибкое распределение, верх 5% — прямо)
        actions += calc_round_neckline(neck_st, neck_rows, neck_start_row, rows_total, straight_percent=0.05)

        # 4) Скос плеча (плавно)
        actions += slope_shoulder(st_shoulder, shoulder_start_row, rows_total, rows_total)

        st.subheader("📋 Единый план — ПЕРЕД")
        make_table(actions, rows_total,
                   rows_to_armhole_end, armhole_start_row, armhole_end_row,
                   neck_start_row, shoulder_start_row)

# -----------------------------
# СПИНКА
# -----------------------------
with tab2:
    st.header("Спинка")
    density_st  = st.number_input("Плотность: петли в 10 см", 1, 999, 23, key="dst2")
    density_row = st.number_input("Плотность: ряды в 10 см",  1, 999, 40, key="drw2")

    chest_cm   = st.number_input("Обхват груди (см)", 50, 200, 90, key="ch2")
    hip_cm     = st.number_input("Обхват низа (см)",  50, 200, 80, key="hip2")
    length_cm  = st.number_input("Длина изделия (см)", 30, 120, 55, key="len2")

    armhole_depth_cm     = st.number_input("Глубина проймы (см)", 10, 40, 23, key="arm2")
    shoulders_width_cm   = st.number_input("Ширина изделия по плечам (см)", 60, 160, 100, key="shw2")

    neck_width_cm = st.number_input("Ширина горловины (см)", 5, 40, 18, key="nw2")
    neck_depth_cm = st.number_input("Глубина горловины (см)", 1, 40, 3, key="nd2")  # спинка — обычно меньше

    shoulder_len_cm   = st.number_input("Длина одного плеча (см)", 5, 30, 12, key="sl2")
    shoulder_slope_cm = st.number_input("Высота скоса плеча (см)", 1, 20, 4, key="ss2")

    if st.button("🔄 Рассчитать спинку"):
        st_chest     = cm_to_st(chest_cm, density_st)
        st_hip       = cm_to_st(hip_cm,   density_st)
        st_shoulders = cm_to_st(shoulders_width_cm, density_st)

        rows_total   = cm_to_rows(length_cm, density_row)
        rows_armhole = cm_to_rows(armhole_depth_cm, density_row)

        neck_st      = cm_to_st(neck_width_cm, density_st)
        neck_rows    = cm_to_rows(neck_depth_cm, density_row)

        st_shoulder  = cm_to_st(shoulder_len_cm, density_st)
        rows_sh_slope= cm_to_rows(shoulder_slope_cm, density_row)

        rows_to_armhole_end = rows_total - rows_armhole
        shoulder_start_row  = rows_total - rows_sh_slope + 1
        neck_start_row      = rows_total - neck_rows + 1
        armhole_start_row   = rows_to_armhole_end + 1
        armhole_end_row     = min(rows_total, shoulder_start_row - 1)

        actions = []
        delta_bottom = st_chest - st_hip
        if delta_bottom > 0:
            actions += sym_increases(delta_bottom, 6, rows_to_armhole_end, rows_total, "бок")

        delta_armhole = st_shoulders - st_chest
        if delta_armhole > 0 and armhole_start_row <= armhole_end_row:
            actions += sym_increases(delta_armhole, armhole_start_row, armhole_end_row, rows_total, "пройма")

        actions += calc_round_neckline(neck_st, neck_rows, neck_start_row, rows_total, straight_percent=0.05)
        actions += slope_shoulder(st_shoulder, shoulder_start_row, rows_total, rows_total)

        st.subheader("📋 Единый план — СПИНКА")
        make_table(actions, rows_total,
                   rows_to_armhole_end, armhole_start_row, armhole_end_row,
                   neck_start_row, shoulder_start_row)

# -----------------------------
# РУКАВ (оверсайз, прямой окат)
# -----------------------------
with tab3:
    st.header("Рукав (оверсайз, прямой окат)")
    density_st  = st.number_input("Плотность: петли в 10 см", 1, 999, 23, key="dst3")
    density_row = st.number_input("Плотность: ряды в 10 см",  1, 999, 40, key="drw3")

    length_cm = st.number_input("Длина рукава (см)", 20, 100, 60, key="rl")
    wrist_cm  = st.number_input("Ширина манжеты (см)", 10, 40, 18, key="rw")
    top_cm    = st.number_input("Ширина рукава вверху (см)", 20, 80, 36, key="rt")

    if st.button("🔄 Рассчитать рукав"):
        st_wrist = cm_to_st(wrist_cm, density_st)
        st_top   = cm_to_st(top_cm,   density_st)
        rows_total = cm_to_rows(length_cm, density_row)

        delta = st_top - st_wrist
        actions = []
        if delta > 0:
            # прибавки от манжеты до верха (кроме последних 2 рядов)
            actions += sym_increases(delta, 6, rows_total - 1, rows_total, "рукав")

        # финальное закрытие (прямой окат) — отдельной строкой на последнем ряду
        actions.append((rows_total, "Закрыть все петли (прямой окат)"))

        st.subheader("📋 Единый план — РУКАВ")
        make_table(actions, rows_total)
