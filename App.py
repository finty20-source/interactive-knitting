import streamlit as st
import numpy as np
import pandas as pd
from collections import defaultdict

st.title("🧶 Интерактивное вязание — полный расчёт модели")

# -----------------------------
# Сессия
# -----------------------------
if "actions" not in st.session_state:
    st.session_state.actions = []
    st.session_state.actions_back = []
    st.session_state.st_hip = 0
    st.session_state.rows_total = 0
    st.session_state.rows_bottom = 0

# -----------------------------
# Конвертеры
# -----------------------------
def cm_to_st(cm, dens_st):
    return int(round((cm/10.0)*dens_st))

def cm_to_rows(cm, dens_row):
    return int(round((cm/10.0)*dens_row))

# -----------------------------
# Вспомогательные
# -----------------------------
def allowed_even_rows(start_row: int, end_row: int, rows_total: int, force_last=False):
    """Чётные ряды: от ≥6 до min(end_row, rows_total-2), либо end_row (если force_last)."""
    if end_row is None: end_row = rows_total
    high = end_row if force_last else min(end_row, rows_total - 2)
    if high < 6: return []
    start = max(6, start_row)
    if start % 2 == 1: start += 1
    if high  % 2 == 1: high  -= 1
    return list(range(start, high + 1, 2)) if start <= high else []

def allowed_all_rows(start_row: int, end_row: int, rows_total: int):
    """Любые ряды (для частичного вязания): от ≥6 до rows_total-2."""
    if end_row is None: end_row = rows_total
    high = min(end_row, rows_total - 2)
    if high < 6: return []
    start = max(6, start_row)
    return list(range(start, high + 1)) if start <= high else []

def split_total_into_steps(total: int, steps: int):
    """Равномерно распределить total по steps шагам: [a,a,...,a+1,...]."""
    if total <= 0 or steps <= 0: return []
    steps = min(steps, total)  # не создаём пустых шагов
    base = total // steps
    rem  = total % steps
    return [base + (1 if i < rem else 0) for i in range(steps)]

def sym_increases(total_add, start_row, end_row, rows_total, label):
    if total_add <= 0: return []
    if total_add % 2 == 1: total_add += 1
    rows = allowed_even_rows(start_row, end_row, rows_total)
    if not rows: return []
    per_side = total_add // 2
    steps = min(len(rows), per_side)
    parts = split_total_into_steps(per_side, steps)
    idxs  = np.linspace(0, len(rows)-1, num=steps, dtype=int)
    chosen= [rows[i] for i in idxs]
    out = []
    for r, v in zip(chosen, parts):
        out.append((r, f"+{v} п. {label} (справа)"))
        out.append((r, f"+{v} п. {label} (слева)"))
    return out

def sym_decreases(total_sub, start_row, end_row, rows_total, label):
    if total_sub <= 0: return []
    if total_sub % 2 == 1: total_sub += 1
    rows = allowed_even_rows(start_row, end_row, rows_total)
    if not rows: return []
    per_side = total_sub // 2
    steps = min(len(rows), per_side)
    parts = split_total_into_steps(per_side, steps)
    idxs  = np.linspace(0, len(rows)-1, num=steps, dtype=int)
    chosen= [rows[i] for i in idxs]
    out = []
    for r, v in zip(chosen, parts):
        out.append((r, f"-{v} п. {label} (справа)"))
        out.append((r, f"-{v} п. {label} (слева)"))
    return out

# -----------------------------
# Пройма (круглая, без пересечений с плечом)
# -----------------------------
def calc_round_armhole(st_chest, st_shoulders, start_row, shoulder_start_row, rows_total,
                       depth_percent=0.05, hold_percent=0.10):
    """Скруглённая пройма заканчивается ДО начала плеча."""
    if shoulder_start_row <= start_row: return []
    end_row = shoulder_start_row - 1
    total_rows = end_row - start_row + 1
    if total_rows <= 0: return []

    depth_armhole_st = int(round(st_chest * depth_percent))
    st_mid = st_chest - depth_armhole_st

    rows_smooth = max(0, int(total_rows * 0.4))
    rows_hold   = max(0, int(total_rows * hold_percent))
    # rows_rest = total_rows - rows_smooth - rows_hold  # (не требуется явно)

    actions = []
    # Этап 1: внутрь
    delta1 = st_mid - st_chest
    if delta1 < 0:
        actions += sym_decreases(-delta1, start_row, start_row+rows_smooth, rows_total, "пройма")
    # Этап 2: прямо (rows_hold)
    # Этап 3: наружу к плечам
    delta2 = st_shoulders - st_mid
    if delta2 > 0:
        actions += sym_increases(delta2, start_row+rows_smooth+rows_hold, end_row, rows_total, "пройма")
    return actions

# -----------------------------
# Горловина + плечо (единый план)
# -----------------------------
def schedule_shoulder_to_zero(need_left, need_right, start_row, end_row, rows_total,
                              forbid_rows=None, allow_overlap_row=None):
    """
    Расписание скоса плеча (чётные ряды) так, чтобы:
    - не попадать в forbid_rows (ряды горловины),
    - разрешить совпадение только в allow_overlap_row (например, центральное закрытие),
    - равномерно разложить по оставшимся чётным рядам,
    - если рядов мало — даём >1 петли в ряд (равномерно).
    """
    actions = []
    even_rows = allowed_even_rows(start_row, end_row, rows_total)
    if not even_rows: return actions

    forb = set(forbid_rows or [])

    # фильтруем чётные, но сохраняем allow_overlap_row (если это чётный и попал в forbid)
    even_rows_clean = []
    for r in even_rows:
        if r in forb and r != allow_overlap_row:
            continue
        even_rows_clean.append(r)

    # левое плечо — чётные как есть
    if even_rows_clean:
        parts_left = split_total_into_steps(need_left, len(even_rows_clean))
        for r, v in zip(even_rows_clean, parts_left):
            if v > 0:
                actions.append((r, f"-{v} п. скос плеча (левое плечо)"))

    # правое — смещение +1, избегая последних рядов и forbid
    right_rows = []
    for r in even_rows_clean:
        rr = r + 1
        if rr <= rows_total - 2 and (rr not in forb or rr == allow_overlap_row):
            right_rows.append(rr)

    if right_rows:
        parts_right = split_total_into_steps(need_right, len(right_rows))
        for r, v in zip(right_rows, parts_right):
            if v > 0:
                actions.append((r, f"-{v} п. скос плеча (правое плечо)"))

    return actions

def plan_neck_and_shoulder(
    neck_st: int,
    neck_rows: int,
    neck_start_row: int,
    st_shldr: int,        # ширина одного плеча в петлях
    rows_slope: int,      # высота скоса плеча в рядах
    rows_total: int,
    straight_percent: float = 0.10,
    allow_first_overlap: bool = True  # можно ли совместить ТОЛЬКО первый ряд горловины и плеча
):
    """
    1) Центральное закрытие горловины = 60% (чётное).
    2) Остаток горловины — частичное вязание (в каждый ряд), но верх straight_percent — прямые ряды.
    3) Скос плеча добирает остаток петель до 0, по чётным рядам; избегает рядов горловины,
       кроме (опционально) самого первого ряда горловины.
    """
    actions = []
    if neck_st <= 0 or neck_rows <= 0 or st_shldr <= 0:
        return actions

    # --- 1. Центральное закрытие (60%), доводим до чётного ---
    first_dec = int(round(neck_st * 0.60))
    if first_dec % 2 == 1: first_dec += 1
    if first_dec > neck_st: first_dec = neck_st if neck_st % 2 == 0 else neck_st - 1
    rest = max(0, neck_st - first_dec)

    central_row = max(6, min(neck_start_row, rows_total - 2))
    actions.append((central_row, f"-{first_dec} п. горловина (центр, разделение на плечи)"))

    # --- 2. Убавки горловины (частичное), верх straight_percent — прямо ---
    straight_rows = max(2, int(round(neck_rows * straight_percent)))
    last_neck_dec_row = min(neck_start_row + neck_rows - 1 - straight_rows, rows_total - 2)

    neck_dec_rows = [central_row]
    left_used = right_used = 0
    if rest > 0 and last_neck_dec_row >= central_row + 1:
        all_rows = allowed_all_rows(central_row + 1, last_neck_dec_row, rows_total)
        if all_rows:
            steps = min(len(all_rows), rest)  # по 1 п. на ряд
            idxs  = np.linspace(0, len(all_rows)-1, num=steps, dtype=int)
            chosen = [all_rows[i] for i in idxs]
            for k, r in enumerate(chosen):
                if k % 2 == 0:
                    actions.append((r, "-1 п. горловина (левое плечо)"))
                    left_used += 1
                else:
                    actions.append((r, "-1 п. горловина (правое плечо)"))
                    right_used += 1
            neck_dec_rows += chosen

    # --- 3. Скос плеча (до нуля), избегая neck_dec_rows ---
    need_left  = max(0, st_shldr - left_used)
    need_right = max(0, st_shldr - right_used)

    shoulder_start_row = rows_total - rows_slope + 1
    if not allow_first_overlap:
        allow_overlap_row = None
    else:
        # разрешаем совпасть только с центральным рядом
        allow_overlap_row = central_row

    actions += schedule_shoulder_to_zero(
        need_left, need_right,
        start_row=shoulder_start_row,
        end_row=rows_total,
        rows_total=rows_total,
        forbid_rows=set(neck_dec_rows),
        allow_overlap_row=allow_overlap_row
    )

    return actions

# -----------------------------
# Слияние действий (страховка)
# -----------------------------
def merge_actions(actions, rows_total):
    """На всякий: горловина+скос не совмещаются, КРОМЕ первого ряда горловины."""
    merged = defaultdict(list)
    for row, note in actions:
        merged[row].append(note)

    fixed = []
    used = set()

    # найдём первый ряд горловины
    neck_rows_sorted = sorted([r for r, notes in merged.items() if any("горловина" in n for n in notes)])
    first_neck_row = neck_rows_sorted[0] if neck_rows_sorted else None

    for row in sorted(merged.keys()):
        notes = merged[row]
        both = ("горловина" in " ".join(notes)) and ("скос плеча" in " ".join(notes))

        if both and (first_neck_row is not None) and row != first_neck_row:
            # оставляем горловину в этом ряду, скос переносим ниже
            neck_notes = [n for n in notes if "горловина" in n]
            sh_notes   = [n for n in notes if "скос плеча" in n]
            fixed.append((row, "; ".join(neck_notes)))
            used.add(row)

            new_r = row + 1
            while (new_r in used) and (new_r < rows_total - 2):
                new_r += 1
            for n in sh_notes:
                fixed.append((new_r, n))
                used.add(new_r)
        else:
            fixed.append((row, "; ".join(notes)))
            used.add(row)

    return sorted(fixed, key=lambda x: int(str(x[0]).split('-')[0]))

# -----------------------------
# Учёт стороны каретки
# -----------------------------
def fix_carriage_side(actions, method=None):
    """
    Стандартные: убавки со стороны каретки.
    Частичное:   убавки с противоположной стороны.
    Нечётные ряды → каретка справа; чётные → каретка слева.
    Переносим ТОЛЬКО действия, где явно указано '(слева)/(справа)'.
    """
    if method is None:
        method = st.session_state.get("method", "Стандартные (со стороны каретки)")

    use_std = method.startswith("Стандартные")
    fixed = []

    for r, note in actions:
        low = note.lower()
        if "слева" in low or "справа" in low:
            if r % 2 == 1:  # нечётный: каретка справа
                correct = "справа" if use_std else "слева"
            else:           # чётный: каретка слева
                correct = "слева" if use_std else "справа"
            if correct not in low:
                # переносим на соседний ряд (но не на rows_total/rows_total-1)
                new_r = r - 1 if r > 2 else r + 1
                if new_r >= rows_total - 1: new_r = r - 1
                fixed.append((new_r, note))
                continue
        fixed.append((r, note))
    return fixed

# -----------------------------
# Сегменты + таблица
# -----------------------------
def section_tags(row, rows_to_armhole_end, neck_start_row, shoulder_start_row):
    tags = []
    if row <= rows_to_armhole_end: tags.append("Низ изделия")
    if rows_to_armhole_end < row < shoulder_start_row: tags.append("Пройма")
    if neck_start_row and row >= neck_start_row: tags.append("Горловина")
    if shoulder_start_row and row >= shoulder_start_row: tags.append("Скос плеча")
    return " + ".join(tags) if tags else "—"

def make_table_full(actions, rows_count, rows_to_armhole_end, neck_start_row, shoulder_start_row, key=None):
    merged = defaultdict(list)
    for row, note in actions:
        if 1 <= row <= rows_count:
            merged[row].append(note)

    rows_sorted = sorted(merged.keys())
    table_rows = []
    prev = 1

    if not rows_sorted:
        seg = section_tags(1, rows_to_armhole_end, neck_start_row, shoulder_start_row)
        table_rows.append((f"1-{rows_count}", "Прямо", seg))
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

        if prev <= rows_count:
            seg = section_tags(prev, rows_to_armhole_end, neck_start_row, shoulder_start_row)
            if prev == rows_count:
                table_rows.append((str(prev), "Прямо", seg))
            else:
                table_rows.append((f"{prev}-{rows_count}", "Прямо", seg))

    df = pd.DataFrame(table_rows, columns=["Ряды", "Действия", "Сегмент"])
    st.dataframe(df, use_container_width=True, hide_index=True)

    if key:
        st.session_state[key] = table_rows

# -----------------------------
# Ввод параметров (с подсказками)
# -----------------------------
st.header("Перед / Спинка")

density_st_str   = st.text_input("Плотность: петли в 10 см", placeholder="например 23")
density_row_str  = st.text_input("Плотность: ряды в 10 см",  placeholder="например 32")

hip_cm_str       = st.text_input("Ширина низа детали (см)", placeholder="например 48")
chest_cm_str     = st.text_input("Ширина детали по груди (см)", placeholder="например 54")
length_cm_str    = st.text_input("Длина изделия (см)", placeholder="например 60")

armhole_depth_cm_str   = st.text_input("Длина проймы (см)", placeholder="например 20")

neck_width_cm_str      = st.text_input("Ширина горловины (см)", placeholder="например 18")
neck_depth_cm_str      = st.text_input("Глубина горловины спереди (см)", placeholder="например 12")
neck_depth_back_cm_str = st.text_input("Глубина горловины спинки (см)", placeholder="например 3")

shoulder_len_cm_str    = st.text_input("Длина плеча (см)", placeholder="например 25")
shoulder_slope_cm_str  = st.text_input("Скос плеча (см)", placeholder="например 6")

method = st.selectbox(
    "Метод убавок",
    ["Стандартные (со стороны каретки)", "Частичное вязание (поворотные ряды)"],
    index=0
)

# -----------------------------
# Кнопка расчёта
# -----------------------------
if st.button("🔄 Рассчитать"):
    # Валидация
    inputs = [
        density_st_str, density_row_str,
        hip_cm_str, chest_cm_str, length_cm_str,
        armhole_depth_cm_str,
        neck_width_cm_str, neck_depth_cm_str, neck_depth_back_cm_str,
        shoulder_len_cm_str, shoulder_slope_cm_str
    ]
    if not all(inputs):
        st.error("⚠️ Заполните все поля"); st.stop()
    try:
        density_st = float(density_st_str.replace(",", "."))
        density_row= float(density_row_str.replace(",", "."))
        hip_cm     = float(hip_cm_str.replace(",", "."))
        chest_cm   = float(chest_cm_str.replace(",", "."))
        length_cm  = float(length_cm_str.replace(",", "."))
        armhole_depth_cm = float(armhole_depth_cm_str.replace(",", "."))
        neck_width_cm    = float(neck_width_cm_str.replace(",", "."))
        neck_depth_cm    = float(neck_depth_cm_str.replace(",", "."))
        neck_depth_back_cm= float(neck_depth_back_cm_str.replace(",", "."))
        shoulder_len_cm  = float(shoulder_len_cm_str.replace(",", "."))
        shoulder_slope_cm= float(shoulder_slope_cm_str.replace(",", "."))
    except:
        st.error("⚠️ Введите числа (можно с запятой)"); st.stop()

    # Пересчёт в петли/ряды
    st_hip     = cm_to_st(hip_cm, density_st)
    st_chest   = cm_to_st(chest_cm, density_st)
    rows_total = cm_to_rows(length_cm, density_row)
    rows_armh  = cm_to_rows(armhole_depth_cm, density_row)

    neck_st    = cm_to_st(neck_width_cm, density_st)
    neck_rows_front  = cm_to_rows(neck_depth_cm, density_row)
    neck_rows_back   = cm_to_rows(neck_depth_back_cm, density_row)

    st_shldr   = cm_to_st(shoulder_len_cm, density_st)
    rows_slope = cm_to_rows(shoulder_slope_cm, density_row)

    st_shoulders = 2 * st_shldr + neck_st
    rows_bottom  = rows_total - rows_armh - rows_slope

    armhole_start_row   = rows_bottom + 1
    shoulder_start_row  = rows_total - rows_slope + 1
    rows_to_armhole_end = rows_bottom

    neck_start_row_front = rows_total - neck_rows_front + 1
    neck_start_row_back  = rows_total - neck_rows_back + 1

    # 📊 Сводка
    st.subheader("📊 Сводка")
    st.write(f"- Набрать петель: **{st_hip}**")
    st.write(f"- Всего рядов: **{rows_total}**")

    # -----------------------------
    # 📋 ПЕРЕД
    # -----------------------------
    st.subheader("📋 Инструкция для переда")
    actions = []

    # Низ
    delta_bottom = (2*st_shldr + neck_st) - st_hip
    if delta_bottom > 0:
        actions += sym_increases(delta_bottom, 6, rows_bottom, rows_total, "бок")
    elif delta_bottom < 0:
        actions += sym_decreases(-delta_bottom, 6, rows_bottom, rows_total, "бок")

    # Пройма
    actions += calc_round_armhole(
        st_chest,
        st_shoulders,
        armhole_start_row,
        shoulder_start_row,
        rows_total
    )

    # Горловина + Плечо
    actions += plan_neck_and_shoulder(
        neck_st=neck_st,
        neck_rows=neck_rows_front,
        neck_start_row=neck_start_row_front,
        st_shldr=st_shldr,
        rows_slope=rows_slope,
        rows_total=rows_total,
        straight_percent=0.10,
        allow_first_overlap=True  # только центральный ряд может совпасть со скосом
    )

    actions = merge_actions(actions, rows_total)
    actions = fix_carriage_side(actions, method)
    make_table_full(actions, rows_total, rows_bottom, neck_start_row_front, shoulder_start_row, key="table_front")

    # -----------------------------
    # 📋 СПИНКА
    # -----------------------------
    st.subheader("📋 Инструкция для спинки")
    actions_back = []

    # Низ
    delta_bottom = (2*st_shldr + neck_st) - st_hip
    if delta_bottom > 0:
        actions_back += sym_increases(delta_bottom, 6, rows_bottom, rows_total, "бок")
    elif delta_bottom < 0:
        actions_back += sym_decreases(-delta_bottom, 6, rows_bottom, rows_total, "бок")

    # Пройма
    actions_back += calc_round_armhole(
        st_chest,
        st_shoulders,
        armhole_start_row,
        shoulder_start_row,
        rows_total
    )

    # Горловина + Плечо (спинка)
    actions_back += plan_neck_and_shoulder(
        neck_st=neck_st,
        neck_rows=neck_rows_back,
        neck_start_row=neck_start_row_back,
        st_shldr=st_shldr,
        rows_slope=rows_slope,
        rows_total=rows_total,
        straight_percent=0.10,
        allow_first_overlap=True
    )

    actions_back = merge_actions(actions_back, rows_total)
    actions_back = fix_carriage_side(actions_back, method)
    make_table_full(actions_back, rows_total, rows_bottom, neck_start_row_back, shoulder_start_row, key="table_back")

    # сохранить для PDF
    st.session_state.actions = actions
    st.session_state.actions_back = actions_back
    st.session_state.st_hip = st_hip
    st.session_state.rows_total = rows_total
    st.session_state.rows_bottom = rows_bottom
