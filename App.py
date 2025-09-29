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
    return int(round((cm/10.0)*dens_st))

def cm_to_rows(cm, dens_row):
    return int(round((cm/10.0)*dens_row))
# -----------------------------
# Рядовые правила (с доработкой)
# -----------------------------
def allowed_even_rows(start_row: int, end_row: int, rows_total: int, force_last=False):
    """Разрешённые чётные ряды: ≥6 и ≤ end_row.
       По умолчанию обрезает по rows_total-2,
       но если force_last=True — идёт до самого конца (end_row)."""
    if end_row is None:
        end_row = rows_total
    high = end_row if force_last else min(end_row, rows_total - 2)
    if high < 6:
        return []
    start = max(6, start_row)
    if start % 2 == 1: start += 1
    if high % 2 == 1: high -= 1
    return list(range(start, high + 1, 2)) if start <= high else []
def split_total_into_steps(total: int, steps: int):
    if total <= 0 or steps <= 0:
        return []
    steps = min(steps, total)
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
# Горловина (круглая, с прямыми рядами сверху)
# -----------------------------
def calc_round_neckline(total_stitches, total_rows, start_row, rows_total, straight_spec=0.20):
    if total_stitches <= 0 or total_rows <= 0:
        return []

    # первый шаг — 60%
    first_dec = int(round(total_stitches * 0.60))
    if first_dec % 2 == 1:
        first_dec += 1
    rest = total_stitches - first_dec

    # последние straight_spec процентов глубины — прямые
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
    idxs   = np.linspace(0, len(rest_rows)-1, num=steps, dtype=int)
    chosen = [rest_rows[i] for i in idxs]
    parts  = split_total_into_steps(rest, steps)

    for r, v in zip(chosen, parts):
        actions.append((r, f"-{v} п. горловина"))

    return actions

# -----------------------------
# Скос плеча
# -----------------------------
def slope_shoulders(total_stitches, start_row, end_row, rows_total):
    """
    Левое и правое плечо: убавки синхронно в ЧЁТНЫХ рядах.
    """
    if total_stitches <= 0:
        return [], []

    rows = allowed_even_rows(start_row, end_row, rows_total)
    if not rows:
        return [], []

    steps = len(rows)
    base = total_stitches // steps
    rem  = total_stitches % steps

    left_actions, right_actions = [], []

    for i, r in enumerate(rows):
        dec = base + (1 if i < rem else 0)
        left_actions.append((r, f"-{dec} п. скос плеча (левое плечо)"))
        right_actions.append((r, f"-{dec} п. скос плеча (правое плечо)"))

    return left_actions, right_actions

# -----------------------------
# Горловина + скос плеча (с разделением на левое/правое плечо)
# -----------------------------
def plan_neck_and_shoulders_split(
    neck_st, neck_rows, neck_start_row,
    st_shoulders, shoulder_start_row, rows_total,
    straight_percent=0.20
):
    actions = []

    # Горловина (с сохранением последних 20% рядов прямыми)
    actions += calc_round_neckline(
        total_stitches=neck_st,
        total_rows=neck_rows,
        start_row=neck_start_row,
        rows_total=rows_total,
        straight_spec=straight_percent
    )

    # Скос плеча (отдельно для левого и правого)
    actions_left, actions_right = slope_shoulders(
        total_stitches=st_shoulders // 2,
        start_row=shoulder_start_row,
        end_row=rows_total,
        rows_total=rows_total
    )
    actions += actions_left + actions_right

    return actions

# -----------------------------
# Слияние действий (горловина + плечо)
# -----------------------------
def merge_actions(actions, rows_total):
    """
    Правила:
    - горловина и скос плеча могут совпасть только в САМОМ ПЕРВОМ ряду горловины,
      в остальных случаях мы их разносим.
    - горловина остаётся в своём ряду.
    - скос переносим на +1 ряд (если занят — ищем дальше).
    """
    merged = defaultdict(list)
    for row, note in actions:
        merged[row].append(note)

    fixed = []
    used_rows = set()
    first_neck_row = None  # запомним первый ряд горловины

    # сначала найдём первый ряд горловины
    for row in sorted(merged.keys()):
        if any("горловина" in n for n in merged[row]):
            first_neck_row = row
            break

    for row in sorted(merged.keys()):
        notes = merged[row]

        if ("горловина" in " ".join(notes)) and ("скос плеча" in " ".join(notes)):
            # если это первый ряд горловины → оставляем вместе
            if row == first_neck_row:
                fixed.append((row, "; ".join(notes)))
                used_rows.add(row)
            else:
                # разделяем: горловина в своём ряду, скос переносим выше
                shoulder_notes = [n for n in notes if "скос плеча" in n]
                neck_notes     = [n for n in notes if "горловина" in n]

                fixed.append((row, "; ".join(neck_notes)))
                used_rows.add(row)

                new_row = row + 1
                while new_row in used_rows and new_row < rows_total:
                    new_row += 1

                for n in shoulder_notes:
                    fixed.append((new_row, n))
                    used_rows.add(new_row)
        else:
            fixed.append((row, "; ".join(notes)))
            used_rows.add(row)

    return sorted(fixed, key=lambda x: int(str(x[0]).split('-')[0]))

# -----------------------------
# Учёт стороны каретки
# -----------------------------
def fix_carriage_side(actions, method=None):
    """
    Стандартные: убавки со стороны каретки.
    Частичное:   убавки с противоположной стороны.
    Нечётные ряды → каретка справа; чётные → каретка слева.
    """
    if method is None:
        method = st.session_state.get("method", "Стандартные (со стороны каретки)")

    use_std = method.startswith("Стандартные")
    fixed = []

    for r, note in actions:
        note_lower = note.lower()

        # где "правильно" делать убавку в этом ряду
        if r % 2 == 1:  # нечётный: каретка справа
            correct_side = "справа" if use_std else "слева"
        else:           # чётный: каретка слева
            correct_side = "слева" if use_std else "справа"

        # переносим только те действия, где сторона указана явным словом
        if (("справа" in note_lower) or ("слева" in note_lower)) and (correct_side not in note_lower):
            new_r = r - 1 if r > 1 else r + 1
            fixed.append((new_r, note))
        else:
            fixed.append((r, note))

    return fixed

# -----------------------------
# Сегменты по рядам
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
    
# -----------------------------
# Таблица + сегменты
# -----------------------------
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

        last_action_row = max(rows_sorted)
        if prev <= last_action_row:
            seg = section_tags(prev, rows_to_armhole_end, neck_start_row, shoulder_start_row)
            if prev == last_action_row:
                table_rows.append((str(prev), "Прямо", seg))
            else:
                table_rows.append((f"{prev}-{last_action_row}", "Прямо", seg))

    df = pd.DataFrame(table_rows, columns=["Ряды", "Действия", "Сегмент"])
    st.dataframe(df, use_container_width=True, hide_index=True)

    if key:
        st.session_state[key] = table_rows

def parse_inputs():
    return (
        float(density_st_str.replace(",", ".")),
        float(density_row_str.replace(",", ".")),
        float(hip_cm_str.replace(",", ".")),
        float(chest_cm_str.replace(",", ".")),
        float(length_cm_str.replace(",", ".")),
        float(armhole_depth_cm_str.replace(",", ".")),
        float(neck_width_cm_str.replace(",", ".")),
        float(neck_depth_cm_str.replace(",", ".")),
        float(neck_depth_back_cm_str.replace(",", ".")),
        float(shoulder_len_cm_str.replace(",", ".")),
        float(shoulder_slope_cm_str.replace(",", "."))
    )

# -----------------------------
# Таблица переда с разделением на плечи
# -----------------------------
def make_table_front_split(actions, rows_count, rows_to_armhole_end, neck_start_row, shoulder_start_row, key=None):
    """
    Делит инструкцию переда на:
    1) До ряда разделения на плечи,
    2) ЛЕВОЕ ПЛЕЧО,
    3) ПРАВОЕ ПЛЕЧО (с возвратом к ряду разделения).
    Если разделение не найдено — тихо падаем обратно на обычную make_table_full.
    """
    # Собираем ряды -> список действий
    merged = defaultdict(list)
    for row, note in actions:
        if isinstance(row, int) and 1 <= row <= rows_count:
            merged[row].append(note)

    if not merged:
        # Нечего отрисовывать
        make_table_full(actions, rows_count, rows_to_armhole_end, neck_start_row, shoulder_start_row, key=key)
        return

    rows_sorted = sorted(merged.keys())

    # Ищем ряд разделения (центральное закрытие горловины)
    split_row = None
    for r in rows_sorted:
        joined = " ; ".join(merged[r]).lower()
        if "разделение на плечи" in joined:
            split_row = r
            break

    if split_row is None:
        # Нет разделения — используем обычный рендер
        make_table_full(actions, rows_count, rows_to_armhole_end, neck_start_row, shoulder_start_row, key=key)
        return

    def section_tags(row):
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

    def push_plain_range(table_rows, a, b):
        if a > b:
            return
        if a == b:
            table_rows.append((str(a), "Прямо", section_tags(a)))
        else:
            table_rows.append((f"{a}-{b}", "Прямо", section_tags(a)))

    # ---------- 1) ДО РАЗДЕЛЕНИЯ ----------
    table_rows = []
    prev = 1
    for r in [x for x in rows_sorted if x < split_row]:
        if r > prev:
            push_plain_range(table_rows, prev, r - 1)
        table_rows.append((str(r), "; ".join(merged[r]), section_tags(r)))
        prev = r + 1
    if prev <= split_row - 1:
        push_plain_range(table_rows, prev, split_row - 1)

    # Ряд разделения показываем всегда
    table_rows.append((str(split_row), "; ".join(merged[split_row]), section_tags(split_row)))

    # Вспомогательные фильтры
    def left_notes(notes):
        out = []
        for n in notes:
            ln = n.lower()
            if "левое плечо" in ln or "каждое плечо" in ln:
                out.append(n)
        return out

    def right_notes(notes, include_split=False):
        out = []
        for n in notes:
            ln = n.lower()
            if "правое плечо" in ln or "каждое плечо" in ln:
                out.append(n)
            # В правом блоке в первом ряду дублируем центральное закрытие для наглядности
            if include_split and "разделение на плечи" in ln and n not in out:
                out.append(n)
        return out

    # ---------- 2) ЛЕВОЕ ПЛЕЧО ----------
    table_rows.append(("— ЛЕВОЕ ПЛЕЧО —", "", ""))

    left_rows = []
    for r in [x for x in rows_sorted if x > split_row]:
        filt = left_notes(merged[r])
        if filt:
            left_rows.append((r, filt))

    prev = split_row + 1
    for r, notes in left_rows:
        if r > prev:
            push_plain_range(table_rows, prev, r - 1)
        table_rows.append((str(r), "; ".join(notes), section_tags(r)))
        prev = r + 1
    if prev <= rows_count:
        push_plain_range(table_rows, prev, rows_count)

    # ---------- 3) ПРАВОЕ ПЛЕЧО ----------
    table_rows.append((f"— ПРАВОЕ ПЛЕЧО — (вернитесь к ряду {split_row})", "", ""))

    right_rows = []
    # включаем split_row, чтобы показать центральное закрытие и возможный скос, если он совпал
    candidate_rows = [split_row] + [x for x in rows_sorted if x > split_row]
    for r in candidate_rows:
        filt = right_notes(merged[r], include_split=(r == split_row))
        if filt:
            right_rows.append((r, filt))

    prev = split_row
    for r, notes in right_rows:
        if r > prev:
            push_plain_range(table_rows, prev, r - 1)
        table_rows.append((str(r), "; ".join(notes), section_tags(r)))
        prev = r + 1
    if prev <= rows_count:
        push_plain_range(table_rows, prev, rows_count)

    # Рендерим
    df = pd.DataFrame(table_rows, columns=["Ряды", "Действия", "Сегмент"])
    st.dataframe(df, use_container_width=True, hide_index=True)

    if key:
        st.session_state[key] = table_rows
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
# Выбор метода убавок
# -----------------------------
method = st.selectbox(
    "Метод убавок",
    ["Стандартные (со стороны каретки)", "Частичное вязание (поворотные ряды)"],
    index=0
)

# -----------------------------
# Кнопка расчёта
# -----------------------------
if st.button("🔄 Рассчитать"):

    inputs = [
        density_st_str, density_row_str,
        hip_cm_str, chest_cm_str, length_cm_str,
        armhole_depth_cm_str,
        neck_width_cm_str, neck_depth_cm_str, neck_depth_back_cm_str,
        shoulder_len_cm_str, shoulder_slope_cm_str
    ]

    if not all(inputs):
        st.error("⚠️ Заполните все поля перед расчётом")
        st.stop()

    try:
        (density_st, density_row, hip_cm, chest_cm, length_cm,
         armhole_depth_cm, neck_width_cm, neck_depth_cm, neck_depth_back_cm,
         shoulder_len_cm, shoulder_slope_cm) = parse_inputs()
    except:
        st.error("⚠️ Введите только числа (можно с точкой или запятой)")
        st.stop()

    # -----------------------------
    # Пересчёт в петли/ряды
    # -----------------------------
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
    armhole_end_row     = shoulder_start_row - 1
    rows_to_armhole_end = rows_bottom
    last_action_row     = rows_total - 1  # последний ряд = убавка, не закрытие

    neck_start_row_front = rows_total - neck_rows_front + 1
    neck_start_row_back  = rows_total - neck_rows_back + 1

    # -----------------------------
    # 📊 Сводка
    # -----------------------------
    st.subheader("📊 Сводка")
    st.write(f"- Набрать петель: **{st_hip}**")
    st.write(f"- Всего рядов: **{rows_total}**")

    # -----------------------------
    # 📋 Перед
    # -----------------------------
    st.subheader("📋 Инструкция для переда")
    actions = []

    # 1. Низ (разница между шириной низа и грудью)
    delta_bottom = st_chest - st_hip
    if delta_bottom > 0:
        actions += sym_increases(delta_bottom, 6, rows_bottom, rows_total, "бок")
    elif delta_bottom < 0:
        actions += sym_decreases(-delta_bottom, 6, rows_bottom, rows_total, "бок")

    # 2. Пройма
    actions += calc_round_armhole(
        st_chest, st_shoulders, armhole_start_row, shoulder_start_row, rows_total
    )

    # 3. Горловина + скос плеча (единая логика с разделением плеч)
    actions += plan_neck_and_shoulders_split(
        neck_st=neck_st,
        neck_rows=neck_rows_front,
        neck_start_row=neck_start_row_front,
        st_shoulders=2 * st_shldr,   # ширина обоих плеч!
        shoulder_start_row=shoulder_start_row,
        rows_total=rows_total,
        straight_percent=0.20        # последние 20% рядов горловины прямо
    )

    # 4. Слияние и коррекция
    actions = merge_actions(actions, rows_total)
    actions = fix_carriage_side(actions, method)

    # 5. Таблица
    make_table_front_split(actions, rows_total, rows_bottom, neck_start_row_front, shoulder_start_row, key="table_front")
    # -----------------------------
    # 📋 Спинка
    # -----------------------------
    st.subheader("📋 Инструкция для спинки")
    actions_back = []

    # 1. Низ (разница между шириной низа и грудью)
    delta_bottom = st_chest - st_hip
    if delta_bottom > 0:
        actions_back += sym_increases(delta_bottom, 6, rows_bottom, rows_total, "бок")
    elif delta_bottom < 0:
        actions_back += sym_decreases(-delta_bottom, 6, rows_bottom, rows_total, "бок")

    # 2. Пройма
    delta_armh = st_shoulders - st_chest
    if delta_armh > 0:
        actions_back += sym_increases(delta_armh, armhole_start_row, armhole_end_row, rows_total, "пройма")
    elif delta_armh < 0:
        actions_back += sym_decreases(-delta_armh, armhole_start_row, armhole_end_row, rows_total, "пройма")

    # 3. Горловина + плечи (вместе)
    actions_back += plan_neck_and_shoulders_split(
        neck_st=neck_st,
        neck_rows=neck_rows_back,
        neck_start_row=neck_start_row_back,
        st_shoulders=2 * st_shldr,   # <-- вот так
        shoulder_start_row=shoulder_start_row,
        rows_total=rows_total
   )

    # 4. Слияние и коррекция
    actions_back = merge_actions(actions_back, rows_total)
    actions_back = fix_carriage_side(actions_back, method)

    # 5. Таблица
    make_table_full(actions_back, rows_total, rows_bottom, neck_start_row_back, shoulder_start_row, key="table_back")
    # -----------------------------
    # сохраняем для PDF
    # -----------------------------
    st.session_state.actions = actions
    st.session_state.actions_back = actions_back
    st.session_state.st_hip = st_hip
    st.session_state.rows_total = rows_total
    st.session_state.rows_bottom = rows_bottom

    # пересчёт в петли/ряды
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
    armhole_end_row     = shoulder_start_row - 1

    # последний ряд — закрытие; манипуляции до rows_total-1
    last_action_row = rows_total - 1

    # старт горловин относительно last_action_row (чтобы не «раньше времени»)
    neck_start_row_front = last_action_row - neck_rows_front + 1
    neck_start_row_back  = last_action_row - neck_rows_back + 1

    # сохранить для PDF
    st.session_state.actions = actions
    st.session_state.actions_back = actions_back
    st.session_state.st_hip = st_hip
    st.session_state.rows_total = rows_total
    st.session_state.rows_bottom = rows_bottom

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import io

# Подключаем шрифт DejaVuSans (файл DejaVuSans.ttf нужно положить рядом с App.py)
pdfmetrics.registerFont(TTFont("DejaVuSans", "DejaVuSans.ttf"))

if st.session_state.actions and st.session_state.actions_back:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()

    # Используем кириллический шрифт во всех стилях
    styles["Normal"].fontName = "DejaVuSans"
    styles["Heading1"].fontName = "DejaVuSans"
    styles["Heading2"].fontName = "DejaVuSans"

    # Заголовок
    elements.append(Paragraph("🧶 Интерактивное вязание — инструкция", styles['Heading1']))
    elements.append(Spacer(1, 12))

    # Сводка
    summary_data = [
        ["Набрать петель", str(st.session_state.st_hip)],
        ["Всего рядов", str(st.session_state.rows_total)],
        ["Низ (до проймы и плеча)", str(st.session_state.rows_bottom)]
    ]
    table = Table(summary_data, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), "DejaVuSans"),
        ("FONTSIZE", (0,0), (-1,-1), 10),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 12))

    # Таблица переда
    elements.append(Paragraph("Инструкция для переда", styles['Heading2']))
    tbl_front_data = st.session_state.get("table_front", [])
    if not tbl_front_data:
        tbl_front_data = [["—", "Нет данных", "—"]]
    tbl_front = Table(tbl_front_data, hAlign="LEFT")
    tbl_front.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), "DejaVuSans"),
        ("FONTSIZE", (0,0), (-1,-1), 10),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
    ]))
    elements.append(tbl_front)
    elements.append(Spacer(1, 12))

    # Таблица спинки
    elements.append(Paragraph("Инструкция для спинки", styles['Heading2']))
    tbl_back_data = st.session_state.get("table_back", [])
    if not tbl_back_data:
        tbl_back_data = [["—", "Нет данных", "—"]]
    tbl_back = Table(tbl_back_data, hAlign="LEFT")
    tbl_back.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), "DejaVuSans"),
        ("FONTSIZE", (0,0), (-1,-1), 10),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
    ]))
    elements.append(tbl_back)

    # Формируем PDF
    doc.build(elements)
    buffer.seek(0)

    st.download_button(
        label="📥 Скачать PDF",
        data=buffer,
        file_name="vyazanie_instructions.pdf",
        mime="application/pdf"
    )
else:
    st.info("Сначала нажмите '🔄 Рассчитать'")
