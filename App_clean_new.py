# App_clean.py
import streamlit as st
import numpy as np
import pandas as pd
import math
from collections import defaultdict

st.title("🧶 Мини-калькулятор: перед+спинка (частичное вязание горловины)")

# -----------------------------
# Инициализация состояния
# -----------------------------
if "table_front" not in st.session_state:
    st.session_state.table_front = []
if "table_back" not in st.session_state:
    st.session_state.table_back = []

# -----------------------------
# Утилиты
# -----------------------------
def cm_to_st(cm, dens_st_10):
    """Петли в 10 см → перевод см → петли (целые)."""
    return int(round((cm / 10.0) * dens_st_10))

def cm_to_rows(cm, dens_row_10):
    """Ряды в 10 см → перевод см → ряды (целые)."""
    return int(round((cm / 10.0) * dens_row_10))

def even_cast_on(x):
    """Набор всегда чётный: если нечёт — плюс 1."""
    return x if x % 2 == 0 else x + 1

def clamp01(x):
    return max(0.0, min(1.0, x))

def split_int_sum(total, steps):
    """Распределяет total по steps целыми, максимально ровно."""
    if steps <= 0 or total <= 0:
        return [0] * max(steps, 0)
    base = total // steps
    rem  = total % steps
    out = [base] * steps
    for i in range(rem):
        out[i] += 1
    return out

# -----------------------------
# Ряды «прямо» и шаблоны вывода
# -----------------------------
def push_plain(table_rows, start, end, tag="—"):
    if start > end:
        return
    label = str(start) if start == end else f"{start}-{end}"
    table_rows.append((label, "Прямо", tag))

def section_tag_simple(row, neck_start_row, shoulder_start_row):
    tags = []
    if neck_start_row and row >= neck_start_row:
        tags.append("Горловина")
    if shoulder_start_row and row >= shoulder_start_row:
        tags.append("Скос плеча")
    return " + ".join(tags) if tags else "—"

# -----------------------------
# Горловина (частичное вязание, круглая)
# -----------------------------
def calc_neckline_partial(
    W,            # ширина горловины, петли
    H,            # глубина горловины, ряды
    row0,         # первый ряд горловины
    rows_total,   # всего рядов
    straight_pct=0.10  # верхний процент глубины без убавок по горловине
):
    """
    Правила:
    - первый ряд горловины: ТОЛЬКО центральное закрытие (без плеч);
    - последние straight_pct*H рядов — прямо (горловина без убавок);
    - частичное вязание: убавка в каждом ряду, но сторона по чётности:
        нечётный ряд → каретка справа → убавка слева,
        чётный ряд  → каретка слева  → убавка справа.
    - адаптивный центр: мелкая глубина → больший центр, глубокая → меньший.
    """
    actions = []
    if W <= 0 or H <= 0:
        return actions

    # 1) Центральное закрытие (адаптивная доля)
    H_min, H_max = 10, 48
    f_max, f_min = 0.70, 0.52
    t = clamp01((H - H_min) / (H_max - H_min))
    f = f_max - (f_max - f_min) * t

    C = int(round(W * f))
    if C % 2 == 1:
        C += 1
    C = min(C, W if W % 2 == 0 else W - 1)
    C = max(2, C)
    R_total = W - C  # остаток по горловине (на обе стороны)
    use_row0 = max(6, min(row0, rows_total - 2))

    # Только центр в первом ряду горловины
    actions.append((use_row0, f"-{C} п. горловина (центр, разделение на плечи)"))
    if R_total <= 0:
        return actions

    # 2) Верхние прямые ряды по горловине
    S = max(2, int(math.ceil(H * max(0.0, straight_pct))))
    N = H - S  # ряды для распределения остатка горловины
    if N <= 0:
        return actions

    # 3) Раскладка остатка по «полукосинусу» (мягкая «окружность»)
    per_side = R_total // 2
    weights = [(1 - math.cos(math.pi * (i / N))) / 2 for i in range(1, N + 1)]
    sum_w = sum(weights) if sum(weights) > 0 else 1.0
    raw = [per_side * w / sum_w for w in weights]
    v = [math.floor(x) for x in raw]
    need = per_side - sum(v)
    # докидаем копейки по наибольшей дробной части
    order = sorted(range(N), key=lambda i: (raw[i] - v[i]), reverse=True)
    for i in order[:need]:
        v[i] += 1
    if sum(v) == 0:
        return actions

    # Ограничим верха по 1 петле (2 последних шага)
    for i in [N - 1, N - 2]:
        if 0 <= i < N and v[i] > 1:
            diff = v[i] - 1
            v[i] = 1
            k = 0
            while diff > 0 and k < i:
                v[k] += 1
                diff -= 1
                k += 1

    # Ряды под убавки: (row0+1) .. (row0+N)
    rows_window = list(range(use_row0 + 1, use_row0 + N + 1))

    # Расставляем по чётности: odd → левый, even → правый
    rows_left  = [r for r in rows_window if r % 2 == 1]  # нечётные
    rows_right = [r for r in rows_window if r % 2 == 0]  # чётные

    # Если какой-то стороны не хватает, добираем из окна (сохраняя упорядоченность)
    def pad_rows(target, want, pool):
        if len(target) >= want:
            return target[:want]
        need = want - len(target)
        extra = [r for r in pool if r not in target]
        extra.sort()
        target = (target + extra[:need])[:want]
        target.sort()
        return target

    rows_left  = pad_rows(rows_left,  N, rows_window)
    rows_right = pad_rows(rows_right, N, rows_window)

    # Ставим убавки симметрично: на каждой стороне используем один и тот же профиль v[i]
    for i in range(N):
        if v[i] > 0:
            rl = rows_left[i]
            rr = rows_right[i]
            # левая
            actions.append((rl, f"-{v[i]} п. горловина (лево, частичное)"))
            # правая
            actions.append((rr, f"-{v[i]} п. горловина (право, частичное)"))

    return actions

# -----------------------------
# Скос плеча (плавно, в каждом 2-м ряду)
# -----------------------------
def calc_shoulder_slope(per_shoulder_st, start_row, rows_slope, rows_total, side):
    """
    per_shoulder_st — ширина плеча в петлях (на одно плечо),
    side: 'L' или 'R'
    Логика:
      - убавки строго по краю плеча,
      - в каждом втором ряду,
      - ЛЕВОЕ — нечётные ряды, ПРАВОЕ — чётные ряды,
      - НЕ делаем действие в первом ряду горловины (если совпадёт — пропустим).
    """
    actions = []
    if per_shoulder_st <= 0 or rows_slope <= 0:
        return actions

    # Ряды для плеча: от start_row до rows_total-1, шаг 2, нужной чётности
    all_rows = list(range(start_row, rows_total))
    if side == "L":
        rows = [r for r in all_rows if r % 2 == 1]  # нечётные
        label = "скос плеча (левое)"
    else:
        rows = [r for r in all_rows if r % 2 == 0]  # чётные
        label = "скос плеча (правое)"

    if not rows:
        return actions

    # Берём ровно rows_slope подходящих рядов (снизу вверх)
    rows = [r for r in rows if r >= start_row][:rows_slope]
    steps = len(rows)
    if steps <= 0:
        return actions

    parts = split_int_sum(per_shoulder_st, steps)  # плавно, но линейно по 1-2 петли
    # слегка «конусом»: первые снизу шаги могут быть на 1 п. больше, чем верхние
    # (но split_int_sum уже даёт ровный профиль, что для «плавного» ок)

    for r, v in zip(rows, parts):
        if v > 0:
            actions.append((r, f"-{v} п. {label}"))

    return actions

# -----------------------------
# Разводим конфликты: первый ряд горловины без плеча
# -----------------------------
def merge_and_separate(actions, first_neck_row):
    """
    1) В первом ряду горловины оставляем ТОЛЬКО горловину (убираем скос).
    2) Схлопываем повторы в одном ряду.
    """
    rows_map = defaultdict(list)
    for r, note in actions:
        rows_map[r].append(note)

    out = []
    for r in sorted(rows_map):
        notes = rows_map[r]
        if r == first_neck_row:
            neck_notes = [n for n in notes if "горловина" in n.lower()]
            if neck_notes:
                notes = neck_notes
        # уникализируем
        seen = set()
        uniq = []
        for n in notes:
            if n not in seen:
                uniq.append(n); seen.add(n)
        out.append((r, "; ".join(uniq)))
    return out

# -----------------------------
# Таблица с разделением плеч (одна деталь)
# -----------------------------
def make_table_split(actions, rows_total, neck_start_row, rows_slope, key=None, title="Деталь"):
    """
    Показываем:
      1) до разделения,
      2) ЛЕВОЕ ПЛЕЧО,
      3) ПРАВОЕ ПЛЕЧО (вернуться к ряду разделения),
    строки «Прямо» собираем диапазонами.
    """
    # Сбор
    m = defaultdict(list)
    for r, note in actions:
        if 1 <= r <= rows_total:
            m[r].append(note)
    if not m:
        df = pd.DataFrame([["1-" + str(rows_total), "Прямо", "—"]], columns=["Ряды", "Действия", "Сегмент"])
        st.dataframe(df, use_container_width=True, hide_index=True)
        if key: st.session_state[key] = df.values.tolist()
        return

    # ряд разделения
    split_row = None
    for r in sorted(m):
        if any("разделение на плечи" in n.lower() for n in m[r]):
            split_row = r
            break

    def tag(r):
        return section_tag_simple(r, neck_start_row, rows_total - rows_slope + 1)

    table_rows = []
    # если разделение не найдено — просто линейная таблица с «Прямо»
    if split_row is None:
        prev = 1
        for r in sorted(m):
            if r > prev:
                push_plain(table_rows, prev, r - 1, tag(prev))
            table_rows.append((str(r), "; ".join(m[r]), tag(r)))
            prev = r + 1
        if prev <= rows_total:
            push_plain(table_rows, prev, rows_total, tag(prev))
        df = pd.DataFrame(table_rows, columns=["Ряды", "Действия", "Сегмент"])
        st.dataframe(df, use_container_width=True, hide_index=True)
        if key: st.session_state[key] = table_rows
        return

    # 1) до разделения
    prev = 1
    for r in [x for x in sorted(m) if x < split_row]:
        if r > prev:
            push_plain(table_rows, prev, r - 1, tag(prev))
        table_rows.append((str(r), "; ".join(m[r]), tag(r)))
        prev = r + 1
    if prev <= split_row - 1:
        push_plain(table_rows, prev, split_row - 1, tag(prev))

    # сам split
    table_rows.append((str(split_row), "; ".join(m[split_row]), tag(split_row)))

    # 2) левое плечо
    table_rows.append(("— ЛЕВОЕ ПЛЕЧО —", "", ""))
    prev = split_row + 1
    for r in [x for x in sorted(m) if x > split_row]:
        # берём только левые события + «каждое плечо» если встретится
        notes = []
        for n in m[r]:
            ln = n.lower()
            if "левое" in ln or "(лево" in ln:
                notes.append(n)
            # горловину в частичном тоже показываем (идёт на обе стороны)
            if "горловина" in ln:
                notes.append(n)
        if notes:
            if r > prev:
                push_plain(table_rows, prev, r - 1, tag(prev))
            table_rows.append((str(r), "; ".join(sorted(set(notes), key=str)), tag(r)))
            prev = r + 1
    if prev <= rows_total:
        push_plain(table_rows, prev, rows_total, tag(prev))

    # 3) правое плечо
    table_rows.append((f"— ПРАВОЕ ПЛЕЧО — (вернитесь к ряду {split_row})", "", ""))
    prev = split_row
    for r in [split_row] + [x for x in sorted(m) if x > split_row]:
        notes = []
        for n in m[r]:
            ln = n.lower()
            if "правое" in ln or "(право" in ln:
                notes.append(n)
            if r == split_row and "горловина" in ln and "центр" in ln:
                notes.append(n)  # показать центр
            if "горловина" in ln:
                notes.append(n)
        if notes:
            if r > prev:
                push_plain(table_rows, prev, r - 1, tag(prev))
            table_rows.append((str(r), "; ".join(sorted(set(notes), key=str)), tag(r)))
            prev = r + 1
    if prev <= rows_total:
        push_plain(table_rows, prev, rows_total, tag(prev))

    df = pd.DataFrame(table_rows, columns=["Ряды", "Действия", "Сегмент"])
    st.dataframe(df, use_container_width=True, hide_index=True)
    if key: st.session_state[key] = table_rows

# -----------------------------
# Ввод
# -----------------------------
st.header("Параметры (см, плотности в 10 см)")
dens_st_10  = st.number_input("Плотность: петли в 10 см", min_value=1.0, value=21.0, step=0.5)
dens_row_10 = st.number_input("Плотность: ряды в 10 см",  min_value=1.0, value=33.0, step=0.5)

hip_cm      = st.number_input("Ширина низа (см)", min_value=1.0, value=50.0, step=0.5)
shoulders_cm= st.number_input("Ширина плеч (оба плеча, см)", min_value=1.0, value=40.0, step=0.5)
length_cm   = st.number_input("Длина изделия (см)", min_value=1.0, value=60.0, step=0.5)

neck_w_cm   = st.number_input("Ширина горловины (см)", min_value=1.0, value=18.0, step=0.5)
neck_h_front_cm = st.number_input("Глубина горловины СПЕРЕДИ (см)", min_value=1.0, value=12.0, step=0.5)
neck_h_back_cm  = st.number_input("Глубина горловины СПИНКИ (см)", min_value=1.0, value=3.0, step=0.5)

shoulder_len_cm  = st.number_input("Длина скоса плеча (по петлям, см)", min_value=1.0, value=12.0, step=0.5)
shoulder_height_cm = st.number_input("Высота скоса плеча (ряды, см)", min_value=1.0, value=12.0, step=0.5)

straight_pct = st.slider("Процент прямых рядов по горловине (верх)", 0, 40, 10)  # 10% по умолчанию

if st.button("🔄 Рассчитать"):
    # Пересчёт в петли/ряды
    st_hip       = even_cast_on(cm_to_st(hip_cm, dens_st_10))
    st_shoulders = cm_to_st(shoulders_cm, dens_st_10)          # общая ширина ДВУХ плеч
    st_neck      = cm_to_st(neck_w_cm, dens_st_10)
    rows_total   = cm_to_rows(length_cm, dens_row_10)

    rows_neck_front = cm_to_rows(neck_h_front_cm, dens_row_10)
    rows_neck_back  = cm_to_rows(neck_h_back_cm,  dens_row_10)

    st_shoulder_one = cm_to_st(shoulder_len_cm, dens_st_10)    # ширина ОДНОГО плеча в петлях (по скосу)
    rows_slope      = cm_to_rows(shoulder_height_cm, dens_row_10)

    # Старт горловины и скоса
    neck_start_front = rows_total - rows_neck_front + 1
    neck_start_back  = rows_total - rows_neck_back + 1
    shoulder_start   = rows_total - rows_slope + 1

    # 1) Низ → ширина у верха. Без проймы.
    #   Целевая ширина вверху полотна = плечи + горловина (оба плеча + центр)
    st_top = st_shoulders + st_neck
    delta_bottom = st_top - st_hip

    # Распределяем изменение ширины (симметрично) по чётным рядам,
    # и НЕ делаем ничего в первых 5 рядах (как ты просила).
    actions_base = []
    if delta_bottom != 0:
        start_change = max(6, 6)  # от 6-го ряда
        end_change = max(6, rows_total - rows_slope)  # до начала плеч
        even_rows = [r for r in range(start_change, end_change + 1) if r % 2 == 0]
        per_side = abs(delta_bottom) // 2
        steps = min(len(even_rows), per_side)
        if steps > 0:
            parts = split_int_sum(per_side, steps)
            idxs = np.linspace(0, len(even_rows) - 1, num=steps, dtype=int)
            chosen = [even_rows[i] for i in idxs]
            for r, v in zip(chosen, parts):
                if delta_bottom > 0:
                    actions_base.append((r, f"+{v} п. бок (слева)"))
                    actions_base.append((r, f"+{v} п. бок (справа)"))
                else:
                    actions_base.append((r, f"-{v} п. бок (слева)"))
                    actions_base.append((r, f"-{v} п. бок (справа)"))

    # 2) Горловина (перед)
    actions_front = actions_base.copy()
    actions_front += calc_neckline_partial(
        W=st_neck,
        H=rows_neck_front,
        row0=neck_start_front,
        rows_total=rows_total,
        straight_pct=straight_pct / 100.0
    )
    # 3) Скос плеч (перед): левое/правое раздельно, каждый 2-й ряд
    actions_front += calc_shoulder_slope(st_shoulder_one, shoulder_start, rows_slope, rows_total, side="L")
    actions_front += calc_shoulder_slope(st_shoulder_one, shoulder_start, rows_slope, rows_total, side="R")

    # Только в первом ряду горловины — без плеча
    actions_front = merge_and_separate(actions_front, neck_start_front)
    make_table_split(actions_front, rows_total, neck_start_front, rows_slope, key="table_front", title="Перед")

    # 4) Горловина (спинка)
    actions_back = actions_base.copy()
    actions_back += calc_neckline_partial(
        W=st_neck,
        H=rows_neck_back,
        row0=neck_start_back,
        rows_total=rows_total,
        straight_pct=straight_pct / 100.0
    )
    actions_back += calc_shoulder_slope(st_shoulder_one, shoulder_start, rows_slope, rows_total, side="L")
    actions_back += calc_shoulder_slope(st_shoulder_one, shoulder_start, rows_slope, rows_total, side="R")
    actions_back = merge_and_separate(actions_back, neck_start_back)
    make_table_split(actions_back, rows_total, neck_start_back, rows_slope, key="table_back", title="Спинка")

    # Сводка
    st.subheader("📊 Сводка")
    st.write(f"- Набор петель: **{st_hip}** (чётный)")
    st.write(f"- Верхняя ширина (плечи+горло): **{st_top} п.**")
    st.write(f"- Всего рядов: **{rows_total}**")
    st.write(f"- Скос плеча: **{rows_slope} рядов**, по **{st_shoulder_one} п.** на каждое плечо")

    # PDF (опционально — можно подключить позже)
    st.info("PDF генератор можно добавить по аналогии, чтобы не раздувать код: использовать st.session_state.table_front/back.")
