import streamlit as st
import numpy as np
import pandas as pd
from collections import defaultdict
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
# Рядовые правила
# -----------------------------
def allowed_rows_any(start_row: int, end_row: int, rows_total: int, forbid_last_two=True):
    """Любые ряды. Можно ограничить, чтобы не ставить в последние 2 ряда."""
    if end_row is None:
        end_row = rows_total
    hi = end_row if not forbid_last_two else min(end_row, rows_total-2)
    lo = max(1, start_row)
    if lo > hi:
        return []
    return list(range(lo, hi+1))

def allowed_rows_parity(start_row: int, end_row: int, rows_total: int, want_even=True, forbid_last_two=True):
    rows = allowed_rows_any(start_row, end_row, rows_total, forbid_last_two=forbid_last_two)
    if want_even:
        return [r for r in rows if r % 2 == 0]
    else:
        return [r for r in rows if r % 2 == 1]

def split_total_into_steps(total: int, steps: int):
    """Распределить total на steps натуральных слагаемых (>=1), как можно ровнее."""
    if total <= 0 or steps <= 0:
        return []
    base = total // steps
    rem  = total % steps
    out = [base]*(steps)
    for i in range(rem):
        out[i] += 1
    # Гарантировать >=1
    out = [max(1, x) for x in out]
    return out

# -----------------------------
# Симметричные прибавки/убавки (в одном ряду)
# -----------------------------
def sym_increases(total_add, start_row, end_row, rows_total, label):
    """Симметрично: обе стороны в одном ряду."""
    if total_add <= 0: return []
    if total_add % 2 == 1: total_add += 1
    per_side = total_add // 2

    # можно в любые ряды (дальше корректируем сторону каретки функцией fix_carriage_side)
    rows = allowed_rows_any(start_row, end_row, rows_total, forbid_last_two=True)
    if not rows: return []

    steps = min(len(rows), per_side)  # по шагам (рядов) не больше, чем требуется
    # равномерно по шагам
    parts = split_total_into_steps(per_side, steps)
    # равномерная выборка рядов
    idxs  = np.linspace(0, len(rows)-1, num=steps, dtype=int)
    chosen= [rows[i] for i in idxs]

    out = []
    for r, v in zip(chosen, parts):
        out.append((r, f"+{v} п. {label} (справа)"))
        out.append((r, f"+{v} п. {label} (слева)"))
    return out

def sym_decreases(total_sub, start_row, end_row, rows_total, label):
    """Симметрично: обе стороны в одном ряду."""
    if total_sub <= 0: return []
    if total_sub % 2 == 1: total_sub += 1
    per_side = total_sub // 2

    rows = allowed_rows_any(start_row, end_row, rows_total, forbid_last_two=True)
    if not rows: return []

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
# Пройма (аккуратная скруглённая)
# -----------------------------
def calc_round_armhole(st_chest, st_shoulders, start_row, shoulder_start_row, rows_total,
                       depth_percent=0.05, hold_percent=0.10):
    """Убавки внутрь (слегка), затем прямо, затем расширение к плечам (овер)."""
    if shoulder_start_row <= start_row:
        return []

    end_row = shoulder_start_row - 1
    total_rows = end_row - start_row + 1
    if total_rows <= 0:
        return []

    depth_armhole_st = int(round(st_chest * depth_percent))
    st_mid = st_chest - depth_armhole_st

    rows_smooth = int(total_rows * 0.4)
    rows_hold   = int(total_rows * hold_percent)

    actions = []
    # внутрь
    delta1 = st_mid - st_chest
    if delta1 < 0:
        actions += sym_decreases(-delta1, start_row, start_row+rows_smooth, rows_total, "пройма")
    # прямо
    # наружу
    delta2 = st_shoulders - st_mid
    if delta2 > 0:
        actions += sym_increases(delta2, start_row+rows_smooth+rows_hold, end_row, rows_total, "пройма")

    return actions

# -----------------------------
# Горловина: 60% + верхние 20% прямо
# -----------------------------
def neckline_plan(total_stitches, total_rows, start_row, rows_total, straight_percent=0.20):
    """
    Возвращает:
    - split_row: ряд центрального закрытия (только горловина!)
    - split_dec: величина центрального закрытия (60%, чётное)
    - left_seq:  список (row, dec) для левого плеча (горловина)
    - right_seq: список (row, dec) для правого плеча (горловина)
    Правила:
    - верхние straight_percent глубины горловины — без убавок по горловине;
    - левое: горловина на ЧЁТНЫХ; правое: горловина на НЕЧЁТНЫХ.
    - если рядов мало — увеличиваем размер шага (до 2,3...), чтобы уложиться.
    """
    if total_stitches <= 0 or total_rows <= 0:
        return None, 0, [], []

    # 60% (чётно)
    first_dec = int(round(total_stitches * 0.60))
    if first_dec % 2 == 1:
        first_dec += 1
    if first_dec > total_stitches:
        first_dec = total_stitches if total_stitches % 2 == 0 else total_stitches - 1
    rest = max(0, total_stitches - first_dec)

    # верхние 20% глубины — без горловины
    straight_rows = max(1, int(np.ceil(total_rows * max(0.20, straight_percent))))
    neck_end_by_depth = start_row + total_rows - 1 - straight_rows
    if neck_end_by_depth < start_row:
        neck_end_by_depth = start_row
    effective_end = min(neck_end_by_depth, rows_total-2)

    # split row — первый доступный ряд (не последний)
    split_row = max(1, min(start_row, rows_total-2))
    # пост-шаги горловины — от split_row+1 до effective_end
    left_rows  = allowed_rows_parity(split_row+1, effective_end, rows_total, want_even=True,  forbid_last_two=True)  # чётные
    right_rows = allowed_rows_parity(split_row+1, effective_end, rows_total, want_even=False, forbid_last_two=True)  # нечётные

    # делим остаток поровну на плечи
    left_need  = rest // 2
    right_need = rest - left_need

    def spread_need(need, rows):
        if need <= 0 or not rows:
            return []
        steps = min(len(rows), need)  # минимально по 1
        parts = split_total_into_steps(need, steps)
        idxs  = np.linspace(0, len(rows)-1, num=steps, dtype=int)
        chosen= [rows[i] for i in idxs]
        return list(zip(chosen, parts))

    left_seq  = spread_need(left_need,  left_rows)
    right_seq = spread_need(right_need, right_rows)

    # если рядов меньше, чем нужно по 1 — parts уже раздуты (>1) и всё равно сойдётся по сумме
    return split_row, first_dec, left_seq, right_seq

# -----------------------------
# Скос плеча: левое — нечётные, правое — чётные
# -----------------------------
def shoulders_plan(st_shldr, shoulder_start_row, rows_total):
    """
    Раскладываем УБАВКИ ПО ПЛЕЧУ:
    - левое плечо: только НЕЧЁТНЫЕ;
    - правое плечо: только ЧЁТНЫЕ;
    сумма по каждому плечу = st_shldr (чтобы уйти в 0 к концу).
    """
    if st_shldr <= 0:
        return [], []

    left_rows  = allowed_rows_parity(shoulder_start_row, rows_total-1, rows_total, want_even=False, forbid_last_two=True)
    right_rows = allowed_rows_parity(shoulder_start_row, rows_total-1, rows_total, want_even=True,  forbid_last_two=True)

    def spread(total, rows):
        if total <= 0 or not rows:
            return []
        steps = min(len(rows), total)  # минимум по 1 за шаг
        parts = split_total_into_steps(total, steps)
        idxs  = np.linspace(0, len(rows)-1, num=steps, dtype=int)
        chosen = [rows[i] for i in idxs]
        return list(zip(chosen, parts))

    left_seq  = spread(st_shldr, left_rows)
    right_seq = spread(st_shldr, right_rows)
    return left_seq, right_seq

# -----------------------------
# Горловина + плечи (единый план, без плеча в split_row)
# -----------------------------
def plan_front_or_back(neck_st, neck_rows, neck_start_row,
                       st_shldr, rows_slope, rows_total,
                       forbid_first_row_shoulder=True, straight_percent=0.20):
    """
    Возвращает список действий:
    - split_row с центральным закрытием (только горловина);
    - левое плечо: горловина(чётные) + плечо(нечётные);
    - правое плечо: горловина(нечётные) + плечо(чётные);
    - плечевые убавки суммарно = st_shldr, горловина — остаток от neck_st после 60%;
    """
    actions = []

    # Горловина
    split_row, split_dec, left_neck, right_neck = neckline_plan(
        neck_st, neck_rows, neck_start_row, rows_total, straight_percent
    )
    if split_row is None:
        return actions

    # 1) split row: ТОЛЬКО горловина
    actions.append((split_row, f"-{split_dec} п. горловина (середина, разделение на плечи)"))

    # 2) Плечо — ряды и объёмы
    shoulder_start_row = rows_total - rows_slope + 1
    left_sh, right_sh = shoulders_plan(st_shldr, shoulder_start_row, rows_total)

    # Если плечо попадает в split_row — выкидываем этот шаг (правило: в split_row только горловина)
    if forbid_first_row_shoulder:
        left_sh  = [(r,v) for (r,v) in left_sh  if r != split_row]
        right_sh = [(r,v) for (r,v) in right_sh if r != split_row]

    # 3) Левое плечо: горловина (чётные), плечо (нечётные)
    for r, v in left_neck:
        actions.append((r, f"-{v} п. горловина [L]"))
    for r, v in left_sh:
        actions.append((r, f"-{v} п. скос плеча [L]"))

    # 4) Правое плечо: горловина (нечётные), плечо (чётные)
    for r, v in right_neck:
        actions.append((r, f"-{v} п. горловина [R]"))
    for r, v in right_sh:
        actions.append((r, f"-{v} п. скос плеча [R]"))

    return actions

# -----------------------------
# Слияние и разведение конфликтов
# -----------------------------
def merge_actions(actions, rows_total):
    """
    Объединяем по рядам (сохраняем порядок), убираем дубль-строки.
    Если в split_row есть плечо — выносим его из split_row (подстраховка).
    """
    merged = defaultdict(list)
    for row, note in actions:
        if isinstance(row, int) and 1 <= row <= rows_total:
            merged[row].append(note)

    # найти split_row
    split_row = None
    for r in sorted(merged.keys()):
        if any("разделение на плечи" in n.lower() or "середина, разделение" in n.lower() for n in merged[r]):
            split_row = r
            break

    final_map = defaultdict(list)
    used = set()

    for r in sorted(merged.keys()):
        # убираем дубли
        seen = set()
        cleaned = []
        for n in merged[r]:
            if n not in seen:
                cleaned.append(n)
                seen.add(n)

        # если это split_row — удалим всё плечевое
        if r == split_row:
            shoulder_notes = [n for n in cleaned if "скос плеча" in n.lower()]
            non_shoulder   = [n for n in cleaned if n not in shoulder_notes]
            final_map[r].extend(non_shoulder)
            used.add(r)
            # плечи перекинем на ближайшие допустимые ряды (сохраняя их парность; но
            # у нас уже плечи запланированы вне split_row в plan_front_or_back)
        else:
            final_map[r].extend(cleaned)
            used.add(r)

    # вернуть плоским списком
    out = []
    for r in sorted(final_map.keys()):
        out.append((r, "; ".join(final_map[r])))
    return out

# -----------------------------
# Коррекция стороны каретки
# -----------------------------
def fix_carriage_side(actions, method=None):
    """
    Нечётный ряд — каретка справа; чётный — слева.
    Стандартные: убавки со стороны каретки; Частичные: с противоположной.
    Переносим на соседний ряд, если в помете явно указан «справа/слева» и это не совпадает.
    """
    if method is None:
        method = st.session_state.get("method", "Стандартные (со стороны каретки)")
    use_std = method.startswith("Стандартные")
    fixed = []

    for r, note in actions:
        note_lower = note.lower()
        if r % 2 == 1:  # нечёт — каретка справа
            correct_side = "справа" if use_std else "слева"
        else:           # чёт — каретка слева
            correct_side = "слева" if use_std else "справа"

        if (("справа" in note_lower) or ("слева" in note_lower)) and (correct_side not in note_lower):
            new_r = r - 1 if r > 1 else r + 1
            if new_r <= 0: new_r = 1
            fixed.append((new_r, note))
        else:
            fixed.append((r, note))
    return fixed

# -----------------------------
# Сегменты/подписи
# -----------------------------
def segment_label(row, rows_to_armhole_end, neck_start_row, shoulder_start_row):
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
# Таблица «перед» с разделением
# -----------------------------
def make_table_front_split(actions, rows_count, rows_to_armhole_end, neck_start_row, shoulder_start_row, key=None):
    # собрать
    merged = defaultdict(list)
    for row, note in actions:
        if isinstance(row, int) and 1 <= row <= rows_count:
            merged[row].append(note)

    if not merged:
        df = pd.DataFrame([["1-"+str(rows_count), "Прямо", segment_label(1, rows_to_armhole_end, neck_start_row, shoulder_start_row)]],
                          columns=["Ряды", "Действия", "Сегмент"])
        st.dataframe(df, use_container_width=True, hide_index=True)
        if key: st.session_state[key] = df.values.tolist()
        return

    # найти split_row
    split_row = None
    for r in sorted(merged.keys()):
        if any("середина, разделение" in n.lower() for n in merged[r]):
            split_row = r
            break

    # Если нет разделения — обычная таблица
    if split_row is None:
        make_table_full(actions, rows_count, rows_to_armhole_end, neck_start_row, shoulder_start_row, key=key)
        return

    def push_plain(table, a, b):
        if a > b: return
        seg = segment_label(a, rows_to_armhole_end, neck_start_row, shoulder_start_row)
        table.append((str(a) if a==b else f"{a}-{b}", "Прямо", seg))

    def clean_notes(notes):
        cleaned = []
        for n in notes:
            cleaned.append(n.replace("[L]","").replace("[R]","").strip())
        return cleaned

    rows_sorted = sorted(merged.keys())
    table_rows = []

    # До split_row
    prev = 1
    for r in [x for x in rows_sorted if x < split_row]:
        if r > prev:
            push_plain(table_rows, prev, r-1)
        table_rows.append((str(r), "; ".join(clean_notes(merged[r])),
                           segment_label(r, rows_to_armhole_end, neck_start_row, shoulder_start_row)))
        prev = r + 1
    if prev <= split_row - 1:
        push_plain(table_rows, prev, split_row-1)

    # split_row
    split_notes = [n for n in merged[split_row] if "горловина" in n.lower()]
    table_rows.append((str(split_row), "; ".join(clean_notes(split_notes)),
                       segment_label(split_row, rows_to_armhole_end, neck_start_row, shoulder_start_row)))

    # подпись с остатками
    # остаток на каждое плечо = st_shldr + (neck_rest/2) (покажем в тексте позже в блоке сводки — здесь просто заголовок)
    table_rows.append(("— ЛЕВОЕ ПЛЕЧО —", "", ""))

    # левое плечо: ноты с [L] + «каждое плечо» (если есть такой текст)
    left_prev = split_row + 1
    left_rows = []
    for r in [x for x in rows_sorted if x > split_row]:
        sel = []
        for n in merged[r]:
            ln = n.lower()
            if "[l]" in ln or "левое плечо" in ln:
                sel.append(n)
            # горловинные шаги для левого помечены [L], уже попадают
        if sel:
            left_rows.append((r, sel))
    for r, notes in left_rows:
        if r > left_prev:
            push_plain(table_rows, left_prev, r-1)
        table_rows.append((str(r), "; ".join(clean_notes(notes)),
                           segment_label(r, rows_to_armhole_end, neck_start_row, shoulder_start_row)))
        left_prev = r + 1
    if left_prev <= rows_count:
        push_plain(table_rows, left_prev, rows_count)

    # правое плечо
    table_rows.append((f"— ПРАВОЕ ПЛЕЧО — (вернитесь к ряду {split_row})", "", ""))

    right_prev = split_row
    right_rows = []
    cand = [split_row] + [x for x in rows_sorted if x > split_row]
    for r in cand:
        sel = []
        for n in merged.get(r, []):
            ln = n.lower()
            if "[r]" in ln or "правое плечо" in ln:
                sel.append(n)
            # горловинные шаги для правого помечены [R]
        if r == split_row and any("середина, разделение" in n.lower() for n in merged[r]):
            sel.append("↳ переход к правому плечу")
        if sel:
            right_rows.append((r, sel))
    for r, notes in right_rows:
        if r > right_prev:
            push_plain(table_rows, right_prev, r-1)
        table_rows.append((str(r), "; ".join(clean_notes(notes)),
                           segment_label(r, rows_to_armhole_end, neck_start_row, shoulder_start_row)))
        right_prev = r + 1
    if right_prev <= rows_count:
        push_plain(table_rows, right_prev, rows_count)

    df = pd.DataFrame(table_rows, columns=["Ряды", "Действия", "Сегмент"])
    st.dataframe(df, use_container_width=True, hide_index=True)
    if key: st.session_state[key] = table_rows

# -----------------------------
# Таблица «спинка» с разделением (аналогично)
# -----------------------------
def make_table_back_split(actions, rows_count, rows_to_armhole_end, neck_start_row, shoulder_start_row, key=None):
    # используем ту же логику
    make_table_front_split(actions, rows_count, rows_to_armhole_end, neck_start_row, shoulder_start_row, key=key)

# -----------------------------
# Обычная таблица (без разделения)
# -----------------------------
def make_table_full(actions, rows_count, rows_to_armhole_end, neck_start_row, shoulder_start_row, key=None):
    merged = defaultdict(list)
    for row, note in actions:
        if isinstance(row, int) and 1 <= row <= rows_count:
            merged[row].append(note)
    rows_sorted = sorted(merged.keys())
    table_rows = []
    prev = 1
    if not rows_sorted:
        table_rows.append((f"1-{rows_count}", "Прямо",
                           segment_label(1, rows_to_armhole_end, neck_start_row, shoulder_start_row)))
    else:
        for r in rows_sorted:
            if r > prev:
                seg = segment_label(prev, rows_to_armhole_end, neck_start_row, shoulder_start_row)
                table_rows.append((str(prev) if prev==r-1 else f"{prev}-{r-1}", "Прямо", seg))
            table_rows.append((str(r), "; ".join(merged[r]),
                               segment_label(r, rows_to_armhole_end, neck_start_row, shoulder_start_row)))
            prev = r + 1
        if prev <= rows_count:
            seg = segment_label(prev, rows_to_armhole_end, neck_start_row, shoulder_start_row)
            table_rows.append((str(prev) if prev==rows_count else f"{prev}-{rows_count}", "Прямо", seg))
    df = pd.DataFrame(table_rows, columns=["Ряды", "Действия", "Сегмент"])
    st.dataframe(df, use_container_width=True, hide_index=True)
    if key: st.session_state[key] = table_rows

# -----------------------------
# Ввод
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

method = st.selectbox(
    "Метод убавок",
    ["Стандартные (со стороны каретки)", "Частичное вязание (поворотные ряды)"],
    index=0
)

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
# Кнопка «Рассчитать»
# -----------------------------
if st.button("🔄 Рассчитать"):
    fields = [density_st_str, density_row_str, hip_cm_str, chest_cm_str, length_cm_str,
              armhole_depth_cm_str, neck_width_cm_str, neck_depth_cm_str, neck_depth_back_cm_str,
              shoulder_len_cm_str, shoulder_slope_cm_str]
    if not all(fields):
        st.error("⚠️ Заполни все поля.")
        st.stop()
    try:
        (density_st, density_row, hip_cm, chest_cm, length_cm,
         armhole_depth_cm, neck_width_cm, neck_depth_cm, neck_depth_back_cm,
         shoulder_len_cm, shoulder_slope_cm) = parse_inputs()
    except:
        st.error("⚠️ Только числа (точка/запятая).")
        st.stop()

    # Пересчёт
    st_hip     = cm_to_st(hip_cm, density_st)
    if st_hip % 2: st_hip += 1  # правило 1
    st_chest   = cm_to_st(chest_cm, density_st)

    rows_total = cm_to_rows(length_cm, density_row)
    rows_armh  = cm_to_rows(armhole_depth_cm, density_row)

    neck_st          = cm_to_st(neck_width_cm, density_st)
    neck_rows_front  = cm_to_rows(neck_depth_cm, density_row)
    neck_rows_back   = cm_to_rows(neck_depth_back_cm, density_row)

    st_shldr   = cm_to_st(shoulder_len_cm, density_st)  # убавок по плечу на КАЖДОМ плече
    rows_slope = cm_to_rows(shoulder_slope_cm, density_row)

    st_shoulders = 2*st_shldr + neck_st
    rows_bottom  = rows_total - rows_armh - rows_slope
    armhole_start_row  = rows_bottom + 1
    shoulder_start_row = rows_total - rows_slope + 1
    armhole_end_row    = shoulder_start_row - 1

    neck_start_row_front = rows_total - neck_rows_front + 1
    neck_start_row_back  = rows_total - neck_rows_back + 1

    # Сводка
    st.subheader("📊 Сводка")
    st.write(f"- Набрать петель: **{st_hip}** (приведено к чётному)")
    st.write(f"- Всего рядов: **{rows_total}**")

    # -------- ПЕРЕД --------
    st.subheader("📋 Инструкция для переда")
    actions = []

    # Низ: симметрично от низа до начала проймы
    delta_bottom = st_chest - st_hip
    if delta_bottom > 0:
        actions += sym_increases(delta_bottom, 6, rows_bottom, rows_total, "бок")  # с 6 ряда
    elif delta_bottom < 0:
        actions += sym_decreases(-delta_bottom, 6, rows_bottom, rows_total, "бок")
    

    # Пройма (до плеч)
    actions += calc_round_armhole(st_chest, st_shoulders, armhole_start_row, shoulder_start_row, rows_total)

    # Горловина + плечи (единый план, НЕТ плеча в split_row)
    actions += plan_front_or_back(
        neck_st=neck_st,
        neck_rows=neck_rows_front,
        neck_start_row=neck_start_row_front,
        st_shldr=st_shldr,
        rows_slope=rows_slope,
        rows_total=rows_total,
        forbid_first_row_shoulder=True,
        straight_percent=0.20
    )

    # Слияние + сторона каретки
    actions = merge_actions(actions, rows_total)
    actions = fix_carriage_side(actions, method)

    # Таблица (с разбиением)
    make_table_front_split(actions, rows_total, rows_bottom, neck_start_row_front, shoulder_start_row, key="table_front")

    # -------- СПИНКА --------
    st.subheader("📋 Инструкция для спинки")
    actions_back = []
# -------- Низ --------
    if delta_bottom > 0:
    actions_back += sym_increases(delta_bottom, 6, rows_bottom, rows_total, "бок")
elif delta_bottom < 0:
    actions_back += sym_decreases(-delta_bottom, 6, rows_bottom, rows_total, "бок")

    # Пройма
    delta_armh = st_shoulders - st_chest
    if delta_armh > 0:
        actions_back += sym_increases(delta_armh, armhole_start_row, armhole_end_row, rows_total, "пройма")
    elif delta_armh < 0:
        actions_back += sym_decreases(-delta_armh, armhole_start_row, armhole_end_row, rows_total, "пройма")

    # Горловина спинки + плечи (те же правила)
    actions_back += plan_front_or_back(
        neck_st=neck_st,
        neck_rows=neck_rows_back,
        neck_start_row=neck_start_row_back,
        st_shldr=st_shldr,
        rows_slope=rows_slope,
        rows_total=rows_total,
        forbid_first_row_shoulder=True,
        straight_percent=0.20
    )

    actions_back = merge_actions(actions_back, rows_total)
    actions_back = fix_carriage_side(actions_back, method)
    make_table_back_split(actions_back, rows_total, rows_bottom, neck_start_row_back, shoulder_start_row, key="table_back")

    # сохранить для PDF
    st.session_state.actions = actions
    st.session_state.actions_back = actions_back
    st.session_state.st_hip = st_hip
    st.session_state.rows_total = rows_total
    st.session_state.rows_bottom = rows_bottom

# -----------------------------
# PDF
# -----------------------------
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import io

pdfmetrics.registerFont(TTFont("DejaVuSans", "DejaVuSans.ttf"))

if st.session_state.actions and st.session_state.actions_back:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()
    styles["Normal"].fontName = "DejaVuSans"
    styles["Heading1"].fontName = "DejaVuSans"
    styles["Heading2"].fontName = "DejaVuSans"

    elements.append(Paragraph("🧶 Интерактивное вязание — инструкция", styles['Heading1']))
    elements.append(Spacer(1, 12))

    summary_data = [
        ["Набрать петель (чётно)", str(st.session_state.st_hip)],
        ["Всего рядов", str(st.session_state.rows_total)],
        ["Низ (до проймы)", str(st.session_state.rows_bottom)]
    ]
    tbl = Table(summary_data, hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), "DejaVuSans"),
        ("FONTSIZE", (0,0), (-1,-1), 10),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
    ]))
    elements.append(tbl)
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Инструкция для переда", styles['Heading2']))
    front = st.session_state.get("table_front", [["—","Нет данных","—"]])
    tbl_front = Table(front, hAlign="LEFT")
    tbl_front.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), "DejaVuSans"),
        ("FONTSIZE", (0,0), (-1,-1), 10),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
    ]))
    elements.append(tbl_front)
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Инструкция для спинки", styles['Heading2']))
    back = st.session_state.get("table_back", [["—","Нет данных","—"]])
    tbl_back = Table(back, hAlign="LEFT")
    tbl_back.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), "DejaVuSans"),
        ("FONTSIZE", (0,0), (-1,-1), 10),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
    ]))
    elements.append(tbl_back)

    doc.build(elements)
    buffer.seek(0)
    st.download_button("📥 Скачать PDF", buffer, file_name="vyazanie_instructions.pdf", mime="application/pdf")
else:
    st.info("Сначала нажми «🔄 Рассчитать».")


# -----------------------------
# Сессия
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
# Рядовые правила
# -----------------------------
def allowed_rows_any(start_row: int, end_row: int, rows_total: int, forbid_last_two=True):
    """Любые ряды. Можно ограничить, чтобы не ставить в последние 2 ряда."""
    if end_row is None:
        end_row = rows_total
    hi = end_row if not forbid_last_two else min(end_row, rows_total-2)
    lo = max(1, start_row)
    if lo > hi:
        return []
    return list(range(lo, hi+1))

def allowed_rows_parity(start_row: int, end_row: int, rows_total: int, want_even=True, forbid_last_two=True):
    rows = allowed_rows_any(start_row, end_row, rows_total, forbid_last_two=forbid_last_two)
    if want_even:
        return [r for r in rows if r % 2 == 0]
    else:
        return [r for r in rows if r % 2 == 1]

def split_total_into_steps(total: int, steps: int):
    """Распределить total на steps натуральных слагаемых (>=1), как можно ровнее."""
    if total <= 0 or steps <= 0:
        return []
    base = total // steps
    rem  = total % steps
    out = [base]*(steps)
    for i in range(rem):
        out[i] += 1
    # Гарантировать >=1
    out = [max(1, x) for x in out]
    return out

# -----------------------------
# Симметричные прибавки/убавки (в одном ряду)
# -----------------------------
def sym_increases(total_add, start_row, end_row, rows_total, label):
    """Симметрично: обе стороны в одном ряду."""
    if total_add <= 0: return []
    if total_add % 2 == 1: total_add += 1
    per_side = total_add // 2

    # можно в любые ряды (дальше корректируем сторону каретки функцией fix_carriage_side)
    rows = allowed_rows_any(start_row, end_row, rows_total, forbid_last_two=True)
    if not rows: return []

    steps = min(len(rows), per_side)  # по шагам (рядов) не больше, чем требуется
    # равномерно по шагам
    parts = split_total_into_steps(per_side, steps)
    # равномерная выборка рядов
    idxs  = np.linspace(0, len(rows)-1, num=steps, dtype=int)
    chosen= [rows[i] for i in idxs]

    out = []
    for r, v in zip(chosen, parts):
        out.append((r, f"+{v} п. {label} (справа)"))
        out.append((r, f"+{v} п. {label} (слева)"))
    return out

def sym_decreases(total_sub, start_row, end_row, rows_total, label):
    """Симметрично: обе стороны в одном ряду."""
    if total_sub <= 0: return []
    if total_sub % 2 == 1: total_sub += 1
    per_side = total_sub // 2

    rows = allowed_rows_any(start_row, end_row, rows_total, forbid_last_two=True)
    if not rows: return []

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
# Пройма (аккуратная скруглённая)
# -----------------------------
def calc_round_armhole(st_chest, st_shoulders, start_row, shoulder_start_row, rows_total,
                       depth_percent=0.05, hold_percent=0.10):
    """Убавки внутрь (слегка), затем прямо, затем расширение к плечам (овер)."""
    if shoulder_start_row <= start_row:
        return []

    end_row = shoulder_start_row - 1
    total_rows = end_row - start_row + 1
    if total_rows <= 0:
        return []

    depth_armhole_st = int(round(st_chest * depth_percent))
    st_mid = st_chest - depth_armhole_st

    rows_smooth = int(total_rows * 0.4)
    rows_hold   = int(total_rows * hold_percent)

    actions = []
    # внутрь
    delta1 = st_mid - st_chest
    if delta1 < 0:
        actions += sym_decreases(-delta1, start_row, start_row+rows_smooth, rows_total, "пройма")
    # прямо
    # наружу
    delta2 = st_shoulders - st_mid
    if delta2 > 0:
        actions += sym_increases(delta2, start_row+rows_smooth+rows_hold, end_row, rows_total, "пройма")

    return actions

# -----------------------------
# Горловина: 60% + верхние 20% прямо
# -----------------------------
def neckline_plan(total_stitches, total_rows, start_row, rows_total, straight_percent=0.20):
    """
    Возвращает:
    - split_row: ряд центрального закрытия (только горловина!)
    - split_dec: величина центрального закрытия (60%, чётное)
    - left_seq:  список (row, dec) для левого плеча (горловина)
    - right_seq: список (row, dec) для правого плеча (горловина)
    Правила:
    - верхние straight_percent глубины горловины — без убавок по горловине;
    - левое: горловина на ЧЁТНЫХ; правое: горловина на НЕЧЁТНЫХ.
    - если рядов мало — увеличиваем размер шага (до 2,3...), чтобы уложиться.
    """
    if total_stitches <= 0 or total_rows <= 0:
        return None, 0, [], []

    # 60% (чётно)
    first_dec = int(round(total_stitches * 0.60))
    if first_dec % 2 == 1:
        first_dec += 1
    if first_dec > total_stitches:
        first_dec = total_stitches if total_stitches % 2 == 0 else total_stitches - 1
    rest = max(0, total_stitches - first_dec)

    # верхние 20% глубины — без горловины
    straight_rows = max(1, int(np.ceil(total_rows * max(0.20, straight_percent))))
    neck_end_by_depth = start_row + total_rows - 1 - straight_rows
    if neck_end_by_depth < start_row:
        neck_end_by_depth = start_row
    effective_end = min(neck_end_by_depth, rows_total-2)

    # split row — первый доступный ряд (не последний)
    split_row = max(1, min(start_row, rows_total-2))
    # пост-шаги горловины — от split_row+1 до effective_end
    left_rows  = allowed_rows_parity(split_row+1, effective_end, rows_total, want_even=True,  forbid_last_two=True)  # чётные
    right_rows = allowed_rows_parity(split_row+1, effective_end, rows_total, want_even=False, forbid_last_two=True)  # нечётные

    # делим остаток поровну на плечи
    left_need  = rest // 2
    right_need = rest - left_need

    def spread_need(need, rows):
        if need <= 0 or not rows:
            return []
        steps = min(len(rows), need)  # минимально по 1
        parts = split_total_into_steps(need, steps)
        idxs  = np.linspace(0, len(rows)-1, num=steps, dtype=int)
        chosen= [rows[i] for i in idxs]
        return list(zip(chosen, parts))

    left_seq  = spread_need(left_need,  left_rows)
    right_seq = spread_need(right_need, right_rows)

    # если рядов меньше, чем нужно по 1 — parts уже раздуты (>1) и всё равно сойдётся по сумме
    return split_row, first_dec, left_seq, right_seq

# -----------------------------
# Скос плеча: левое — нечётные, правое — чётные
# -----------------------------
def shoulders_plan(st_shldr, shoulder_start_row, rows_total):
    """
    Раскладываем УБАВКИ ПО ПЛЕЧУ:
    - левое плечо: только НЕЧЁТНЫЕ;
    - правое плечо: только ЧЁТНЫЕ;
    сумма по каждому плечу = st_shldr (чтобы уйти в 0 к концу).
    """
    if st_shldr <= 0:
        return [], []

    left_rows  = allowed_rows_parity(shoulder_start_row, rows_total-1, rows_total, want_even=False, forbid_last_two=True)
    right_rows = allowed_rows_parity(shoulder_start_row, rows_total-1, rows_total, want_even=True,  forbid_last_two=True)

    def spread(total, rows):
        if total <= 0 or not rows:
            return []
        steps = min(len(rows), total)  # минимум по 1 за шаг
        parts = split_total_into_steps(total, steps)
        idxs  = np.linspace(0, len(rows)-1, num=steps, dtype=int)
        chosen = [rows[i] for i in idxs]
        return list(zip(chosen, parts))

    left_seq  = spread(st_shldr, left_rows)
    right_seq = spread(st_shldr, right_rows)
    return left_seq, right_seq

# -----------------------------
# Горловина + плечи (единый план, без плеча в split_row)
# -----------------------------
def plan_front_or_back(neck_st, neck_rows, neck_start_row,
                       st_shldr, rows_slope, rows_total,
                       forbid_first_row_shoulder=True, straight_percent=0.20):
    """
    Возвращает список действий:
    - split_row с центральным закрытием (только горловина);
    - левое плечо: горловина(чётные) + плечо(нечётные);
    - правое плечо: горловина(нечётные) + плечо(чётные);
    - плечевые убавки суммарно = st_shldr, горловина — остаток от neck_st после 60%;
    """
    actions = []

    # Горловина
    split_row, split_dec, left_neck, right_neck = neckline_plan(
        neck_st, neck_rows, neck_start_row, rows_total, straight_percent
    )
    if split_row is None:
        return actions

    # 1) split row: ТОЛЬКО горловина
    actions.append((split_row, f"-{split_dec} п. горловина (середина, разделение на плечи)"))

    # 2) Плечо — ряды и объёмы
    shoulder_start_row = rows_total - rows_slope + 1
    left_sh, right_sh = shoulders_plan(st_shldr, shoulder_start_row, rows_total)  # до rows_total-1

    # Если плечо попадает в split_row — выкидываем этот шаг (правило: в split_row только горловина)
    if forbid_first_row_shoulder:
        left_sh  = [(r,v) for (r,v) in left_sh  if r != split_row]
        right_sh = [(r,v) for (r,v) in right_sh if r != split_row]

    # 3) Левое плечо: горловина (чётные), плечо (нечётные)
    for r, v in left_neck:
        actions.append((r, f"-{v} п. горловина [L]"))
    for r, v in left_sh:
        actions.append((r, f"-{v} п. скос плеча [L]"))

    # 4) Правое плечо: горловина (нечётные), плечо (чётные)
    for r, v in right_neck:
        actions.append((r, f"-{v} п. горловина [R]"))
    for r, v in right_sh:
        actions.append((r, f"-{v} п. скос плеча [R]"))

    return actions

# -----------------------------
# Слияние и разведение конфликтов
# -----------------------------
def merge_actions(actions, rows_total):
    """
    Объединяем по рядам (сохраняем порядок), убираем дубль-строки.
    Если в split_row есть плечо — выносим его из split_row (подстраховка).
    """
    merged = defaultdict(list)
    for row, note in actions:
        if isinstance(row, int) and 1 <= row <= rows_total:
            merged[row].append(note)

    # найти split_row
    split_row = None
    for r in sorted(merged.keys()):
        if any("разделение на плечи" in n.lower() or "середина, разделение" in n.lower() for n in merged[r]):
            split_row = r
            break

    final_map = defaultdict(list)
    used = set()

    for r in sorted(merged.keys()):
        # убираем дубли
        seen = set()
        cleaned = []
        for n in merged[r]:
            if n not in seen:
                cleaned.append(n)
                seen.add(n)

        # если это split_row — удалим всё плечевое
        if r == split_row:
            shoulder_notes = [n for n in cleaned if "скос плеча" in n.lower()]
            non_shoulder   = [n for n in cleaned if n not in shoulder_notes]
            final_map[r].extend(non_shoulder)
            used.add(r)
            # плечи перекинем на ближайшие допустимые ряды (сохраняя их парность; но
            # у нас уже плечи запланированы вне split_row в plan_front_or_back)
        else:
            final_map[r].extend(cleaned)
            used.add(r)

    # вернуть плоским списком
    out = []
    for r in sorted(final_map.keys()):
        out.append((r, "; ".join(final_map[r])))
    return out

# -----------------------------
# Коррекция стороны каретки
# -----------------------------
def fix_carriage_side(actions, method=None):
    """
    Нечётный ряд — каретка справа; чётный — слева.
    Стандартные: убавки со стороны каретки; Частичные: с противоположной.
    Переносим на соседний ряд, если в помете явно указан «справа/слева» и это не совпадает.
    """
    if method is None:
        method = st.session_state.get("method", "Стандартные (со стороны каретки)")
    use_std = method.startswith("Стандартные")
    fixed = []

    for r, note in actions:
        note_lower = note.lower()
        if r % 2 == 1:  # нечёт — каретка справа
            correct_side = "справа" if use_std else "слева"
        else:           # чёт — каретка слева
            correct_side = "слева" if use_std else "справа"

        if (("справа" in note_lower) or ("слева" in note_lower)) and (correct_side not in note_lower):
            new_r = r - 1 if r > 1 else r + 1
            if new_r <= 0: new_r = 1
            fixed.append((new_r, note))
        else:
            fixed.append((r, note))
    return fixed

# -----------------------------
# Сегменты/подписи
# -----------------------------
def segment_label(row, rows_to_armhole_end, neck_start_row, shoulder_start_row):
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
# Таблица «перед» с разделением
# -----------------------------
def make_table_front_split(actions, rows_count, rows_to_armhole_end, neck_start_row, shoulder_start_row, key=None):
    # собрать
    merged = defaultdict(list)
    for row, note in actions:
        if isinstance(row, int) and 1 <= row <= rows_count:
            merged[row].append(note)

    if not merged:
        df = pd.DataFrame([["1-"+str(rows_count), "Прямо", segment_label(1, rows_to_armhole_end, neck_start_row, shoulder_start_row)]],
                          columns=["Ряды", "Действия", "Сегмент"])
        st.dataframe(df, use_container_width=True, hide_index=True)
        if key: st.session_state[key] = df.values.tolist()
        return

    # найти split_row
    split_row = None
    for r in sorted(merged.keys()):
        if any("середина, разделение" in n.lower() for n in merged[r]):
            split_row = r
            break

    # Если нет разделения — обычная таблица
    if split_row is None:
        make_table_full(actions, rows_count, rows_to_armhole_end, neck_start_row, shoulder_start_row, key=key)
        return

    def push_plain(table, a, b):
        if a > b: return
        seg = segment_label(a, rows_to_armhole_end, neck_start_row, shoulder_start_row)
        table.append((str(a) if a==b else f"{a}-{b}", "Прямо", seg))

    def clean_notes(notes):
        cleaned = []
        for n in notes:
            cleaned.append(n.replace("[L]","").replace("[R]","").strip())
        return cleaned

    rows_sorted = sorted(merged.keys())
    table_rows = []

    # До split_row
    prev = 1
    for r in [x for x in rows_sorted if x < split_row]:
        if r > prev:
            push_plain(table_rows, prev, r-1)
        table_rows.append((str(r), "; ".join(clean_notes(merged[r])),
                           segment_label(r, rows_to_armhole_end, neck_start_row, shoulder_start_row)))
        prev = r + 1
    if prev <= split_row - 1:
        push_plain(table_rows, prev, split_row-1)

    # split_row
    split_notes = [n for n in merged[split_row] if "горловина" in n.lower()]
    table_rows.append((str(split_row), "; ".join(clean_notes(split_notes)),
                       segment_label(split_row, rows_to_armhole_end, neck_start_row, shoulder_start_row)))

    # подпись с остатками
    # остаток на каждое плечо = st_shldr + (neck_rest/2) (покажем в тексте позже в блоке сводки — здесь просто заголовок)
    table_rows.append(("— ЛЕВОЕ ПЛЕЧО —", "", ""))

    # левое плечо: ноты с [L] + «каждое плечо» (если есть такой текст)
    left_prev = split_row + 1
    left_rows = []
    for r in [x for x in rows_sorted if x > split_row]:
        sel = []
        for n in merged[r]:
            ln = n.lower()
            if "[l]" in ln or "левое плечо" in ln:
                sel.append(n)
            # горловинные шаги для левого помечены [L], уже попадают
        if sel:
            left_rows.append((r, sel))
    for r, notes in left_rows:
        if r > left_prev:
            push_plain(table_rows, left_prev, r-1)
        table_rows.append((str(r), "; ".join(clean_notes(notes)),
                           segment_label(r, rows_to_armhole_end, neck_start_row, shoulder_start_row)))
        left_prev = r + 1
    if left_prev <= rows_count:
        push_plain(table_rows, left_prev, rows_count)

    # правое плечо
    table_rows.append((f"— ПРАВОЕ ПЛЕЧО — (вернитесь к ряду {split_row})", "", ""))

    right_prev = split_row
    right_rows = []
    cand = [split_row] + [x for x in rows_sorted if x > split_row]
    for r in cand:
        sel = []
        for n in merged.get(r, []):
            ln = n.lower()
            if "[r]" in ln or "правое плечо" in ln:
                sel.append(n)
            # горловинные шаги для правого помечены [R]
        if r == split_row and any("середина, разделение" in n.lower() for n in merged[r]):
            sel.append("↳ переход к правому плечу")
        if sel:
            right_rows.append((r, sel))
    for r, notes in right_rows:
        if r > right_prev:
            push_plain(table_rows, right_prev, r-1)
        table_rows.append((str(r), "; ".join(clean_notes(notes)),
                           segment_label(r, rows_to_armhole_end, neck_start_row, shoulder_start_row)))
        right_prev = r + 1
    if right_prev <= rows_count:
        push_plain(table_rows, right_prev, rows_count)

    df = pd.DataFrame(table_rows, columns=["Ряды", "Действия", "Сегмент"])
    st.dataframe(df, use_container_width=True, hide_index=True)
    if key: st.session_state[key] = table_rows

# -----------------------------
# Таблица «спинка» с разделением (аналогично)
# -----------------------------
def make_table_back_split(actions, rows_count, rows_to_armhole_end, neck_start_row, shoulder_start_row, key=None):
    # используем ту же логику
    make_table_front_split(actions, rows_count, rows_to_armhole_end, neck_start_row, shoulder_start_row, key=key)

# -----------------------------
# Обычная таблица (без разделения)
# -----------------------------
def make_table_full(actions, rows_count, rows_to_armhole_end, neck_start_row, shoulder_start_row, key=None):
    merged = defaultdict(list)
    for row, note in actions:
        if isinstance(row, int) and 1 <= row <= rows_count:
            merged[row].append(note)
    rows_sorted = sorted(merged.keys())
    table_rows = []
    prev = 1
    if not rows_sorted:
        table_rows.append((f"1-{rows_count}", "Прямо",
                           segment_label(1, rows_to_armhole_end, neck_start_row, shoulder_start_row)))
    else:
        for r in rows_sorted:
            if r > prev:
                seg = segment_label(prev, rows_to_armhole_end, neck_start_row, shoulder_start_row)
                table_rows.append((str(prev) if prev==r-1 else f"{prev}-{r-1}", "Прямо", seg))
            table_rows.append((str(r), "; ".join(merged[r]),
                               segment_label(r, rows_to_armhole_end, neck_start_row, shoulder_start_row)))
            prev = r + 1
        if prev <= rows_count:
            seg = segment_label(prev, rows_to_armhole_end, neck_start_row, shoulder_start_row)
            table_rows.append((str(prev) if prev==rows_count else f"{prev}-{rows_count}", "Прямо", seg))
    df = pd.DataFrame(table_rows, columns=["Ряды", "Действия", "Сегмент"])
    st.dataframe(df, use_container_width=True, hide_index=True)
    if key: st.session_state[key] = table_rows

# -----------------------------
# Ввод
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

method = st.selectbox(
    "Метод убавок",
    ["Стандартные (со стороны каретки)", "Частичное вязание (поворотные ряды)"],
    index=0
)

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
# Кнопка «Рассчитать»
# -----------------------------
if st.button("🔄 Рассчитать"):
    fields = [density_st_str, density_row_str, hip_cm_str, chest_cm_str, length_cm_str,
              armhole_depth_cm_str, neck_width_cm_str, neck_depth_cm_str, neck_depth_back_cm_str,
              shoulder_len_cm_str, shoulder_slope_cm_str]
    if not all(fields):
        st.error("⚠️ Заполни все поля.")
        st.stop()
    try:
        (density_st, density_row, hip_cm, chest_cm, length_cm,
         armhole_depth_cm, neck_width_cm, neck_depth_cm, neck_depth_back_cm,
         shoulder_len_cm, shoulder_slope_cm) = parse_inputs()
    except:
        st.error("⚠️ Только числа (точка/запятая).")
        st.stop()

    # Пересчёт
    st_hip     = cm_to_st(hip_cm, density_st)
    if st_hip % 2: st_hip += 1  # правило 1
    st_chest   = cm_to_st(chest_cm, density_st)

    rows_total = cm_to_rows(length_cm, density_row)
    rows_armh  = cm_to_rows(armhole_depth_cm, density_row)

    neck_st          = cm_to_st(neck_width_cm, density_st)
    neck_rows_front  = cm_to_rows(neck_depth_cm, density_row)
    neck_rows_back   = cm_to_rows(neck_depth_back_cm, density_row)

    st_shldr   = cm_to_st(shoulder_len_cm, density_st)  # убавок по плечу на КАЖДОМ плече
    rows_slope = cm_to_rows(shoulder_slope_cm, density_row)

    st_shoulders = 2*st_shldr + neck_st
    rows_bottom  = rows_total - rows_armh - rows_slope
    armhole_start_row  = rows_bottom + 1
    shoulder_start_row = rows_total - rows_slope + 1
    armhole_end_row    = shoulder_start_row - 1

    neck_start_row_front = rows_total - neck_rows_front + 1
    neck_start_row_back  = rows_total - neck_rows_back + 1

    # Сводка
    st.subheader("📊 Сводка")
    st.write(f"- Набрать петель: **{st_hip}** (приведено к чётному)")
    st.write(f"- Всего рядов: **{rows_total}**")

    # -------- ПЕРЕД --------
    st.subheader("📋 Инструкция для переда")
    actions = []

    # Низ: симметрично от низа до начала проймы
    delta_bottom = st_chest - st_hip
    if delta_bottom > 0:
        actions += sym_increases(delta_bottom, 1, rows_bottom, rows_total, "бок")
    elif delta_bottom < 0:
        actions += sym_decreases(-delta_bottom, 1, rows_bottom, rows_total, "бок")

    # Пройма (до плеч)
    actions += calc_round_armhole(st_chest, st_shoulders, armhole_start_row, shoulder_start_row, rows_total)

    # Горловина + плечи (единый план, НЕТ плеча в split_row)
    actions += plan_front_or_back(
        neck_st=neck_st,
        neck_rows=neck_rows_front,
        neck_start_row=neck_start_row_front,
        st_shldr=st_shldr,
        rows_slope=rows_slope,
        rows_total=rows_total,
        forbid_first_row_shoulder=True,
        straight_percent=0.20
    )

    # Слияние + сторона каретки
    actions = merge_actions(actions, rows_total)
    actions = fix_carriage_side(actions, method)

    # Таблица (с разбиением)
    make_table_front_split(actions, rows_total, rows_bottom, neck_start_row_front, shoulder_start_row, key="table_front")

    # -------- СПИНКА --------
    st.subheader("📋 Инструкция для спинки")
    actions_back = []

    # Низ
    delta_bottom = st_chest - st_hip
    if delta_bottom > 0:
        actions_back += sym_increases(delta_bottom, 1, rows_bottom, rows_total, "бок")
    elif delta_bottom < 0:
        actions_back += sym_decreases(-delta_bottom, 1, rows_bottom, rows_total, "бок")

    # Пройма
    delta_armh = st_shoulders - st_chest
    if delta_armh > 0:
        actions_back += sym_increases(delta_armh, armhole_start_row, armhole_end_row, rows_total, "пройма")
    elif delta_armh < 0:
        actions_back += sym_decreases(-delta_armh, armhole_start_row, armhole_end_row, rows_total, "пройма")

    # Горловина спинки + плечи (те же правила)
    actions_back += plan_front_or_back(
        neck_st=neck_st,
        neck_rows=neck_rows_back,
        neck_start_row=neck_start_row_back,
        st_shldr=st_shldr,
        rows_slope=rows_slope,
        rows_total=rows_total,
        forbid_first_row_shoulder=True,
        straight_percent=0.20
    )

    actions_back = merge_actions(actions_back, rows_total)
    actions_back = fix_carriage_side(actions_back, method)
    make_table_back_split(actions_back, rows_total, rows_bottom, neck_start_row_back, shoulder_start_row, key="table_back")

    # сохранить для PDF
    st.session_state.actions = actions
    st.session_state.actions_back = actions_back
    st.session_state.st_hip = st_hip
    st.session_state.rows_total = rows_total
    st.session_state.rows_bottom = rows_bottom

# -----------------------------
# PDF
# -----------------------------
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import io

pdfmetrics.registerFont(TTFont("DejaVuSans", "DejaVuSans.ttf"))

if st.session_state.actions and st.session_state.actions_back:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()
    styles["Normal"].fontName = "DejaVuSans"
    styles["Heading1"].fontName = "DejaVuSans"
    styles["Heading2"].fontName = "DejaVuSans"

    elements.append(Paragraph("🧶 Интерактивное вязание — инструкция", styles['Heading1']))
    elements.append(Spacer(1, 12))

    summary_data = [
        ["Набрать петель (чётно)", str(st.session_state.st_hip)],
        ["Всего рядов", str(st.session_state.rows_total)],
        ["Низ (до проймы)", str(st.session_state.rows_bottom)]
    ]
    tbl = Table(summary_data, hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), "DejaVuSans"),
        ("FONTSIZE", (0,0), (-1,-1), 10),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
    ]))
    elements.append(tbl)
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Инструкция для переда", styles['Heading2']))
    front = st.session_state.get("table_front", [["—","Нет данных","—"]])
    tbl_front = Table(front, hAlign="LEFT")
    tbl_front.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), "DejaVuSans"),
        ("FONTSIZE", (0,0), (-1,-1), 10),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
    ]))
    elements.append(tbl_front)
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Инструкция для спинки", styles['Heading2']))
    back = st.session_state.get("table_back", [["—","Нет данных","—"]])
    tbl_back = Table(back, hAlign="LEFT")
    tbl_back.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), "DejaVuSans"),
        ("FONTSIZE", (0,0), (-1,-1), 10),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
    ]))
    elements.append(tbl_back)

    doc.build(elements)
    buffer.seek(0)
    st.download_button("📥 Скачать PDF", buffer, file_name="vyazanie_instructions.pdf", mime="application/pdf")
else:
    st.info("Сначала нажми «🔄 Рассчитать».")
