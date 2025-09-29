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
    st.session_state.table_front = []
    st.session_state.table_back = []

# -----------------------------
# Утилиты
# -----------------------------
def cm_to_st(cm, dens_st):  return int(round((cm/10.0)*dens_st))
def cm_to_rows(cm, dens_r): return int(round((cm/10.0)*dens_r))

def split_total_into_steps(total: int, steps: int):
    if total <= 0 or steps <= 0: return []
    base = total // steps
    rem  = total % steps
    return [base + (1 if i < rem else 0) for i in range(steps)]

def allowed_rows_any(start_row, end_row, rows_total, min_row=6, cut_last_two=True):
    if end_row is None: end_row = rows_total
    high = min(end_row, rows_total - 2) if cut_last_two else end_row
    low  = max(min_row, start_row)
    return list(range(low, high+1)) if low <= high else []

def allowed_even_rows(start_row, end_row, rows_total, min_row=6, cut_last_two=True):
    rows = allowed_rows_any(start_row, end_row, rows_total, min_row, cut_last_two)
    return [r for r in rows if r % 2 == 0]

def allowed_odd_rows(start_row, end_row, rows_total, min_row=6, cut_last_two=True):
    rows = allowed_rows_any(start_row, end_row, rows_total, min_row, cut_last_two)
    return [r for r in rows if r % 2 == 1]

# симметричные операции (по бокам/пройме и т.п.) — выполняются в одном ряду
def sym_increases(total_add, start_row, end_row, rows_total, label):
    if total_add <= 0: return []
    if total_add % 2: total_add += 1
    # разрешены любые ряды (а факт стороны потом поправит fix_carriage_side)
    rows = allowed_rows_any(max(6,start_row), end_row, rows_total)
    if not rows: return []
    per_side = total_add // 2
    steps = min(len(rows), per_side)
    idxs  = np.linspace(0, len(rows)-1, num=steps, dtype=int)
    chosen= [rows[i] for i in idxs]
    parts = split_total_into_steps(per_side, steps)
    out=[]
    for r,v in zip(chosen,parts):
        out.append((r, f"+{v} п. {label} (справа)"))
        out.append((r, f"+{v} п. {label} (слева)"))
    return out

def sym_decreases(total_sub, start_row, end_row, rows_total, label):
    if total_sub <= 0: return []
    if total_sub % 2: total_sub += 1
    rows = allowed_rows_any(max(6,start_row), end_row, rows_total)
    if not rows: return []
    per_side = total_sub // 2
    steps = min(len(rows), per_side)
    idxs  = np.linspace(0, len(rows)-1, num=steps, dtype=int)
    chosen= [rows[i] for i in idxs]
    parts = split_total_into_steps(per_side, steps)
    out=[]
    for r,v in zip(chosen,parts):
        out.append((r, f"-{v} п. {label} (справа)"))
        out.append((r, f"-{v} п. {label} (слева)"))
    return out

# -----------------------------
# Пройма (скруглённая, 3 зоны)
# -----------------------------
def calc_round_armhole(st_chest, st_shoulders, start_row, shoulder_start_row, rows_total,
                       depth_percent=0.05, hold_percent=0.10):
    if shoulder_start_row <= start_row: return []
    end_row = shoulder_start_row - 1
    total_rows = end_row - start_row + 1
    if total_rows <= 0: return []

    depth_armhole_st = int(round(st_chest * depth_percent))
    st_mid = st_chest - depth_armhole_st

    rows_smooth = int(round(total_rows * 0.40))
    rows_hold   = int(round(total_rows * hold_percent))
    if rows_smooth + rows_hold > total_rows:
        rows_smooth = max(0, total_rows - rows_hold)
    rows_rest = max(0, total_rows - rows_smooth - rows_hold)

    actions=[]

    # 1) внутрь
    delta1 = st_mid - st_chest
    if delta1 < 0:
        actions += sym_decreases(-delta1, start_row, start_row+rows_smooth-1, rows_total, "пройма")
    # 2) прямо — ничего не пишем
    # 3) наружу
    delta2 = st_shoulders - st_mid
    if delta2 > 0:
        actions += sym_increases(delta2, start_row+rows_smooth+rows_hold, end_row, rows_total, "пройма")

    return actions

# -----------------------------
# Горловина (круглая)
# -----------------------------
def plan_neck_and_shoulders_split(neck_st, neck_rows, neck_start_row,
                                  st_shldr, rows_slope, rows_total,
                                  straight_percent=0.20):
    """
    Горловина + плечи с разделением:
      - 1-й ряд горловины: ТОЛЬКО горловина (без плеча);
      - верхние straight_percent глубины горловины — прямо;
      - ЛЕВОЕ плечо: плечо в нечётных, горловина в чётных;
        ПРАВОЕ плечо: плечо в чётных, горловина в нечётных;
      - плечо уходит в ноль на последнем рабочем ряду (rows_total-1).
    """
    actions = []

    if neck_st <= 0 or neck_rows <= 0 or st_shldr <= 0: 
        return actions

    # 1) центральное закрытие (60%, чётное)
    first_dec = int(round(neck_st * 0.60))
    if first_dec % 2: first_dec += 1
    if first_dec > neck_st: first_dec = neck_st - (neck_st % 2)
    rest = max(0, neck_st - first_dec)

    # границы горловины
    straight_rows = max(2, int(round(neck_rows * straight_percent)))
    neck_last_dec_row = min(neck_start_row + neck_rows - 1 - straight_rows, rows_total - 2)
    if neck_last_dec_row < neck_start_row: neck_last_dec_row = neck_start_row

    # 1-й ряд горловины — только горло
    central_row = max(6, min(neck_start_row, rows_total-2))
    actions.append((central_row, f"-{first_dec} п. горловина (центр, разделение на плечи)"))

    # Сколько петель на каждом плече прямо ПОСЛЕ центрального закрытия (ещё до дальнейших убавок горла):
    remain_neck = neck_st - first_dec
    left_after_split  = st_shldr + remain_neck//2
    right_after_split = st_shldr + (remain_neck - remain_neck//2)
    # вставим служебную строку для таблицы (заголовок потом сформируем в таблице)
    actions.append((central_row, f"[SPLIT_INFO] Левое плечо: {left_after_split} п., Правое плечо: {right_after_split} п."))

    # 2) Остальное горло — распределяем по чёт/нечёт, согласно плечам
    if rest > 0 and neck_last_dec_row > central_row:
        left_rows_for_neck  = allowed_even_rows(central_row+1, neck_last_dec_row, rows_total)  # левое горло — чётные
        right_rows_for_neck = allowed_odd_rows (central_row+1, neck_last_dec_row, rows_total)  # правое горло — нечётные

        # чередуем убавки поровну между сторонами
        seq = []
        li, ri = 0, 0
        while len(seq) < rest and (li < len(left_rows_for_neck) or ri < len(right_rows_for_neck)):
            if li < len(left_rows_for_neck):
                seq.append( ("L", left_rows_for_neck[li]) ); li += 1
            if len(seq) >= rest: break
            if ri < len(right_rows_for_neck):
                seq.append( ("R", right_rows_for_neck[ri]) ); ri += 1

        for side, row in seq:
            if side == "L":
                actions.append((row, "-1 п. горловина [L]"))
            else:
                actions.append((row, "-1 п. горловина [R]"))

    # 3) Скос плеча — уходит в ноль к rows_total-1.
    shoulder_start_row = rows_total - rows_slope + 1
    last_action_row    = rows_total - 1

    # ЛЕВОЕ плечо: нечётные
    left_rows_sh = allowed_odd_rows(shoulder_start_row, last_action_row, rows_total)
    # ПРАВОЕ плечо: чётные
    right_rows_sh = allowed_even_rows(shoulder_start_row, last_action_row, rows_total)

    # распределим весь объём плеча (стежков на одно плечо) по его рядам
    parts_left  = split_total_into_steps(st_shldr, max(1, len(left_rows_sh)))
    parts_right = split_total_into_steps(st_shldr, max(1, len(right_rows_sh)))

    # ВАЖНО: первый ряд горла (central_row) — без плечевых убавок. Мы уже начинаем плечо с shoulder_start_row, так что конфликта нет.
    for r, v in zip(left_rows_sh, parts_left):
        actions.append((r, f"-{v} п. скос плеча [L]"))
    for r, v in zip(right_rows_sh, parts_right):
        actions.append((r, f"-{v} п. скос плеча [R]"))

    return actions

# -----------------------------
# Слияние (минимальное) — убираем дубли, разводим «горло+плечо» если вдруг совпали
# -----------------------------
def merge_actions(actions, rows_total):
    grouped = defaultdict(list)
    for row, note in actions:
        if isinstance(row, int) and 1 <= row <= rows_total:
            grouped[row].append(note)

    # уникализируем
    for r in grouped:
        seen=set(); uniq=[]
        for n in grouped[r]:
            if n not in seen:
                uniq.append(n); seen.add(n)
        grouped[r]=uniq

    # запрещаем совпадение горло+плечо в одном ряду, кроме центрального (где мы и так не ставим плечо)
    out=[]
    for r in sorted(grouped):
        notes = grouped[r]
        neck = [n for n in notes if "горловина" in n.lower()]
        sh   = [n for n in notes if "скос плеча" in n.lower()]
        other= [n for n in notes if n not in neck and n not in sh]
        if neck and sh:
            # переносим плечо на соседний допустимый ряд согласно его [L]/[R] и чётности
            for n in neck: out.append((r,n))
            for n in other: out.append((r,n))
            for s in sh:
                is_left  = "[l]" in s.lower()
                is_right = "[r]" in s.lower()
                target = r
                # подвинем на +1 ряд нужной чётности
                if is_left:
                    target = r+1 if (r+1)%2==1 else r+2
                elif is_right:
                    target = r+1 if (r+1)%2==0 else r+2
                # не залезать в последние 2 ряда закрытия
                if target >= rows_total: target = rows_total-1
                out.append((target, s))
        else:
            for n in notes: out.append((r,n))
    # ещё раз сгруппируем и склеим
    merged=defaultdict(list)
    for r,n in out: merged[r].append(n)
    result=[]
    for r in sorted(merged):
        result.append((r, "; ".join(merged[r])))
    return result

# -----------------------------
# Учёт стороны каретки
# -----------------------------
def fix_carriage_side(actions, method=None):
    if method is None:
        method = st.session_state.get("method","Стандартные (со стороны каретки)")
    use_std = method.startswith("Стандартные")
    fixed=[]
    for r, note in actions:
        nl = note.lower()
        # ряды: нечётный — каретка справа; чётный — слева
        correct_side = ("справа" if r%2==1 else "слева") if use_std else ("слева" if r%2==1 else "справа")
        if (("справа" in nl) or ("слева" in nl)) and (correct_side not in nl):
            new_r = r-1 if r>1 else r+1
            fixed.append((new_r, note))
        else:
            fixed.append((r, note))
    return fixed

# -----------------------------
# Сегменты
# -----------------------------
def tag_segment(row, rows_to_armhole_end, neck_start_row, shoulder_start_row):
    tags=[]
    if row <= rows_to_armhole_end: tags.append("Низ изделия")
    if rows_to_armhole_end < row < shoulder_start_row: tags.append("Пройма")
    if neck_start_row and row >= neck_start_row: tags.append("Горловина")
    if shoulder_start_row and row >= shoulder_start_row: tags.append("Скос плеча")
    return " + ".join(tags) if tags else "—"

# -----------------------------
# Таблицы (общая + фронт/бэк с разделением)
# -----------------------------
def make_table_full(actions, rows_count, rows_to_armhole_end, neck_start_row, shoulder_start_row, key=None):
    merged = defaultdict(list)
    for r,n in actions:
        if 1<=r<=rows_count: merged[r].append(n)
    rows_sorted = sorted(merged)
    table=[]
    prev=1
    if not rows_sorted:
        table.append((f"1-{rows_count}", "Прямо", tag_segment(1,rows_to_armhole_end,neck_start_row,shoulder_start_row)))
    else:
        for r in rows_sorted:
            if r>prev:
                seg = tag_segment(prev, rows_to_armhole_end, neck_start_row, shoulder_start_row)
                table.append((str(prev) if prev==r-1 else f"{prev}-{r-1}", "Прямо", seg))
            table.append((str(r), "; ".join(merged[r]),
                          tag_segment(r, rows_to_armhole_end, neck_start_row, shoulder_start_row)))
            prev=r+1
        if prev<=rows_count:
            seg = tag_segment(prev, rows_to_armhole_end, neck_start_row, shoulder_start_row)
            table.append((str(prev) if prev==rows_count else f"{prev}-{rows_count}", "Прямо", seg))
    df = pd.DataFrame(table, columns=["Ряды","Действия","Сегмент"])
    st.dataframe(df, use_container_width=True, hide_index=True)
    if key: st.session_state[key]=table

def _make_table_split(actions, rows_count, rows_to_armhole_end, neck_start_row, shoulder_start_row, key=None, title_left="— ЛЕВОЕ ПЛЕЧО —", title_right="— ПРАВОЕ ПЛЕЧО —"):
    merged = defaultdict(list)
    for r,n in actions:
        if 1<=r<=rows_count: merged[r].append(n)
    if not merged:
        make_table_full(actions, rows_count, rows_to_armhole_end, neck_start_row, shoulder_start_row, key=key)
        return
    # поищем split
    rows_sorted = sorted(merged)
    split_row=None
    for r in rows_sorted:
        if any("разделение на плечи" in n.lower() for n in merged[r]):
            split_row=r; break
    if split_row is None:
        make_table_full(actions, rows_count, rows_to_armhole_end, neck_start_row, shoulder_start_row, key=key)
        return
    def clean(text): return text.replace("[L]","").replace("[R]","").strip()
    def push_plain(tbl,a,b):
        if a>b: return
        seg = tag_segment(a, rows_to_armhole_end, neck_start_row, shoulder_start_row)
        tbl.append((str(a) if a==b else f"{a}-{b}", "Прямо", seg))

    table=[]
    # 1) до split
    prev=1
    for r in [x for x in rows_sorted if x<split_row]:
        if r>prev: push_plain(table, prev, r-1)
        table.append((str(r), "; ".join(clean(n) for n in merged[r]), tag_segment(r,rows_to_armhole_end,neck_start_row,shoulder_start_row)))
        prev=r+1
    if prev<=split_row-1: push_plain(table, prev, split_row-1)
    # сам split — показываем только горловину и инфо
    split_notes=[n for n in merged[split_row] if ("горловина" in n.lower()) or ("[split_info]" in n.lower())]
    if split_notes:
        table.append((str(split_row), "; ".join(clean(n) for n in split_notes),
                      tag_segment(split_row,rows_to_armhole_end,neck_start_row,shoulder_start_row)))
    # 2) левое плечо
    table.append((title_left, "", ""))
    left_rows=[]; prev=split_row+1
    for r in [x for x in rows_sorted if x>split_row]:
        notes=[]
        for n in merged[r]:
            ln=n.lower()
            if ("скос плеча" in ln and "[l]" in ln) or ("горловина" in ln and "[r]" not in ln):
                notes.append(n)
        if notes: left_rows.append((r,notes))
    for r,notes in left_rows:
        if r>prev: push_plain(table, prev, r-1)
        table.append((str(r), "; ".join(clean(n) for n in notes), tag_segment(r,rows_to_armhole_end,neck_start_row,shoulder_start_row)))
        prev=r+1
    if prev<=rows_count: push_plain(table, prev, rows_count)
    # 3) правое плечо (с возвращением к split)
    table.append((f"{title_right} (вернитесь к ряду {split_row})", "", ""))
    prev=split_row
    right_rows=[]
    cand=[split_row]+[x for x in rows_sorted if x>split_row]
    for r in cand:
        notes=[]
        for n in merged.get(r,[]):
            ln=n.lower()
            if ("скос плеча" in ln and "[r]" in ln) or ("горловина" in ln and "[l]" not in ln):
                notes.append(n)
        if r==split_row and any("разделение на плечи" in n.lower() for n in merged[split_row]):
            notes.append("↳ переход к правому плечу")
        if notes: right_rows.append((r,notes))
    for r,notes in right_rows:
        if r>prev: push_plain(table, prev, r-1)
        table.append((str(r), "; ".join(clean(n) for n in notes), tag_segment(r,rows_to_armhole_end,neck_start_row,shoulder_start_row)))
        prev=r+1
    if prev<=rows_count: push_plain(table, prev, rows_count)

    df=pd.DataFrame(table, columns=["Ряды","Действия","Сегмент"])
    st.dataframe(df, use_container_width=True, hide_index=True)
    if key: st.session_state[key]=table

def make_table_front_split(actions, rows_count, rows_to_armhole_end, neck_start_row, shoulder_start_row, key=None):
    _make_table_split(actions, rows_count, rows_to_armhole_end, neck_start_row, shoulder_start_row, key, "— ЛЕВОЕ ПЛЕЧО —", "— ПРАВОЕ ПЛЕЧО —")

def make_table_back_split(actions, rows_count, rows_to_armhole_end, neck_start_row, shoulder_start_row, key=None):
    _make_table_split(actions, rows_count, rows_to_armhole_end, neck_start_row, shoulder_start_row, key, "— ЛЕВОЕ ПЛЕЧО —", "— ПРАВОЕ ПЛЕЧО —")

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
    inputs = [density_st_str, density_row_str, hip_cm_str, chest_cm_str, length_cm_str,
              armhole_depth_cm_str, neck_width_cm_str, neck_depth_cm_str, neck_depth_back_cm_str,
              shoulder_len_cm_str, shoulder_slope_cm_str]
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

    # Пересчёты
    st_hip     = cm_to_st(hip_cm, density_st)
    if st_hip % 2: st_hip += 1  # чётный набор
    st_chest   = cm_to_st(chest_cm, density_st)
    rows_total = cm_to_rows(length_cm, density_row)
    rows_armh  = cm_to_rows(armhole_depth_cm, density_row)

    neck_st    = cm_to_st(neck_width_cm, density_st)
    neck_rows_front  = cm_to_rows(neck_depth_cm, density_row)
    neck_rows_back   = cm_to_rows(neck_depth_back_cm, density_row)

    st_shldr   = cm_to_st(shoulder_len_cm, density_st)  # на одно плечо
    rows_slope = cm_to_rows(shoulder_slope_cm, density_row)

    st_shoulders = 2*st_shldr + neck_st
    rows_bottom  = rows_total - rows_armh - rows_slope
    if rows_bottom < 0: rows_bottom = 0

    armhole_start_row   = rows_bottom + 1
    shoulder_start_row  = rows_total - rows_slope + 1
    armhole_end_row     = shoulder_start_row - 1
    rows_to_armhole_end = rows_bottom
    last_action_row     = rows_total - 1

    neck_start_row_front = rows_total - neck_rows_front + 1
    neck_start_row_back  = rows_total - neck_rows_back + 1

    # -----------------------------
    # Сводка
    # -----------------------------
    st.subheader("📊 Сводка")
    st.write(f"- Набрать петель: **{st_hip}**")
    st.write(f"- Всего рядов: **{rows_total}**")

    # -----------------------------
    # ПЕРЕД
    # -----------------------------
    st.subheader("📋 Инструкция для переда")
    actions = []

    # Низ: не раньше 6-го ряда
    delta_bottom = st_chest - st_hip
    if delta_bottom > 0:
        actions += sym_increases(delta_bottom, 6, rows_bottom, rows_total, "бок")
    elif delta_bottom < 0:
        actions += sym_decreases(-delta_bottom, 6, rows_bottom, rows_total, "бок")

    # Пройма
    actions += calc_round_armhole(st_chest, st_shoulders, armhole_start_row, shoulder_start_row, rows_total)

    # Горловина + плечи
    actions += plan_neck_and_shoulders_split(
        neck_st=neck_st,
        neck_rows=neck_rows_front,
        neck_start_row=neck_start_row_front,
        st_shldr=st_shldr,
        rows_slope=rows_slope,
        rows_total=rows_total,
        straight_percent=0.20
    )

    actions = merge_actions(actions, rows_total)
    actions = fix_carriage_side(actions, method)

    make_table_front_split(actions, rows_total, rows_bottom, neck_start_row_front, shoulder_start_row, key="table_front")

    # -----------------------------
    # СПИНКА
    # -----------------------------
    st.subheader("📋 Инструкция для спинки")
    actions_back = []

    # Низ
    delta_bottom = st_chest - st_hip
    if delta_bottom > 0:
        actions_back += sym_increases(delta_bottom, 6, rows_bottom, rows_total, "бок")
    elif delta_bottom < 0:
        actions_back += sym_decreases(-delta_bottom, 6, rows_bottom, rows_total, "бок")

    # Пройма (как на переде)
    actions_back += calc_round_armhole(st_chest, st_shoulders, armhole_start_row, shoulder_start_row, rows_total)

    # Горловина + плечи (спинка)
    actions_back += plan_neck_and_shoulders_split(
        neck_st=neck_st,
        neck_rows=neck_rows_back,
        neck_start_row=neck_start_row_back,
        st_shldr=st_shldr,
        rows_slope=rows_slope,
        rows_total=rows_total,
        straight_percent=0.20
    )

    actions_back = merge_actions(actions_back, rows_total)
    actions_back = fix_carriage_side(actions_back, method)

    make_table_back_split(actions_back, rows_total, rows_bottom, neck_start_row_back, shoulder_start_row, key="table_back")

    # сохранить для PDF (если нужно)
    st.session_state.actions = actions
    st.session_state.actions_back = actions_back
    st.session_state.st_hip = st_hip
    st.session_state.rows_total = rows_total
    st.session_state.rows_bottom = rows_bottom
