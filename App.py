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

# -----------------------------
# Конвертеры
# -----------------------------
def cm_to_st(cm, dens_st):
    return int(round((cm/10.0)*dens_st))

def cm_to_rows(cm, dens_row):
    return int(round((cm/10.0)*dens_row))

# -----------------------------
# Вспомогательные функции
# -----------------------------
def allowed_even_rows(start_row: int, end_row: int, rows_total: int, force_last=False):
    if end_row is None:
        end_row = rows_total
    high = end_row if force_last else min(end_row, rows_total - 2)
    if high < 6: return []
    start = max(6, start_row)
    if start % 2 == 1: start += 1
    if high % 2 == 1: high -= 1
    return list(range(start, high + 1, 2)) if start <= high else []

def allowed_all_rows(start_row: int, end_row: int, rows_total: int):
    if end_row is None:
        end_row = rows_total
    high = min(end_row, rows_total - 2)
    if high < 6: return []
    start = max(6, start_row)
    return list(range(start, high + 1)) if start <= high else []

def split_total_into_steps(total: int, steps: int):
    if total <= 0 or steps <= 0: return []
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
# Пройма
# -----------------------------
def calc_round_armhole(st_chest, st_shoulders, start_row, shoulder_start_row, rows_total, depth_percent=0.05, hold_percent=0.1):
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
    delta1 = st_mid - st_chest
    if delta1 < 0:
        actions += sym_decreases(-delta1, start_row, start_row+rows_smooth, rows_total, "пройма")
    delta2 = st_shoulders - st_mid
    if delta2 > 0:
        actions += sym_increases(delta2, start_row+rows_smooth+rows_hold, end_row, rows_total, "пройма")
    return actions

# -----------------------------
# Горловина + плечо
# -----------------------------
def plan_neck_and_shoulder(
    neck_st: int,
    neck_rows: int,
    neck_start_row: int,
    st_shldr: int,
    rows_slope: int,
    rows_total: int,
    straight_percent: float = 0.10
):
    actions = []
    if neck_st <= 0 or neck_rows <= 0 or st_shldr <= 0:
        return actions

    # 1. Центральное закрытие
    first_dec = int(round(neck_st * 0.6))
    if first_dec % 2 == 1: first_dec += 1
    if first_dec > neck_st: first_dec = neck_st if neck_st % 2 == 0 else neck_st - 1
    rest = max(0, neck_st - first_dec)
    central_row = max(6, min(neck_start_row, rows_total-2))
    actions.append((central_row, f"-{first_dec} п. горловина (центр, разделение на плечи)"))

    # 2. Убавки горловины
    straight_rows = max(2, int(round(neck_rows * straight_percent)))
    last_neck_row = neck_start_row + neck_rows - straight_rows
    neck_rows_list = list(range(central_row+1, min(last_neck_row, rows_total-2)+1))
    left_used = right_used = 0
    if rest > 0 and neck_rows_list:
        steps = min(len(neck_rows_list), rest)
        idxs  = np.linspace(0, len(neck_rows_list)-1, num=steps, dtype=int)
        chosen = [neck_rows_list[i] for i in idxs]
        for k, r in enumerate(chosen):
            if k % 2 == 0:
                actions.append((r, "-1 п. горловина (левое плечо)"))
                left_used += 1
            else:
                actions.append((r, "-1 п. горловина (правое плечо)"))
                right_used += 1

    # 3. Скос плеча
    need_left  = max(0, st_shldr - left_used)
    need_right = max(0, st_shldr - right_used)
    start_row = rows_total - rows_slope + 1
    rows_even = allowed_even_rows(start_row, rows_total, rows_total)

    parts_left = split_total_into_steps(need_left, len(rows_even))
    for r, v in zip(rows_even, parts_left):
        actions.append((r, f"-{v} п. скос плеча (левое плечо)"))

    right_rows = [r+1 for r in rows_even if r+1 <= rows_total-2]
    parts_right = split_total_into_steps(need_right, len(right_rows))
    for r, v in zip(right_rows, parts_right):
        actions.append((r, f"-{v} п. скос плеча (правое плечо)"))

    return actions

# -----------------------------
# Слияние действий
# -----------------------------
def merge_actions(actions, rows_total):
    merged = defaultdict(list)
    for row, note in actions: merged[row].append(note)
    fixed, used_rows = [], set()
    first_neck_row = None
    for row in sorted(merged.keys()):
        if any("горловина" in n for n in merged[row]):
            first_neck_row = row
            break
    for row in sorted(merged.keys()):
        notes = merged[row]
        if ("горловина" in " ".join(notes)) and ("скос плеча" in " ".join(notes)):
            if row == first_neck_row:
                fixed.append((row, "; ".join(notes))); used_rows.add(row)
            else:
                shoulder_notes = [n for n in notes if "скос плеча" in n]
                neck_notes     = [n for n in notes if "горловина" in n]
                fixed.append((row, "; ".join(neck_notes))); used_rows.add(row)
                new_row = row + 1
                while new_row in used_rows and new_row < rows_total: new_row += 1
                for n in shoulder_notes: fixed.append((new_row, n)); used_rows.add(new_row)
        else:
            fixed.append((row, "; ".join(notes))); used_rows.add(row)
    return sorted(fixed, key=lambda x: int(str(x[0]).split('-')[0]))

# -----------------------------
# Учёт стороны каретки
# -----------------------------
def fix_carriage_side(actions, method=None):
    if method is None:
        method = st.session_state.get("method", "Стандартные (со стороны каретки)")
    use_std = method.startswith("Стандартные")
    fixed = []
    for r, note in actions:
        note_lower = note.lower()
        if r % 2 == 1: correct_side = "справа" if use_std else "слева"
        else:          correct_side = "слева" if use_std else "справа"
        if (("справа" in note_lower) or ("слева" in note_lower)) and (correct_side not in note_lower):
            new_r = r - 1 if r > 1 else r + 1
            fixed.append((new_r, note))
        else:
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
        if 1 <= row <= rows_count: merged[row].append(note)
    rows_sorted = sorted(merged.keys()); table_rows = []; prev = 1
    if not rows_sorted:
        seg = section_tags(1, rows_to_armhole_end, neck_start_row, shoulder_start_row)
        table_rows.append((f"1-{rows_count}", "Прямо", seg))
    else:
        for r in rows_sorted:
            if r > prev:
                seg = section_tags(prev, rows_to_armhole_end, neck_start_row, shoulder_start_row)
                if prev == r-1: table_rows.append((str(prev), "Прямо", seg))
                else: table_rows.append((f"{prev}-{r-1}", "Прямо", seg))
            table_rows.append((str(r), "; ".join(merged[r]), section_tags(r, rows_to_armhole_end, neck_start_row, shoulder_start_row)))
            prev = r + 1
    df = pd.DataFrame(table_rows, columns=["Ряды", "Действия", "Сегмент"])
    st.dataframe(df, use_container_width=True, hide_index=True)
    if key: st.session_state[key] = table_rows

# -----------------------------
# Ввод параметров
# -----------------------------
st.header("Перед / Спинка")
density_st_str   = st.text_input("Плотность: петли в 10 см")
density_row_str  = st.text_input("Плотность: ряды в 10 см")
hip_cm_str       = st.text_input("Ширина низа детали (см)")
chest_cm_str     = st.text_input("Ширина детали по груди (см)")
length_cm_str    = st.text_input("Длина изделия (см)")
armhole_depth_cm_str   = st.text_input("Длина проймы (см)")
neck_width_cm_str      = st.text_input("Ширина горловины (см)")
neck_depth_cm_str      = st.text_input("Глубина горловины спереди (см)")
neck_depth_back_cm_str = st.text_input("Глубина горловины спинки (см)")
shoulder_len_cm_str    = st.text_input("Длина плеча (см)")
shoulder_slope_cm_str  = st.text_input("Скос плеча (см)")

method = st.selectbox("Метод убавок", ["Стандартные (со стороны каретки)", "Частичное вязание (поворотные ряды)"], index=0)

# -----------------------------
# Кнопка расчёта
# -----------------------------
if st.button("🔄 Рассчитать"):
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
        st.error("⚠️ Введите только числа"); st.stop()

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
    shoulder_start_row = rows_total - rows_slope + 1
    neck_start_row_front = rows_total - neck_rows_front + 1
    neck_start_row_back  = rows_total - neck_rows_back + 1

    st.subheader("📊 Сводка")
    st.write(f"- Набрать петель: **{st_hip}**")
    st.write(f"- Всего рядов: **{rows_total}**")

    # Перед
    st.subheader("📋 Инструкция для переда")
    actions = []
    delta_bottom = (2*st_shldr + neck_st) - st_hip
    if delta_bottom > 0: actions += sym_increases(delta_bottom, 6, rows_bottom, rows_total, "бок")
    elif delta_bottom < 0: actions += sym_decreases(-delta_bottom, 6, rows_bottom, rows_total, "бок")
    actions += calc_round_armhole(st_chest, st_shoulders, rows_bottom+1, shoulder_start_row, rows_total)
    actions += plan_neck_and_shoulder(neck_st, neck_rows_front, neck_start_row_front, st_shldr, rows_slope, rows_total, 0.10)
    actions = merge_actions(actions, rows_total)
    actions = fix_carriage_side(actions, method)
    make_table_full(actions, rows_total, rows_bottom, neck_start_row_front, shoulder_start_row, key="table_front")

    # Спинка
    st.subheader("📋 Инструкция для спинки")
    actions_back = []
    delta_bottom = (2*st_shldr + neck_st) - st_hip
    if delta_bottom > 0: actions_back += sym_increases(delta_bottom, 6, rows_bottom, rows_total, "бок")
    elif delta_bottom < 0: actions_back += sym_decreases(-delta_bottom, 6, rows_bottom, rows_total, "бок")
    actions_back += calc_round_armhole(st_chest, st_shoulders, rows_bottom+1, shoulder_start_row, rows_total)
    actions_back += plan_neck_and_shoulder(neck_st, neck_rows_back, neck_start_row_back, st_shldr, rows_slope, rows_total, 0.10)
    actions_back = merge_actions(actions_back, rows_total)
    actions_back = fix_carriage_side(actions_back, method)
    make_table_full(actions_back, rows_total, rows_bottom, neck_start_row_back, shoulder_start_row, key="table_back")

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
    tbl_front = Table(st.session_state.table_front, hAlign="LEFT")
    tbl_front.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), "DejaVuSans"),
        ("FONTSIZE", (0,0), (-1,-1), 10),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
    ]))
    elements.append(tbl_front)
    elements.append(Spacer(1, 12))

    # Таблица спинки
    elements.append(Paragraph("Инструкция для спинки", styles['Heading2']))
    tbl_back = Table(st.session_state.table_back, hAlign="LEFT")
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
