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
    return [(r, f"+{v} п. {label} (с каждой стороны)") for r, v in zip(chosen, parts)]

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
    return [(r, f"-{v} п. {label} (с каждой стороны)") for r, v in zip(chosen, parts)]


# -----------------------------
# Скос плеча
# -----------------------------
def slope_shoulder(total_stitches, start_row, end_row, rows_total):
    if total_stitches <= 0:
        return []
    # ⚡️ теперь доходит до самого конца
    rows = allowed_even_rows(start_row, end_row, rows_total, force_last=True)
    if not rows:
        return []
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
def calc_round_neckline(total_stitches, total_rows, start_row, rows_total, last_row, straight_percent=0.20):
    if total_stitches <= 0 or total_rows <= 0:
        return []

    # первый шаг = 60% и доводим до чётного
    first_dec = int(round(total_stitches * 0.60))
    if first_dec % 2 == 1:
        first_dec += 1
    rest = total_stitches - first_dec

    # прямые ряды перед концом = % от всей глубины
    straight_rows = max(2, int(round(total_rows * straight_percent)))
    neck_end_by_depth = start_row + total_rows - 1 - straight_rows

    # ⚡️ ограничиваем по last_row
    effective_end = min(neck_end_by_depth, last_row)

    rows = allowed_even_rows(start_row, effective_end, rows_total, force_last=True)
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

    # последние шаги делаем по 1 петле (если надо)
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


def make_table_full(actions, rows_total, rows_to_armhole_end, neck_start_row, shoulder_start_row, last_row, key):
    merged = defaultdict(list)
    for row, note in actions:
        merged[row].append(note)

    table_rows = []
    prev = 1

    # проходим по всем рядам до конца
    for r in range(1, last_row + 1):
        if r in merged:
            # сначала закрываем прямой участок, если он был
            if prev < r:
                seg = section_tags(prev, rows_to_armhole_end, neck_start_row, shoulder_start_row)
                if prev == r - 1:
                    table_rows.append((str(prev), "Прямо", seg))
                else:
                    table_rows.append((f"{prev}-{r-1}", "Прямо", seg))
            # потом добавляем действие
            seg = section_tags(r, rows_to_armhole_end, neck_start_row, shoulder_start_row)
            table_rows.append((str(r), "; ".join(merged[r]), seg))
            prev = r + 1

    # если после последнего действия остались прямые ряды
    if prev <= last_row:
        seg = section_tags(prev, rows_to_armhole_end, neck_start_row, shoulder_start_row)
        if prev == last_row:
            table_rows.append((str(prev), "Прямо", seg))
        else:
            table_rows.append((f"{prev}-{last_row}", "Прямо", seg))

    # сохраняем таблицу в session_state (для PDF)
    st.session_state[key] = table_rows

    # показываем в приложении
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
    # -----------------------------
    # Проверка заполненности всех полей
    # -----------------------------
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
        st.error("⚠️ Пожалуйста, вводите только числа (можно с точкой или запятой)")
        st.stop()

    # -----------------------------
    # Пересчёт в петли/ряды
    # -----------------------------
    st_hip     = cm_to_st(hip_cm, density_st)        # низ
    st_chest   = cm_to_st(chest_cm, density_st)      # грудь
    rows_total = cm_to_rows(length_cm, density_row)  # вся высота изделия
    rows_armh  = cm_to_rows(armhole_depth_cm, density_row)  # глубина проймы

    neck_st    = cm_to_st(neck_width_cm, density_st)
    neck_rows_front  = cm_to_rows(neck_depth_cm, density_row)
    neck_rows_back   = cm_to_rows(neck_depth_back_cm, density_row)

    st_shldr   = cm_to_st(shoulder_len_cm, density_st)
    rows_slope = cm_to_rows(shoulder_slope_cm, density_row)

    st_shoulders = 2 * st_shldr + neck_st   # скрытая ширина по плечам

    # низ = вся высота - пройма - плечо
    rows_bottom = rows_total - rows_armh - rows_slope

    # начало и конец этапов
    armhole_start_row   = rows_bottom + 1
    shoulder_start_row  = rows_total - rows_slope + 1
    armhole_end_row     = shoulder_start_row - 1

    # последний ряд изделия (окончание плеча)
    last_row = shoulder_start_row + rows_slope - 1

    # начало горловин
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

    delta_bottom = st_chest - st_hip
    if delta_bottom > 0:
        actions += sym_increases(delta_bottom, 6, rows_bottom, rows_total, "бок")
    elif delta_bottom < 0:
        actions += sym_decreases(-delta_bottom, 6, rows_bottom, rows_total, "бок")

    # пройма
    actions += calc_round_armhole(st_chest, st_shoulders, armhole_start_row, shoulder_start_row, rows_total)

    # горловина (20% прямых рядов по умолчанию)
    actions += calc_round_neckline(neck_st, neck_rows_front, neck_start_row_front, rows_total, last_row)

    # плечо
    actions += slope_shoulder(st_shldr, shoulder_start_row, last_row, rows_total)

    # ⚡️ не дать горловине и плечу совпасть в один ряд
    actions = merge_actions(actions, rows_total)

    make_table_full(actions, rows_total, rows_bottom, neck_start_row_front, shoulder_start_row, last_row, key="table_front")

    # -----------------------------
    # 📋 Спинка
    # -----------------------------
    st.subheader("📋 Инструкция для спинки")
    actions_back = []

    delta_bottom = st_chest - st_hip
    if delta_bottom > 0:
        actions_back += sym_increases(delta_bottom, 6, rows_bottom, rows_total, "бок")
    elif delta_bottom < 0:
        actions_back += sym_decreases(-delta_bottom, 6, rows_bottom, rows_total, "бок")

    # пройма
    actions_back += calc_round_armhole(st_chest, st_shoulders, armhole_start_row, shoulder_start_row, rows_total)

    # горловина спинки: прямые ряды = 2%
    actions_back += calc_round_neckline(
        neck_st, neck_rows_back, neck_start_row_back, rows_total, last_row, straight_percent=0.02
    )

    # плечо
    actions_back += slope_shoulder(st_shldr, shoulder_start_row, last_row, rows_total)

    # ⚡️ не дать горловине и плечу совпасть
    actions_back = merge_actions(actions_back, rows_total)

    make_table_full(actions_back, rows_total, rows_bottom, neck_start_row_back, shoulder_start_row, last_row, key="table_back")
    # -----------------------------
    # сохраняем результаты для PDF
    # -----------------------------
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
