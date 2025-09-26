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
def calc_round_neckline(total_stitches, total_rows, start_row, rows_total, straight_percent=0.05):
    if total_stitches <= 0 or total_rows <= 0:
        return []

    first_dec = int(round(total_stitches * 0.60))
    if first_dec % 2 == 1:
        first_dec += 1
    rest = total_stitches - first_dec

    straight_rows = max(2, int(round(total_rows * straight_percent)))
    neck_end_by_depth = start_row + total_rows - 1 - straight_rows
    effective_end = min(neck_end_by_depth, rows_total - 2)
