def distribute_side_increases(start_row, end_row, total_delta, label):
    """
    Прибавки симметричные: выполняются парой в одном ряду.
    Например: +1 слева и +1 справа → всё в один ряд.
    """
    pairs = total_delta // 2
    rows = spread_rows(max(2, start_row), end_row, pairs)
    out = []
    for r in rows:
        out.append((r, f"+1 п. {label} слева и +1 п. {label} справа"))
    # если нечётное число петель — последнюю прибавку в отдельном ряду
    if total_delta % 2 == 1 and rows:
        out.append((rows[-1] + 2, f"+1 п. {label} (доп.)"))
    return out


def calc_round_neckline_by_percent(total_stitches, total_rows, start_row, percentages=(60,20,10,5,5)):
    """
    Горловина: симметричные убавки = разные ряды.
    Например, -4 п. → ряд 120 слева, ряд 121 справа.
    """
    if total_stitches <= 0 or total_rows <= 0:
        return []
    parts = [int(round(total_stitches * p / 100.0)) for p in percentages]
    diff = total_stitches - sum(parts)
    if diff != 0:
        parts[0] += diff  # корректируем первую группу

    actions = []
    row = max(2, start_row)
    for dec in parts:
        if dec > 0 and row <= start_row + total_rows - 1:
            # сначала левая
            actions.append((row, f"-{dec} п. горловина (левая)"))
            # потом правая в следующем ряду
            actions.append((row + 1, f"-{dec} п. горловина (правая)"))
        row += 2
    return actions


def slope_shoulder_steps(total_stitches, start_row, end_row, steps=3):
    """
    Скос плеча: убавки симметричные → разные ряды (лево/право).
    """
    if total_stitches <= 0 or end_row < start_row or steps <= 0:
        return []
    base = total_stitches // steps
    parts = [base]*steps
    parts[-1] += (total_stitches - base*steps)

    rows = spread_rows(max(2, start_row), end_row, steps)
    out = []
    for r, p in zip(rows, parts):
        out.append((r, f"закрыть {p} п. плечо (левое)"))
        out.append((r + 1, f"закрыть {p} п. плечо (правое)"))
    return out
