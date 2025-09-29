# -----------------------------
# Таблица переда с разделением на плечи
# -----------------------------
def make_table_front_split(actions, rows_count, rows_to_armhole_end, neck_start_row, shoulder_start_row, key=None):
    """Строим таблицу переда с отдельными блоками плеч."""
    merged = defaultdict(list)
    for row, note in actions:
        if isinstance(row, int) and 1 <= row <= rows_count:
            merged[row].append(note)

    if not merged:
        make_table_full(actions, rows_count, rows_to_armhole_end, neck_start_row, shoulder_start_row, key=key)
        return

    def make_table_front_split(actions, rows_count, rows_to_armhole_end, neck_start_row, shoulder_start_row, key=None):
    st.write("⚡ DEBUG: Запуск make_table_front_split")
    st.write("Всего действий:", len(actions))

    merged = defaultdict(list)
    for row, note in actions:
        if isinstance(row, int) and 1 <= row <= rows_count:
            merged[row].append(note)

    if not merged:
        st.write("⚠️ merged пуст → вызываем make_table_full")
        make_table_full(actions, rows_count, rows_to_armhole_end, neck_start_row, shoulder_start_row, key=key)
        return

    # убираем дубли
    for row in merged:
        merged[row] = list(dict.fromkeys(merged[row]))

    rows_sorted = sorted(merged.keys())

    # ряд разделения (центральное закрытие горловины)
    split_row = None
    for r in rows_sorted:
        if any("разделение на плечи" in n.lower() for n in merged[r]):
            split_row = r
            break
    if split_row is None:
        make_table_full(actions, rows_count, rows_to_armhole_end, neck_start_row, shoulder_start_row, key=key)
        return

    def push_plain(table_rows, start, end):
        if start > end:
            return
        segment = section_tags(start, rows_to_armhole_end, neck_start_row, shoulder_start_row)
        label = str(start) if start == end else f"{start}-{end}"
        table_rows.append((label, "Прямо", segment))

    def clean_notes(notes):
        return [n.replace("[L]", "").replace("[R]", "").strip() for n in notes]

    table_rows = []
    prev = 1

    # ----- до разделения -----
    for r in [x for x in rows_sorted if x < split_row]:
        if r > prev:
            push_plain(table_rows, prev, r - 1)
        table_rows.append((str(r), "; ".join(clean_notes(merged[r])),
                           section_tags(r, rows_to_armhole_end, neck_start_row, shoulder_start_row)))
        prev = r + 1
    if prev <= split_row - 1:
        push_plain(table_rows, prev, split_row - 1)

    # сам split_row → только горловина
    split_notes = [n for n in merged[split_row] if "горловина" in n.lower()]
    if split_notes:
        table_rows.append((str(split_row), "; ".join(clean_notes(split_notes)),
                           "Горловина"))

    # ----- ЛЕВОЕ ПЛЕЧО -----
    table_rows.append(("— ЛЕВОЕ ПЛЕЧО —", "", ""))
    left_prev = split_row + 1
    for r in [x for x in rows_sorted if x > split_row]:
        notes = [n for n in merged[r] if "[L]" in n]
        if notes:
            if r > left_prev:
                push_plain(table_rows, left_prev, r - 1)
            table_rows.append((str(r), "; ".join(clean_notes(notes)),
                               section_tags(r, rows_to_armhole_end, neck_start_row, shoulder_start_row)))
            left_prev = r + 1
    if left_prev <= rows_count:
        push_plain(table_rows, left_prev, rows_count)

    # ----- ПРАВОЕ ПЛЕЧО -----
    table_rows.append((f"— ПРАВОЕ ПЛЕЧО — (вернитесь к ряду {split_row})", "", ""))
    right_prev = split_row + 1
    for r in [x for x in rows_sorted if x >= split_row]:
        notes = [n for n in merged[r] if "[R]" in n]
        if notes:
            if r > right_prev:
                push_plain(table_rows, right_prev, r - 1)
            table_rows.append((str(r), "; ".join(clean_notes(notes)),
                               section_tags(r, rows_to_armhole_end, neck_start_row, shoulder_start_row)))
            right_prev = r + 1
    if right_prev <= rows_count:
        push_plain(table_rows, right_prev, rows_count)

    df = pd.DataFrame(table_rows, columns=["Ряды", "Действия", "Сегмент"])
    st.dataframe(df, use_container_width=True, hide_index=True)
    if key:
        st.session_state[key] = table_rows


# -----------------------------
# Таблица спинки с разделением на плечи
# -----------------------------
def make_table_back_split(actions, rows_count, rows_to_armhole_end, neck_start_row, shoulder_start_row, key=None):
    """Строим таблицу спинки с отдельными блоками плеч."""
    merged = defaultdict(list)
    for row, note in actions:
        if isinstance(row, int) and 1 <= row <= rows_count:
            merged[row].append(note)

    if not merged:
        make_table_full(actions, rows_count, rows_to_armhole_end, neck_start_row, shoulder_start_row, key=key)
        return

    for row in merged:
        merged[row] = list(dict.fromkeys(merged[row]))

    rows_sorted = sorted(merged.keys())

    split_row = None
    for r in rows_sorted:
        if any("разделение на плечи" in n.lower() for n in merged[r]):
            split_row = r
            break
    if split_row is None:
        make_table_full(actions, rows_count, rows_to_armhole_end, neck_start_row, shoulder_start_row, key=key)
        return

    def push_plain(table_rows, start, end):
        if start > end:
            return
        segment = section_tags(start, rows_to_armhole_end, neck_start_row, shoulder_start_row)
        label = str(start) if start == end else f"{start}-{end}"
        table_rows.append((label, "Прямо", segment))

    def clean_notes(notes):
        return [n.replace("[L]", "").replace("[R]", "").strip() for n in notes]

    table_rows = []
    prev = 1

    # ----- до разделения -----
    for r in [x for x in rows_sorted if x < split_row]:
        if r > prev:
            push_plain(table_rows, prev, r - 1)
        table_rows.append((str(r), "; ".join(clean_notes(merged[r])),
                           section_tags(r, rows_to_armhole_end, neck_start_row, shoulder_start_row)))
        prev = r + 1
    if prev <= split_row - 1:
        push_plain(table_rows, prev, split_row - 1)

    split_notes = [n for n in merged[split_row] if "горловина" in n.lower()]
    if split_notes:
        table_rows.append((str(split_row), "; ".join(clean_notes(split_notes)), "Горловина"))

    # ----- ЛЕВОЕ ПЛЕЧО -----
    table_rows.append(("— ЛЕВОЕ ПЛЕЧО —", "", ""))
    left_prev = split_row + 1
    for r in [x for x in rows_sorted if x > split_row]:
        notes = [n for n in merged[r] if "[L]" in n]
        if notes:
            if r > left_prev:
                push_plain(table_rows, left_prev, r - 1)
            table_rows.append((str(r), "; ".join(clean_notes(notes)),
                               section_tags(r, rows_to_armhole_end, neck_start_row, shoulder_start_row)))
            left_prev = r + 1
    if left_prev <= rows_count:
        push_plain(table_rows, left_prev, rows_count)

    # ----- ПРАВОЕ ПЛЕЧО -----
    table_rows.append((f"— ПРАВОЕ ПЛЕЧО — (вернитесь к ряду {split_row})", "", ""))
    right_prev = split_row + 1
    for r in [x for x in rows_sorted if x >= split_row]:
        notes = [n for n in merged[r] if "[R]" in n]
        if notes:
            if r > right_prev:
                push_plain(table_rows, right_prev, r - 1)
            table_rows.append((str(r), "; ".join(clean_notes(notes)),
                               section_tags(r, rows_to_armhole_end, neck_start_row, shoulder_start_row)))
            right_prev = r + 1
    if right_prev <= rows_count:
        push_plain(table_rows, right_prev, rows_count)

    df = pd.DataFrame(table_rows, columns=["Ряды", "Действия", "Сегмент"])
    st.dataframe(df, use_container_width=True, hide_index=True)
    if key:
        st.session_state[key] = table_rows
