"""
Excel Prediction Workbook Exporter
===================================
Exports predictions into an Excel workbook (.xlsx) with interactive formulas:
- Column P (Actual O/U): =IF(O{row}="","",IF(O{row}>K{row},"OVER",IF(O{row}<K{row},"UNDER","PUSH")))
- Column Q (Correct):    =IF(OR(O{row}="",P{row}=""),"",IF(P{row}="PUSH","PUSH",IF(J{row}=P{row},"Yes","No")))
- Column R (Miss Diff):  =IF(O{row}="","",IF(J{row}="OVER", IF(O{row}<K{row}, K{row}-O{row}, ""), IF(O{row}>K{row}, O{row}-K{row}, "")))
- Summary Table 1 (Columns T & U): Overall & >10 edge hit rates
- Summary Table 2 (Columns X..AA): Positional hit rates (QB, RB, WR, TE)
"""

import os
import pandas as pd


def export_predictions_xlsx(df: pd.DataFrame, xlsx_filename: str = "predictions_cv.xlsx") -> str:
    """
    Exports the predictions DataFrame to a single-sheet Excel workbook at `xlsx_filename`
    with interactive formulas and top-right KPI accuracy tables (Overall, High Conf, Med Conf,
    Low Conf, >10 edge, and positional breakdowns for QB, RB, WR, TE).
    """
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment
        from openpyxl.formatting.rule import CellIsRule

        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        xlsx_path = xlsx_filename if os.path.isabs(xlsx_filename) else os.path.join(root_dir, xlsx_filename)

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Predictions"
        ws.views.sheetView[0].showGridLines = True
        ws.freeze_panes = "A2"

        header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
        header_fill = PatternFill(start_color="000000", end_color="000000", fill_type="solid")
        summary_header_font = Font(name="Calibri", size=11, bold=True, color="000000")

        center_align = Alignment(horizontal="center", vertical="center")
        left_align = Alignment(horizontal="left", vertical="center")
        right_align = Alignment(horizontal="right", vertical="center")

        columns = [
            'Date', 'Player', 'Matchup', 'Pos', 'Stat', 'Rank',
            'Avg', 'PRED', 'Edge', 'O/U', 'Line', 'True Edge',
            'Sigma', 'Confidence Level',
            'Actual', 'Actual O/U', 'Correct', 'Miss Diff', 'Model_Architecture'
        ]

        # Write Main Headers (Row 1)
        for col_idx, col_name in enumerate(columns, start=1):
            disp_header = "Architecture" if col_name == 'Model_Architecture' else col_name
            cell = ws.cell(row=1, column=col_idx, value=disp_header)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = center_align

        # Pre-define Soft Green and Soft Red styles for Correct column
        green_fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
        green_font = Font(name="Calibri", size=11, color="006100", bold=True)
        red_fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
        red_font = Font(name="Calibri", size=11, color="9C0006", bold=True)

        # Write Data (Rows 2..N+1)
        df_clean = df.copy() if df is not None else pd.DataFrame()
        df_reset = df_clean.reset_index(drop=True)
        for r_idx, row in df_reset.iterrows():
            row_num = r_idx + 2
            ws.cell(row=row_num, column=1, value=str(row['Date']) if pd.notna(row.get('Date')) else '').alignment = center_align
            ws.cell(row=row_num, column=2, value=str(row['Player']) if pd.notna(row.get('Player')) else '').alignment = left_align
            ws.cell(row=row_num, column=3, value=str(row['Matchup']) if pd.notna(row.get('Matchup')) else '').alignment = center_align
            ws.cell(row=row_num, column=4, value=str(row['Pos']) if pd.notna(row.get('Pos')) else '').alignment = center_align
            ws.cell(row=row_num, column=5, value=str(row['Stat']) if pd.notna(row.get('Stat')) else '').alignment = left_align
            ws.cell(row=row_num, column=6, value=int(row['Rank']) if pd.notna(row.get('Rank')) else 16).alignment = center_align
            ws.cell(row=row_num, column=7, value=float(row['Avg']) if pd.notna(row.get('Avg')) else 0.0).alignment = right_align
            ws.cell(row=row_num, column=8, value=float(row['PRED']) if pd.notna(row.get('PRED')) else 0.0).alignment = right_align
            ws.cell(row=row_num, column=9, value=float(row['Edge']) if pd.notna(row.get('Edge')) else 0.0).alignment = right_align
            ws.cell(row=row_num, column=10, value=str(row['O/U']) if pd.notna(row.get('O/U')) else '').alignment = center_align
            ws.cell(row=row_num, column=11, value=float(row['Line']) if pd.notna(row.get('Line')) else 0.0).alignment = right_align
            ws.cell(row=row_num, column=12, value=float(row['True Edge']) if pd.notna(row.get('True Edge')) else 0.0).alignment = right_align

            # Column 13 (M): Sigma
            sig_val = row.get('Sigma')
            ws.cell(row=row_num, column=13, value=float(sig_val) if pd.notna(sig_val) and str(sig_val).strip() != '' else None).alignment = right_align

            # Column 14 (N): Confidence Level
            conf_val = str(row.get('Confidence Level', '')).strip()
            ws.cell(row=row_num, column=14, value=conf_val if conf_val else None).alignment = center_align

            # Column 15 (O): Actual
            val_actual = row.get('Actual')
            if pd.notna(val_actual) and str(val_actual).strip() != '':
                ws.cell(row=row_num, column=15, value=float(val_actual)).alignment = right_align
            else:
                ws.cell(row=row_num, column=15, value=None).alignment = right_align

            # Column 16 (P): Actual O/U formula
            ws.cell(
                row=row_num, column=16,
                value=f'=IF(O{row_num}="","",IF(O{row_num}>K{row_num},"OVER",IF(O{row_num}<K{row_num},"UNDER","PUSH")))'
            ).alignment = center_align

            # Column 17 (Q): Correct formula + highlight color
            c_cell = ws.cell(
                row=row_num, column=17,
                value=f'=IF(OR(O{row_num}="",P{row_num}=""),"",IF(P{row_num}="PUSH","PUSH",IF(J{row_num}=P{row_num},"Yes","No")))'
            )
            c_cell.alignment = center_align

            corr_val = str(row.get('Correct', '')).strip()
            if corr_val == 'Yes':
                c_cell.fill = green_fill
                c_cell.font = green_font
            elif corr_val == 'No':
                c_cell.fill = red_fill
                c_cell.font = red_font

            # Column 18 (R): Miss Diff formula
            ws.cell(
                row=row_num, column=18,
                value=f'=IF(O{row_num}="","",IF(J{row_num}="OVER", IF(O{row_num}<K{row_num}, K{row_num}-O{row_num}, ""), IF(O{row_num}>K{row_num}, O{row_num}-K{row_num}, "")))'
            ).alignment = right_align

            # Column 19 (S): Model Architecture
            arch_val = str(row.get('Model_Architecture', '') or row.get('architecture', 'level_cv')).strip()
            ws.cell(row=row_num, column=19, value=arch_val).alignment = center_align

        n_rows = len(df_reset)
        if n_rows > 0:
            ws.auto_filter.ref = f"A1:S{n_rows+1}"
            rule_yes = CellIsRule(operator='equal', formula=['"Yes"'], stopIfTrue=True, fill=green_fill, font=green_font)
            rule_no = CellIsRule(operator='equal', formula=['"No"'], stopIfTrue=True, fill=red_fill, font=red_font)
            ws.conditional_formatting.add(f"Q2:Q{n_rows+1}", rule_yes)
            ws.conditional_formatting.add(f"Q2:Q{n_rows+1}", rule_no)
        else:
            ws.auto_filter.ref = "A1:S1"

        # Summary Table 1 & 2 Header Row (Row 1)
        ws.cell(row=1, column=21, value="Correct Percentage").font = summary_header_font
        ws.cell(row=1, column=21).alignment = center_align

        ws.merge_cells("X1:AA1")
        role_header = ws.cell(row=1, column=24, value="Role Correct %")
        role_header.font = summary_header_font
        role_header.alignment = center_align

        # Role Column Headers (Row 2)
        roles = [("QB", 24), ("RB", 25), ("WR", 26), ("TE", 27)]
        for role_name, col_num in roles:
            cell_r = ws.cell(row=2, column=col_num, value=role_name)
            cell_r.font = summary_header_font
            cell_r.alignment = center_align

        # Helper to set cell value: if string (e.g. 'N/A'), center and grey font; if formula, right align and 0.0% format
        def set_summary_cell(r, c, val):
            cell = ws.cell(row=r, column=c)
            if val == "N/A":
                cell.value = "N/A"
                cell.font = Font(name="Calibri", size=11, color="888888")
                cell.alignment = center_align
            else:
                cell.value = val
                cell.alignment = right_align
                cell.number_format = '0.0%'

        # Comprehensive Summary Breakdown: Overall, High/Med/Low Confidence, >10 Edge, and Stats
        summary_rows = [
            (
                3, "Overall",
                '=IFERROR(COUNTIFS(Q:Q, "Yes") / (COUNTIFS(Q:Q, "Yes") + COUNTIFS(Q:Q, "No")), "N/A")',
                {
                    "QB": '=IFERROR(COUNTIFS(D:D, "QB", Q:Q, "Yes") / (COUNTIFS(D:D, "QB", Q:Q, "Yes") + COUNTIFS(D:D, "QB", Q:Q, "No")), "N/A")',
                    "RB": '=IFERROR(COUNTIFS(D:D, "RB", Q:Q, "Yes") / (COUNTIFS(D:D, "RB", Q:Q, "Yes") + COUNTIFS(D:D, "RB", Q:Q, "No")), "N/A")',
                    "WR": '=IFERROR(COUNTIFS(D:D, "WR", Q:Q, "Yes") / (COUNTIFS(D:D, "WR", Q:Q, "Yes") + COUNTIFS(D:D, "WR", Q:Q, "No")), "N/A")',
                    "TE": '=IFERROR(COUNTIFS(D:D, "TE", Q:Q, "Yes") / (COUNTIFS(D:D, "TE", Q:Q, "Yes") + COUNTIFS(D:D, "TE", Q:Q, "No")), "N/A")',
                }
            ),
            (
                4, "High Conf",
                '=IFERROR(COUNTIFS(N:N, "HIGH", Q:Q, "Yes") / (COUNTIFS(N:N, "HIGH", Q:Q, "Yes") + COUNTIFS(N:N, "HIGH", Q:Q, "No")), "N/A")',
                {
                    "QB": '=IFERROR(COUNTIFS(D:D, "QB", N:N, "HIGH", Q:Q, "Yes") / (COUNTIFS(D:D, "QB", N:N, "HIGH", Q:Q, "Yes") + COUNTIFS(D:D, "QB", N:N, "HIGH", Q:Q, "No")), "N/A")',
                    "RB": '=IFERROR(COUNTIFS(D:D, "RB", N:N, "HIGH", Q:Q, "Yes") / (COUNTIFS(D:D, "RB", N:N, "HIGH", Q:Q, "Yes") + COUNTIFS(D:D, "RB", N:N, "HIGH", Q:Q, "No")), "N/A")',
                    "WR": '=IFERROR(COUNTIFS(D:D, "WR", N:N, "HIGH", Q:Q, "Yes") / (COUNTIFS(D:D, "WR", N:N, "HIGH", Q:Q, "Yes") + COUNTIFS(D:D, "WR", N:N, "HIGH", Q:Q, "No")), "N/A")',
                    "TE": '=IFERROR(COUNTIFS(D:D, "TE", N:N, "HIGH", Q:Q, "Yes") / (COUNTIFS(D:D, "TE", N:N, "HIGH", Q:Q, "Yes") + COUNTIFS(D:D, "TE", N:N, "HIGH", Q:Q, "No")), "N/A")',
                }
            ),
            (
                5, "Med Conf",
                '=IFERROR(COUNTIFS(N:N, "MED", Q:Q, "Yes") / (COUNTIFS(N:N, "MED", Q:Q, "Yes") + COUNTIFS(N:N, "MED", Q:Q, "No")), "N/A")',
                {
                    "QB": '=IFERROR(COUNTIFS(D:D, "QB", N:N, "MED", Q:Q, "Yes") / (COUNTIFS(D:D, "QB", N:N, "MED", Q:Q, "Yes") + COUNTIFS(D:D, "QB", N:N, "MED", Q:Q, "No")), "N/A")',
                    "RB": '=IFERROR(COUNTIFS(D:D, "RB", N:N, "MED", Q:Q, "Yes") / (COUNTIFS(D:D, "RB", N:N, "MED", Q:Q, "Yes") + COUNTIFS(D:D, "RB", N:N, "MED", Q:Q, "No")), "N/A")',
                    "WR": '=IFERROR(COUNTIFS(D:D, "WR", N:N, "MED", Q:Q, "Yes") / (COUNTIFS(D:D, "WR", N:N, "MED", Q:Q, "Yes") + COUNTIFS(D:D, "WR", N:N, "MED", Q:Q, "No")), "N/A")',
                    "TE": '=IFERROR(COUNTIFS(D:D, "TE", N:N, "MED", Q:Q, "Yes") / (COUNTIFS(D:D, "TE", N:N, "MED", Q:Q, "Yes") + COUNTIFS(D:D, "TE", N:N, "MED", Q:Q, "No")), "N/A")',
                }
            ),
            (
                6, "Low Conf",
                '=IFERROR(COUNTIFS(N:N, "*LOW*", Q:Q, "Yes") / (COUNTIFS(N:N, "*LOW*", Q:Q, "Yes") + COUNTIFS(N:N, "*LOW*", Q:Q, "No")), "N/A")',
                {
                    "QB": '=IFERROR(COUNTIFS(D:D, "QB", N:N, "*LOW*", Q:Q, "Yes") / (COUNTIFS(D:D, "QB", N:N, "*LOW*", Q:Q, "Yes") + COUNTIFS(D:D, "QB", N:N, "*LOW*", Q:Q, "No")), "N/A")',
                    "RB": '=IFERROR(COUNTIFS(D:D, "RB", N:N, "*LOW*", Q:Q, "Yes") / (COUNTIFS(D:D, "RB", N:N, "*LOW*", Q:Q, "Yes") + COUNTIFS(D:D, "RB", N:N, "*LOW*", Q:Q, "No")), "N/A")',
                    "WR": '=IFERROR(COUNTIFS(D:D, "WR", N:N, "*LOW*", Q:Q, "Yes") / (COUNTIFS(D:D, "WR", N:N, "*LOW*", Q:Q, "Yes") + COUNTIFS(D:D, "WR", N:N, "*LOW*", Q:Q, "No")), "N/A")',
                    "TE": '=IFERROR(COUNTIFS(D:D, "TE", N:N, "*LOW*", Q:Q, "Yes") / (COUNTIFS(D:D, "TE", N:N, "*LOW*", Q:Q, "Yes") + COUNTIFS(D:D, "TE", N:N, "*LOW*", Q:Q, "No")), "N/A")',
                }
            ),
            (
                7, ">10 edge",
                '=IFERROR((COUNTIFS(Q:Q, "Yes", L:L, ">10") + COUNTIFS(Q:Q, "Yes", L:L, "<-10")) / ((COUNTIFS(Q:Q, "Yes", L:L, ">10") + COUNTIFS(Q:Q, "Yes", L:L, "<-10")) + (COUNTIFS(Q:Q, "No", L:L, ">10") + COUNTIFS(Q:Q, "No", L:L, "<-10"))), "N/A")',
                {
                    "QB": '=IFERROR((COUNTIFS(D:D, "QB", Q:Q, "Yes", L:L, ">10") + COUNTIFS(D:D, "QB", Q:Q, "Yes", L:L, "<-10")) / ((COUNTIFS(D:D, "QB", Q:Q, "Yes", L:L, ">10") + COUNTIFS(D:D, "QB", Q:Q, "Yes", L:L, "<-10")) + (COUNTIFS(D:D, "QB", Q:Q, "No", L:L, ">10") + COUNTIFS(D:D, "QB", Q:Q, "No", L:L, "<-10"))), "N/A")',
                    "RB": '=IFERROR((COUNTIFS(D:D, "RB", Q:Q, "Yes", L:L, ">10") + COUNTIFS(D:D, "RB", Q:Q, "Yes", L:L, "<-10")) / ((COUNTIFS(D:D, "RB", Q:Q, "Yes", L:L, ">10") + COUNTIFS(D:D, "RB", Q:Q, "Yes", L:L, "<-10")) + (COUNTIFS(D:D, "RB", Q:Q, "No", L:L, ">10") + COUNTIFS(D:D, "RB", Q:Q, "No", L:L, "<-10"))), "N/A")',
                    "WR": '=IFERROR((COUNTIFS(D:D, "WR", Q:Q, "Yes", L:L, ">10") + COUNTIFS(D:D, "WR", Q:Q, "Yes", L:L, "<-10")) / ((COUNTIFS(D:D, "WR", Q:Q, "Yes", L:L, ">10") + COUNTIFS(D:D, "WR", Q:Q, "Yes", L:L, "<-10")) + (COUNTIFS(D:D, "WR", Q:Q, "No", L:L, ">10") + COUNTIFS(D:D, "WR", Q:Q, "No", L:L, "<-10"))), "N/A")',
                    "TE": '=IFERROR((COUNTIFS(D:D, "TE", Q:Q, "Yes", L:L, ">10") + COUNTIFS(D:D, "TE", Q:Q, "Yes", L:L, "<-10")) / ((COUNTIFS(D:D, "TE", Q:Q, "Yes", L:L, ">10") + COUNTIFS(D:D, "TE", Q:Q, "Yes", L:L, "<-10")) + (COUNTIFS(D:D, "TE", Q:Q, "No", L:L, ">10") + COUNTIFS(D:D, "TE", Q:Q, "No", L:L, "<-10"))), "N/A")',
                }
            ),
            (
                8, "Rushing",
                '=IFERROR(COUNTIFS(E:E, "*Rush*", Q:Q, "Yes") / (COUNTIFS(E:E, "*Rush*", Q:Q, "Yes") + COUNTIFS(E:E, "*Rush*", Q:Q, "No")), "N/A")',
                {
                    "QB": '=IFERROR(COUNTIFS(D:D, "QB", E:E, "*Rush*", Q:Q, "Yes") / (COUNTIFS(D:D, "QB", E:E, "*Rush*", Q:Q, "Yes") + COUNTIFS(D:D, "QB", E:E, "*Rush*", Q:Q, "No")), "N/A")',
                    "RB": '=IFERROR(COUNTIFS(D:D, "RB", E:E, "*Rush*", Q:Q, "Yes") / (COUNTIFS(D:D, "RB", E:E, "*Rush*", Q:Q, "Yes") + COUNTIFS(D:D, "RB", E:E, "*Rush*", Q:Q, "No")), "N/A")',
                    "WR": "N/A",
                    "TE": "N/A",
                }
            ),
            (
                9, "Receiving",
                '=IFERROR(COUNTIFS(E:E, "*Receiv*", Q:Q, "Yes") / (COUNTIFS(E:E, "*Receiv*", Q:Q, "Yes") + COUNTIFS(E:E, "*Receiv*", Q:Q, "No")), "N/A")',
                {
                    "QB": "N/A",
                    "RB": '=IFERROR(COUNTIFS(D:D, "RB", E:E, "*Receiv*", Q:Q, "Yes") / (COUNTIFS(D:D, "RB", E:E, "*Receiv*", Q:Q, "Yes") + COUNTIFS(D:D, "RB", E:E, "*Receiv*", Q:Q, "No")), "N/A")',
                    "WR": '=IFERROR(COUNTIFS(D:D, "WR", E:E, "*Receiv*", Q:Q, "Yes") / (COUNTIFS(D:D, "WR", E:E, "*Receiv*", Q:Q, "Yes") + COUNTIFS(D:D, "WR", E:E, "*Receiv*", Q:Q, "No")), "N/A")',
                    "TE": '=IFERROR(COUNTIFS(D:D, "TE", E:E, "*Receiv*", Q:Q, "Yes") / (COUNTIFS(D:D, "TE", E:E, "*Receiv*", Q:Q, "Yes") + COUNTIFS(D:D, "TE", E:E, "*Receiv*", Q:Q, "No")), "N/A")',
                }
            ),
            (
                10, "Passing",
                '=IFERROR(COUNTIFS(E:E, "*Pass*", Q:Q, "Yes") / (COUNTIFS(E:E, "*Pass*", Q:Q, "Yes") + COUNTIFS(E:E, "*Pass*", Q:Q, "No")), "N/A")',
                {
                    "QB": '=IFERROR(COUNTIFS(D:D, "QB", E:E, "*Pass*", Q:Q, "Yes") / (COUNTIFS(D:D, "QB", E:E, "*Pass*", Q:Q, "Yes") + COUNTIFS(D:D, "QB", E:E, "*Pass*", Q:Q, "No")), "N/A")',
                    "RB": "N/A",
                    "WR": "N/A",
                    "TE": "N/A",
                }
            ),
            (
                11, ">10 Rushing",
                '=IFERROR((COUNTIFS(E:E, "*Rush*", Q:Q, "Yes", L:L, ">10") + COUNTIFS(E:E, "*Rush*", Q:Q, "Yes", L:L, "<-10")) / ((COUNTIFS(E:E, "*Rush*", Q:Q, "Yes", L:L, ">10") + COUNTIFS(E:E, "*Rush*", Q:Q, "Yes", L:L, "<-10")) + (COUNTIFS(E:E, "*Rush*", Q:Q, "No", L:L, ">10") + COUNTIFS(E:E, "*Rush*", Q:Q, "No", L:L, "<-10"))), "N/A")',
                {
                    "QB": '=IFERROR((COUNTIFS(D:D, "QB", E:E, "*Rush*", Q:Q, "Yes", L:L, ">10") + COUNTIFS(D:D, "QB", E:E, "*Rush*", Q:Q, "Yes", L:L, "<-10")) / ((COUNTIFS(D:D, "QB", E:E, "*Rush*", Q:Q, "Yes", L:L, ">10") + COUNTIFS(D:D, "QB", E:E, "*Rush*", Q:Q, "Yes", L:L, "<-10")) + (COUNTIFS(D:D, "QB", E:E, "*Rush*", Q:Q, "No", L:L, ">10") + COUNTIFS(D:D, "QB", E:E, "*Rush*", Q:Q, "No", L:L, "<-10"))), "N/A")',
                    "RB": '=IFERROR((COUNTIFS(D:D, "RB", E:E, "*Rush*", Q:Q, "Yes", L:L, ">10") + COUNTIFS(D:D, "RB", E:E, "*Rush*", Q:Q, "Yes", L:L, "<-10")) / ((COUNTIFS(D:D, "RB", E:E, "*Rush*", Q:Q, "Yes", L:L, ">10") + COUNTIFS(D:D, "RB", E:E, "*Rush*", Q:Q, "Yes", L:L, "<-10")) + (COUNTIFS(D:D, "RB", E:E, "*Rush*", Q:Q, "No", L:L, ">10") + COUNTIFS(D:D, "RB", E:E, "*Rush*", Q:Q, "No", L:L, "<-10"))), "N/A")',
                    "WR": "N/A",
                    "TE": "N/A",
                }
            ),
            (
                12, ">10 Receiving",
                '=IFERROR((COUNTIFS(E:E, "*Receiv*", Q:Q, "Yes", L:L, ">10") + COUNTIFS(E:E, "*Receiv*", Q:Q, "Yes", L:L, "<-10")) / ((COUNTIFS(E:E, "*Receiv*", Q:Q, "Yes", L:L, ">10") + COUNTIFS(E:E, "*Receiv*", Q:Q, "Yes", L:L, "<-10")) + (COUNTIFS(E:E, "*Receiv*", Q:Q, "No", L:L, ">10") + COUNTIFS(E:E, "*Receiv*", Q:Q, "No", L:L, "<-10"))), "N/A")',
                {
                    "QB": "N/A",
                    "RB": '=IFERROR((COUNTIFS(D:D, "RB", E:E, "*Receiv*", Q:Q, "Yes", L:L, ">10") + COUNTIFS(D:D, "RB", E:E, "*Receiv*", Q:Q, "Yes", L:L, "<-10")) / ((COUNTIFS(D:D, "RB", E:E, "*Receiv*", Q:Q, "Yes", L:L, ">10") + COUNTIFS(D:D, "RB", E:E, "*Receiv*", Q:Q, "Yes", L:L, "<-10")) + (COUNTIFS(D:D, "RB", E:E, "*Receiv*", Q:Q, "No", L:L, ">10") + COUNTIFS(D:D, "RB", E:E, "*Receiv*", Q:Q, "No", L:L, "<-10"))), "N/A")',
                    "WR": '=IFERROR((COUNTIFS(D:D, "WR", E:E, "*Receiv*", Q:Q, "Yes", L:L, ">10") + COUNTIFS(D:D, "WR", E:E, "*Receiv*", Q:Q, "Yes", L:L, "<-10")) / ((COUNTIFS(D:D, "WR", E:E, "*Receiv*", Q:Q, "Yes", L:L, ">10") + COUNTIFS(D:D, "WR", E:E, "*Receiv*", Q:Q, "Yes", L:L, "<-10")) + (COUNTIFS(D:D, "WR", E:E, "*Receiv*", Q:Q, "No", L:L, ">10") + COUNTIFS(D:D, "WR", E:E, "*Receiv*", Q:Q, "No", L:L, "<-10"))), "N/A")',
                    "TE": '=IFERROR((COUNTIFS(D:D, "TE", E:E, "*Receiv*", Q:Q, "Yes", L:L, ">10") + COUNTIFS(D:D, "TE", E:E, "*Receiv*", Q:Q, "Yes", L:L, "<-10")) / ((COUNTIFS(D:D, "TE", E:E, "*Receiv*", Q:Q, "Yes", L:L, ">10") + COUNTIFS(D:D, "TE", E:E, "*Receiv*", Q:Q, "Yes", L:L, "<-10")) + (COUNTIFS(D:D, "TE", E:E, "*Receiv*", Q:Q, "No", L:L, ">10") + COUNTIFS(D:D, "TE", E:E, "*Receiv*", Q:Q, "No", L:L, "<-10"))), "N/A")',
                }
            ),
            (
                13, ">10 Passing",
                '=IFERROR((COUNTIFS(E:E, "*Pass*", Q:Q, "Yes", L:L, ">10") + COUNTIFS(E:E, "*Pass*", Q:Q, "Yes", L:L, "<-10")) / ((COUNTIFS(E:E, "*Pass*", Q:Q, "Yes", L:L, ">10") + COUNTIFS(E:E, "*Pass*", Q:Q, "Yes", L:L, "<-10")) + (COUNTIFS(E:E, "*Pass*", Q:Q, "No", L:L, ">10") + COUNTIFS(E:E, "*Pass*", Q:Q, "No", L:L, "<-10"))), "N/A")',
                {
                    "QB": '=IFERROR((COUNTIFS(D:D, "QB", E:E, "*Pass*", Q:Q, "Yes", L:L, ">10") + COUNTIFS(D:D, "QB", E:E, "*Pass*", Q:Q, "Yes", L:L, "<-10")) / ((COUNTIFS(D:D, "QB", E:E, "*Pass*", Q:Q, "Yes", L:L, ">10") + COUNTIFS(D:D, "QB", E:E, "*Pass*", Q:Q, "Yes", L:L, "<-10")) + (COUNTIFS(D:D, "QB", E:E, "*Pass*", Q:Q, "No", L:L, ">10") + COUNTIFS(D:D, "QB", E:E, "*Pass*", Q:Q, "No", L:L, "<-10"))), "N/A")',
                    "RB": "N/A",
                    "WR": "N/A",
                    "TE": "N/A",
                }
            )
        ]

        for row_idx, label, total_formula, role_dict in summary_rows:
            ws.cell(row=row_idx, column=20, value=label).font = summary_header_font
            ws.cell(row=row_idx, column=20).alignment = left_align
            set_summary_cell(row_idx, 21, total_formula)
            for role_name, col_num in roles:
                set_summary_cell(row_idx, col_num, role_dict.get(role_name, "N/A"))

        # Explicit column widths
        explicit_widths = {
            'A': 11,   # Date
            'B': 20,   # Player
            'C': 12,   # Matchup
            'D': 6,    # Pos
            'E': 16,   # Stat
            'F': 6,    # Rank
            'G': 7,    # Avg
            'H': 7,    # PRED
            'I': 7,    # Edge
            'J': 7,    # O/U
            'K': 7,    # Line
            'L': 10,   # True Edge
            'M': 8,    # Sigma
            'N': 16,   # Confidence Level
            'O': 8,    # Actual
            'P': 11,   # Actual O/U
            'Q': 9,    # Correct
            'R': 10,   # Miss Diff
            'S': 2,    # Slim spacer column
            'T': 18,   # Label (Overall, High Conf, Med Conf, Low Conf, >10 edge...)
            'U': 18,   # Correct Percentage
            'V': 2,    # Slim spacer column
            'W': 2,    # Slim spacer column
            'X': 10,   # QB
            'Y': 10,   # RB
            'Z': 10,   # WR
            'AA': 10   # TE
        }
        for col_letter, w in explicit_widths.items():
            ws.column_dimensions[col_letter].width = w

        try:
            wb.save(xlsx_path)
            print(f"[Excel Tracker] Generated interactive {xlsx_path} with High/Med/Low accuracy breakdown tables")
            return xlsx_path
        except PermissionError as pe:
            print(f"[Excel Tracker] PermissionError saving '{xlsx_path}': {pe}. Close Excel to update.")
            return ""
    except Exception as e:
        print(f"[Excel Tracker] Warning generating {xlsx_filename}: {e}")
        return ""


def export_weekly_matchups_xlsx(schedule_data: dict, week=None, season: int = 2026, output_file: str = None) -> bytes:
    """
    Generates a beautifully formatted, multi-sheet Excel workbook (.xlsx) containing:
    1. Matchups Overview (Vegas lines, implied totals, weather, records, results, funnel notes)
    2. Side-by-Side Stat Comparison (Head-to-head offensive & defensive stats with 1-32 league ranks)
    3. 32-Team League Baseline (Comprehensive 32-team rankings and metrics table)

    Returns workbook bytes for direct browser download and optionally saves to output_file.
    """
    import io
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    wb = openpyxl.Workbook()
    # Remove default sheet
    wb.remove(wb.active)

    # Styles
    title_font = Font(name="Calibri", size=14, bold=True, color="FFFFFF")
    title_fill = PatternFill(start_color="0F172A", end_color="0F172A", fill_type="solid")
    subtitle_font = Font(name="Calibri", size=10, italic=True, color="94A3B8")

    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="1E293B", end_color="1E293B", fill_type="solid")
    sub_header_fill = PatternFill(start_color="334155", end_color="334155", fill_type="solid")

    green_fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    green_font = Font(name="Calibri", size=10, color="006100", bold=True)
    red_fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
    red_font = Font(name="Calibri", size=10, color="9C0006", bold=True)
    blue_fill = PatternFill(start_color="E0F2FE", end_color="E0F2FE", fill_type="solid")
    blue_font = Font(name="Calibri", size=10, color="0369A1", bold=True)

    zebra_fill = PatternFill(start_color="F8FAFC", end_color="F8FAFC", fill_type="solid")
    thin_border_side = Side(style='thin', color="E2E8F0")
    cell_border = Border(left=thin_border_side, right=thin_border_side, top=thin_border_side, bottom=thin_border_side)

    center_align = Alignment(horizontal="center", vertical="center")
    left_align = Alignment(horizontal="left", vertical="center")
    right_align = Alignment(horizontal="right", vertical="center")

    # Filter weeks
    weeks_list = schedule_data.get('weeks', [])
    if week is not None and str(week).lower() != 'all':
        try:
            target_w = int(week)
            weeks_list = [w for w in weeks_list if w.get('week') == target_w]
        except (ValueError, TypeError):
            pass

    week_title_str = f"Week {week}" if (week is not None and str(week).lower() != 'all') else "All Regular Season Weeks"
    baseline_note = schedule_data.get('stats_baseline_note', f"{season} Season Stats")

    # Helper function to style rank cells
    def style_rank_cell(cell, rank_val, high_is_good=True):
        cell.alignment = center_align
        cell.border = cell_border
        if rank_val is None or rank_val == "" or rank_val == "--":
            return
        try:
            r = int(rank_val)
            is_top = (r <= 8) if high_is_good else (r <= 8)
            is_bot = (r >= 25) if high_is_good else (r >= 25)
            if (high_is_good and r <= 8) or (not high_is_good and r <= 8):
                cell.fill = green_fill
                cell.font = green_font
            elif (high_is_good and r >= 25) or (not high_is_good and r >= 25):
                cell.fill = red_fill
                cell.font = red_font
            else:
                cell.fill = blue_fill
                cell.font = blue_font
        except (ValueError, TypeError):
            pass

    # =========================================================================
    # SHEET 1: MATCHUPS OVERVIEW
    # =========================================================================
    ws1 = wb.create_sheet(title=f"Matchups Overview")
    ws1.views.sheetView[0].showGridLines = True

    # Title Block
    ws1.merge_cells("A1:V1")
    t1 = ws1.cell(row=1, column=1, value=f"NFL {season} Season • {week_title_str} Matchups & Odds Overview")
    t1.font = title_font
    t1.fill = title_fill
    t1.alignment = left_align
    ws1.row_dimensions[1].height = 28

    ws1.merge_cells("A2:V2")
    t1_sub = ws1.cell(row=2, column=1, value=f"Stats Baseline: {baseline_note} • Source: nflreadpy & The Odds API")
    t1_sub.font = subtitle_font
    t1_sub.fill = title_fill
    t1_sub.alignment = left_align
    ws1.row_dimensions[2].height = 18

    # Table 1 Headers
    overview_headers = [
        "Week", "Date", "Day", "Kickoff", "Matchup",
        "Away Team", "Away Name", "Away Record",
        "Home Team", "Home Name", "Home Record",
        "Spread", "Total (O/U)", "Implied Away Pts", "Implied Home Pts",
        "Stadium / Venue", "Roof / Surface", "Temp (°F)", "Wind (mph)",
        "Status", "Away Score", "Home Score", "Matchup & Funnel Notes"
    ]

    ws1.row_dimensions[3].height = 24
    for col_idx, h_text in enumerate(overview_headers, start=1):
        c = ws1.cell(row=3, column=col_idx, value=h_text)
        c.font = header_font
        c.fill = header_fill
        c.alignment = center_align
        c.border = cell_border

    curr_row = 4
    for w in weeks_list:
        games = w.get('games', [])
        for g in games:
            ws1.row_dimensions[curr_row].height = 20
            is_zebra = (curr_row % 2 == 1)
            row_fill = zebra_fill if is_zebra else None

            # Calculate Spread text
            spread_str = "--"
            if g.get('spread_line') is not None:
                sp = g['spread_line']
                spread_str = "PK" if sp == 0 else (f"{g['home_team']} -{sp}" if sp > 0 else f"{g['away_team']} -{abs(sp)}")

            # Status string
            status_str = "FINAL" if g.get('is_final') else "UPCOMING"

            # Funnel Notes summary
            funnel = g.get('funnel') or {}
            funnel_notes = []
            if funnel.get('pass_funnel_alert'):
                funnel_notes.append("Pass Funnel Alert")
            if funnel.get('rush_funnel_alert'):
                funnel_notes.append("Rush Funnel Alert")
            if funnel.get('shootout_alert'):
                funnel_notes.append("High-Scoring Shootout")
            if funnel.get('grind_alert'):
                funnel_notes.append("Defensive Grind")
            if not funnel_notes and funnel.get('summary'):
                funnel_notes.append(funnel['summary'])
            funnel_str = " • ".join(funnel_notes) if funnel_notes else "--"

            row_data = [
                g.get('week'),
                g.get('gameday', ''),
                g.get('weekday', ''),
                g.get('gametime', ''),
                f"{g.get('away_team')} @ {g.get('home_team')}",
                g.get('away_team', ''),
                g.get('away_team_name', ''),
                g.get('away_record', '0-0'),
                g.get('home_team', ''),
                g.get('home_team_name', ''),
                g.get('home_record', '0-0'),
                spread_str,
                g.get('total_line'),
                g.get('implied_away_total'),
                g.get('implied_home_total'),
                g.get('stadium', ''),
                f"{g.get('roof', '')} / {g.get('surface', '')}".strip(" /"),
                g.get('temp'),
                g.get('wind'),
                status_str,
                g.get('away_score'),
                g.get('home_score'),
                funnel_str
            ]

            for c_idx, val in enumerate(row_data, start=1):
                cell = ws1.cell(row=curr_row, column=c_idx, value=val)
                cell.border = cell_border
                if row_fill:
                    cell.fill = row_fill

                # Alignments & formats
                if c_idx in [1, 2, 3, 4, 5, 6, 8, 12, 17, 20]:
                    cell.alignment = center_align
                elif c_idx in [7, 10, 16, 23]:
                    cell.alignment = left_align
                else:
                    cell.alignment = right_align

                if c_idx in [13, 14, 15, 18, 19] and isinstance(val, (int, float)):
                    cell.number_format = '0.0'
                elif c_idx in [21, 22] and isinstance(val, (int, float)):
                    cell.number_format = '0'

                if c_idx == 20:  # Status highlight
                    if val == "FINAL":
                        cell.fill = green_fill
                        cell.font = green_font

            curr_row += 1

    ws1.freeze_panes = "A4"
    ws1.auto_filter.ref = f"A3:W{max(curr_row - 1, 3)}"

    # Auto column widths for Sheet 1
    for col in ws1.columns:
        max_len = max(len(str(cell.value or '')) for cell in col)
        col_letter = get_column_letter(col[0].column)
        ws1.column_dimensions[col_letter].width = max(max_len + 3, 9)
    ws1.column_dimensions['W'].width = 32  # Matchup notes

    # =========================================================================
    # SHEET 2: SIDE-BY-SIDE MATCHUP COMPARISON
    # =========================================================================
    ws2 = wb.create_sheet(title=f"Matchup H2H Stats")
    ws2.views.sheetView[0].showGridLines = True

    ws2.merge_cells("A1:AZ1")
    t2 = ws2.cell(row=1, column=1, value=f"NFL {season} Season • {week_title_str} Head-to-Head Stats & 1-32 League Rankings")
    t2.font = title_font
    t2.fill = title_fill
    t2.alignment = left_align
    ws2.row_dimensions[1].height = 28

    ws2.merge_cells("A2:AZ2")
    t2_sub = ws2.cell(row=2, column=1, value=f"Rankings reflect 1 (Best in NFL) to 32 (Worst in NFL) • Baseline: {baseline_note}")
    t2_sub.font = subtitle_font
    t2_sub.fill = title_fill
    t2_sub.alignment = left_align
    ws2.row_dimensions[2].height = 18

    h2h_headers = [
        "Week", "Matchup", "Away", "Home",
        # Scoring
        "Away PPG", "Away PPG Rank", "Home PPG", "Home PPG Rank",
        "Away PAPG", "Away PAPG Rank", "Home PAPG", "Home PAPG Rank",
        "Away Net Margin", "Home Net Margin",
        # Passing
        "Away Pass YPG", "Away Pass Rank", "Home Pass YPG", "Home Pass Rank",
        "Away Pass Def YPG", "Away Pass Def Rank", "Home Pass Def YPG", "Home Pass Def Rank",
        # Rushing
        "Away Rush YPG", "Away Rush Rank", "Home Rush YPG", "Home Rush Rank",
        "Away Rush Def YPG", "Away Rush Def Rank", "Home Rush Def YPG", "Home Rush Def Rank",
        # Total
        "Away Total YPG", "Away Total Rank", "Home Total YPG", "Home Total Rank",
        "Away Total Def YPG", "Away Total Def Rank", "Home Total Def YPG", "Home Total Def Rank",
        # Defense / Sacks / Turnovers
        "Away Sacks F/G", "Away Sacks F Rank", "Home Sacks F/G", "Home Sacks F Rank",
        "Away Sacks S/G", "Away Sacks S Rank", "Home Sacks S/G", "Home Sacks S Rank",
        "Away Turnover Margin", "Away TO Rank", "Home Turnover Margin", "Home TO Rank",
        # Matchup Verdicts
        "Away Defense vs RB1", "Home Defense vs RB1",
        "Away Defense vs RB2", "Home Defense vs RB2",
        "Away Defense vs WR1", "Home Defense vs WR1",
        "Away Defense vs WR2", "Home Defense vs WR2",
        "Away QB Scramble Contain", "Home QB Scramble Contain",
        "Away RB Rec Contain", "Home RB Rec Contain"
    ]

    ws2.row_dimensions[3].height = 24
    for col_idx, h_text in enumerate(h2h_headers, start=1):
        c = ws2.cell(row=3, column=col_idx, value=h_text)
        c.font = header_font
        c.fill = header_fill
        c.alignment = center_align
        c.border = cell_border

    curr_row2 = 4
    for w in weeks_list:
        for g in w.get('games', []):
            ws2.row_dimensions[curr_row2].height = 20
            is_zebra = (curr_row2 % 2 == 1)
            row_fill = zebra_fill if is_zebra else None

            a = g.get('away_stats') or {}
            h = g.get('home_stats') or {}

            # Build stat row values
            h2h_row = [
                g.get('week'),
                f"{g.get('away_team')} @ {g.get('home_team')}",
                g.get('away_team'),
                g.get('home_team'),
                # Scoring
                a.get('pts_per_game'), a.get('rank_pts_scored'),
                h.get('pts_per_game'), h.get('rank_pts_scored'),
                a.get('pts_allowed_per_game'), a.get('rank_pts_allowed'),
                h.get('pts_allowed_per_game'), h.get('rank_pts_allowed'),
                a.get('point_differential'), h.get('point_differential'),
                # Passing
                a.get('pass_yds_per_game'), a.get('rank_pass_yds'),
                h.get('pass_yds_per_game'), h.get('rank_pass_yds'),
                a.get('pass_yds_allowed_per_game'), a.get('rank_pass_allowed'),
                h.get('pass_yds_allowed_per_game'), h.get('rank_pass_allowed'),
                # Rushing
                a.get('rush_yds_per_game'), a.get('rank_rush_yds'),
                h.get('rush_yds_per_game'), h.get('rank_rush_yds'),
                a.get('rush_yds_allowed_per_game'), a.get('rank_rush_allowed'),
                h.get('rush_yds_allowed_per_game'), h.get('rank_rush_allowed'),
                # Total
                a.get('total_yds_per_game'), a.get('rank_total_yds'),
                h.get('total_yds_per_game'), h.get('rank_total_yds'),
                a.get('total_yds_allowed_per_game'), a.get('rank_total_allowed'),
                h.get('total_yds_allowed_per_game'), h.get('rank_total_allowed'),
                # Sacks / TO
                a.get('sacks_forced_per_game'), a.get('rank_sacks_forced'),
                h.get('sacks_forced_per_game'), h.get('rank_sacks_forced'),
                a.get('sacks_suffered_per_game'), a.get('rank_sacks_suffered'),
                h.get('sacks_suffered_per_game'), h.get('rank_sacks_suffered'),
                a.get('turnover_diff'), a.get('rank_turnover_diff'),
                h.get('turnover_diff'), h.get('rank_turnover_diff'),
                # Verdicts
                a.get('def_star_rb_verdict', '--'), h.get('def_star_rb_verdict', '--'),
                a.get('def_rb2_verdict', '--'), h.get('def_rb2_verdict', '--'),
                a.get('def_star_wr_verdict', '--'), h.get('def_star_wr_verdict', '--'),
                a.get('def_wr2_verdict', '--'), h.get('def_wr2_verdict', '--'),
                a.get('def_te1_verdict', '--'), h.get('def_te1_verdict', '--'),
                a.get('def_te2_verdict', '--'), h.get('def_te2_verdict', '--'),
                a.get('def_qb_rush_verdict', '--'), h.get('def_qb_rush_verdict', '--'),
                a.get('def_rb_rec_verdict', '--'), h.get('def_rb_rec_verdict', '--')
            ]

            # Rank column indices in h2h_headers (1-indexed)
            rank_col_map = {
                6: True, 8: True, 10: False, 12: False,
                16: True, 18: True, 20: False, 22: False,
                24: True, 26: True, 28: False, 30: False,
                32: True, 34: True, 36: False, 38: False,
                40: True, 42: True, 44: False, 46: False,
                48: True, 50: True
            }

            for c_idx, val in enumerate(h2h_row, start=1):
                cell = ws2.cell(row=curr_row2, column=c_idx, value=val)
                cell.border = cell_border
                if row_fill:
                    cell.fill = row_fill

                if c_idx in rank_col_map:
                    high_good = rank_col_map[c_idx]
                    style_rank_cell(cell, val, high_is_good=high_good)
                elif c_idx in [1, 2, 3, 4]:
                    cell.alignment = center_align
                elif c_idx >= 51:
                    cell.alignment = left_align
                else:
                    cell.alignment = right_align
                    if isinstance(val, (int, float)):
                        cell.number_format = '0.0'

            curr_row2 += 1

    ws2.freeze_panes = "E4"
    ws2.auto_filter.ref = f"A3:AZ{max(curr_row2 - 1, 3)}"

    for col in ws2.columns:
        max_len = max(len(str(cell.value or '')) for cell in col)
        col_letter = get_column_letter(col[0].column)
        ws2.column_dimensions[col_letter].width = max(max_len + 3, 9)

    # =========================================================================
    # SHEET 3: 32-TEAM LEAGUE BASELINE RANKINGS
    # =========================================================================
    ws3 = wb.create_sheet(title="32-Team League Stats")
    ws3.views.sheetView[0].showGridLines = True

    ws3.merge_cells("A1:AB1")
    t3 = ws3.cell(row=1, column=1, value=f"NFL 32-Team Statistical Overview & Rankings • {baseline_note}")
    t3.font = title_font
    t3.fill = title_fill
    t3.alignment = left_align
    ws3.row_dimensions[1].height = 28

    ws3.merge_cells("A2:AB2")
    t3_sub = ws3.cell(row=2, column=1, value="Yahoo Sports-style comprehensive rankings • Green = Top 8, Red = Bottom 8")
    t3_sub.font = subtitle_font
    t3_sub.fill = title_fill
    t3_sub.alignment = left_align
    ws3.row_dimensions[2].height = 18

    league_headers = [
        "Pts Rank", "Team", "Games",
        "PPG", "Pts Allowed/G", "Point Diff",
        "Pass YPG", "Pass Rank", "Pass Allowed/G", "Pass Def Rank",
        "Rush YPG", "Rush Rank", "Rush Allowed/G", "Rush Def Rank",
        "Total YPG", "Total Rank", "Total Allowed/G", "Total Def Rank",
        "Carries/G", "Pass Att/G", "Yds/Carry", "Yds/Pass Att", "Cmp %",
        "Turnover Diff", "TO Rank", "Sacks F/G", "Sacks S/G",
        "Defense vs RB1", "Defense vs RB2", "Defense vs WR1", "Defense vs WR2"
    ]

    ws3.row_dimensions[3].height = 24
    for col_idx, h_text in enumerate(league_headers, start=1):
        c = ws3.cell(row=3, column=col_idx, value=h_text)
        c.font = header_font
        c.fill = header_fill
        c.alignment = center_align
        c.border = cell_border

    # Extract all teams from games or schedule
    all_teams_stats = {}
    for w in schedule_data.get('weeks', []):
        for g in w.get('games', []):
            if g.get('home_team') and g.get('home_stats'):
                all_teams_stats[g['home_team']] = g['home_stats']
            if g.get('away_team') and g.get('away_stats'):
                all_teams_stats[g['away_team']] = g['away_stats']

    sorted_teams = sorted(
        all_teams_stats.values(),
        key=lambda x: (x.get('rank_pts_scored') or 99, -(x.get('pts_per_game') or 0))
    )

    curr_row3 = 4
    for t_stat in sorted_teams:
        ws3.row_dimensions[curr_row3].height = 20
        is_zebra = (curr_row3 % 2 == 1)
        row_fill = zebra_fill if is_zebra else None

        t_row = [
            t_stat.get('rank_pts_scored', 16),
            t_stat.get('team', ''),
            t_stat.get('games_played', 0),
            t_stat.get('pts_per_game', 0.0),
            t_stat.get('pts_allowed_per_game', 0.0),
            t_stat.get('point_differential', 0.0),
            t_stat.get('pass_yds_per_game', 0.0),
            t_stat.get('rank_pass_yds', 16),
            t_stat.get('pass_yds_allowed_per_game', 0.0),
            t_stat.get('rank_pass_allowed', 16),
            t_stat.get('rush_yds_per_game', 0.0),
            t_stat.get('rank_rush_yds', 16),
            t_stat.get('rush_yds_allowed_per_game', 0.0),
            t_stat.get('rank_rush_allowed', 16),
            t_stat.get('total_yds_per_game', 0.0),
            t_stat.get('rank_total_yds', 16),
            t_stat.get('total_yds_allowed_per_game', 0.0),
            t_stat.get('rank_total_allowed', 16),
            t_stat.get('carries_per_game', 0.0),
            t_stat.get('pass_att_per_game', 0.0),
            t_stat.get('yds_per_carry', 0.0),
            t_stat.get('yds_per_att', 0.0),
            t_stat.get('cmp_pct', 0.0),
            t_stat.get('turnover_diff', 0),
            t_stat.get('rank_turnover_diff', 16),
            t_stat.get('sacks_forced_per_game', 0.0),
            t_stat.get('sacks_suffered_per_game', 0.0),
            t_stat.get('def_star_rb_verdict', '--'),
            t_stat.get('def_rb2_verdict', '--'),
            t_stat.get('def_star_wr_verdict', '--'),
            t_stat.get('def_wr2_verdict', '--')
        ]

        # Rank columns in sheet 3: 1, 8, 10, 12, 14, 16, 18, 25
        s3_rank_cols = {1: True, 8: True, 10: False, 12: True, 14: False, 16: True, 18: False, 25: True}

        for c_idx, val in enumerate(t_row, start=1):
            cell = ws3.cell(row=curr_row3, column=c_idx, value=val)
            cell.border = cell_border
            if row_fill:
                cell.fill = row_fill

            if c_idx in s3_rank_cols:
                style_rank_cell(cell, val, high_is_good=s3_rank_cols[c_idx])
            elif c_idx in [2, 3]:
                cell.alignment = center_align
            elif c_idx in [28, 29]:
                cell.alignment = left_align
            else:
                cell.alignment = right_align
                if isinstance(val, (int, float)):
                    if c_idx == 23:
                        cell.number_format = '0.0"%"'
                    elif c_idx == 24:
                        cell.number_format = '0'
                    else:
                        cell.number_format = '0.0'

        curr_row3 += 1

    ws3.freeze_panes = "C4"
    ws3.auto_filter.ref = f"A3:AC{max(curr_row3 - 1, 3)}"

    for col in ws3.columns:
        max_len = max(len(str(cell.value or '')) for cell in col)
        col_letter = get_column_letter(col[0].column)
        ws3.column_dimensions[col_letter].width = max(max_len + 3, 9)

    # Save to buffer
    buf = io.BytesIO()
    wb.save(buf)
    excel_bytes = buf.getvalue()

    if output_file:
        try:
            with open(output_file, "wb") as f:
                f.write(excel_bytes)
        except Exception as e:
            print(f"[Excel Matchup Export] Could not save copy to {output_file}: {e}")

    return excel_bytes

