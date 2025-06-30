import pandas as pd
def debug_carriermix(in_frac, out_frac):
    # 1) Gather only the entries you know exist
    records = []
    for (g,f), v in out_frac.items():
        records.append({
            'tech':      g,
            'direction': 'Export',
            'carrier':   f,
            'value':     round(v,4)
            })
    for (g,f), v in in_frac.items():
        # in_frac was your dict[(g,f)] = v (only positives)
        records.append({
            'tech':      g,
            'direction': 'Import',
            'carrier':   f,
            'value':     round(v,4)
        })

    # 2) Build a flat DataFrame
    df = pd.DataFrame.from_records(records)

    # 3) Pivot so carriers become columns, blanking missing cells
    df_pivot = (
        df
        .pivot(index=['tech','direction'], columns='carrier', values='value')
        .fillna('')
    )

    # 4) Write to Excel
    df_pivot.to_excel('in_out_frac.xlsx', sheet_name='σ-fractions')
    print("► wrote in_out_frac.xlsx")