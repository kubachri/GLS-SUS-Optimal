import pandas as pd
from pyomo.environ import value

# After solving your model, collect production and storage results into DataFrames
def export_results_to_excel(model, filename='results.xlsx'):
    # Production for all technologies
    prod_data = []
    for (g, f) in model.OUT:
        for t in model.T:
            gen = value(model.Generation[g, f, t])
            prod_data.append({
                'Tech': g,
                'Energy': f,
                'Hour': t,
                'Generation': gen
            })
    df_prod = pd.DataFrame(prod_data)

    # Storage metrics: charge (FuelUseTotal), discharge (Generation), state-of-charge
    stor_data = []
    for g in model.G_s:
        for t in model.T:
            charge = value(model.FuelUseTotal[g, t])
            # discharge: sum of all out-flows for this storage tech
            discharge = sum(
                value(model.Generation[g, e, t]) * model.out_frac[g, e]
                for (gg, e) in model.OUT if gg == g
            )
            soc = value(model.Volume[g, t])
            stor_data.append({
                'StorageTech': g,
                'Hour': t,
                'Charge': charge,
                'Discharge': discharge,
                'StateOfCharge': soc,
                'ChargeStatus': value(model.Charge[g, t])
            })
    df_stor = pd.DataFrame(stor_data)

    # Write to Excel with two sheets
    with pd.ExcelWriter('results.xlsx') as writer:
        df_prod.to_excel(writer, sheet_name='Production', index=False)
        df_stor.to_excel(writer, sheet_name='Storage', index=False)

    return filename

# Example usage:
# filepath = export_results_to_excel(model)
# print(f"[Download results]({filepath})")
