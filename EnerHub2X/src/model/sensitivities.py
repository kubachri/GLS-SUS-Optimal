# src/data/sensitivity.py

def apply_sensitivity_overrides(tech_df, data):
    """
    Modify tech_df and/or data dictionaries in-place based on desired sensitivity case.
    """
    # Example override
    if 'CO2Storage' in tech_df.index:
        tech_df.at['CO2Storage', 'VariableOmcost'] = 100

    # You can add more changes:
    # tech_df.at['Electrolyzer', 'StartupCost'] = 50000
    # data['price_buy'][('DK1', 'Electricity', 'T001')] = 200

    return tech_df, data