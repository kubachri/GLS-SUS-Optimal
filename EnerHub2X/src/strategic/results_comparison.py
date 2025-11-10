import pandas as pd
from pathlib import Path

def compare_results(results_dir: str = "results", base_name: str = "Results.xlsx", test_mode: bool = True):
    """
    Compare key outputs between centralized and strategic runs.
    Looks for:
        results/Results.xlsx
        results/Results-strategic.xlsx
    or if test mode:
        results/test_Results.xlsx
        results/test_Results-strategic.xlsx

    Key comparisons:
      - CO2 export price (carbon price)
      - CO2 sale quantities
      - Methanol sale quantities (if available)
    """
    results_dir = Path(results_dir)

    if test_mode:
        base_name = "test_" + base_name

    # --- locate files ---
    base_file = results_dir / base_name
    strat_file = results_dir / base_file.name.replace(".xlsx", "-strategic.xlsx")

    if not base_file.exists() or not strat_file.exists():
        raise FileNotFoundError(f"Could not find result files in {results_dir}")

    print(f"[INFO] Comparing results:\n  Centralized: {base_file.name}\n  Strategic:   {strat_file.name}")

    # --- Load relevant sheets ---
    df_base_A = pd.read_excel(base_file, sheet_name="ResultAsum")
    df_strat_A = pd.read_excel(strat_file, sheet_name="ResultAsum")

    # Normalize column names (safeguard)
    df_base_A.columns = [c.strip() for c in df_base_A.columns]
    df_strat_A.columns = [c.strip() for c in df_strat_A.columns]

    # --- Focus on Sale and Export_price_EUR for CO2 (and optionally Methanol) ---
    relevant_results = ["Sale", "Export_price_EUR"]
    energies = ["CO2", "Methanol"]

    # --- Detect available structure (some ResultAsum sheets lack "energy" column) ---
    if "energy" in df_base_A.columns:
        print("[INFO] Detected long-format ResultAsum (energy column present)")
        base_filt = df_base_A[df_base_A["Result"].isin(["Sale", "Export_price_EUR"])]
        strat_filt = df_strat_A[df_strat_A["Result"].isin(["Sale", "Export_price_EUR"])]
        merge_keys = ["Result", "area", "energy"]
        shared_energies = sorted(set(base_filt["energy"]) & set(strat_filt["energy"]))

    else:
        print("[INFO] Detected wide-format ResultAsum (energies as columns)")
        base_filt = df_base_A[df_base_A["Result"].isin(["Sale", "Export_price_EUR"])].copy()
        strat_filt = df_strat_A[df_strat_A["Result"].isin(["Sale", "Export_price_EUR"])].copy()
        merge_keys = ["Result", "area"]
        # automatically detect all fuels present in both
        shared_energies = sorted(
            set(base_filt.columns) & set(strat_filt.columns) - set(merge_keys)
        )

    print(f"[INFO] Detected shared energy columns: {shared_energies}")

    # --- Merge for comparison ---
    merged = pd.merge(
        base_filt, strat_filt,
        on=merge_keys, suffixes=("_central", "_strategic"), how="outer"
    )

    # --- Compute numeric deltas Δ for each fuel column ---
    for e in shared_energies:
        if f"{e}_central" in merged.columns and f"{e}_strategic" in merged.columns:
            merged[f"{e}_Δ"] = merged[f"{e}_strategic"] - merged[f"{e}_central"]

    # --- Sale quantity comparison (subset) ---
    sale_df = merged[merged["Result"] == "Sale"]
    print("\n[SUMMARY] Sale quantity comparison (Δ = strategic - central):")
    print(sale_df[merge_keys + [c for c in sale_df.columns if c.endswith('Δ')]].to_string(index=False))

    # --- Export price comparison ---
    price_df = merged[merged["Result"] == "Export_price_EUR"]
    print("\n[SUMMARY] Export price comparison (Δ = strategic - central):")
    print(price_df[merge_keys + [c for c in price_df.columns if c.endswith('Δ')]].to_string(index=False))

    # --- Optional: emphasize CO2 and Methanol ---
    for fuel in ["CO2", "Methanol"]:
        if any(f"{fuel}_Δ" in c for c in merged.columns):
            avg_sale_delta = sale_df[f"{fuel}_Δ"].mean(skipna=True)
            avg_price_delta = price_df[f"{fuel}_Δ"].mean(skipna=True)
            print(f"\n[HIGHLIGHT] {fuel}: Δ Sale = {avg_sale_delta:+.3f}, Δ Price = {avg_price_delta:+.3f}")

    # --- Save optional detailed comparison file ---
    output_path = results_dir / "Results_comparison.xlsx"
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        merged.to_excel(writer, sheet_name="KeyComparisons", index=False)
    print(f"\n[INFO] Comparison exported: {output_path.resolve()}")


compare_results()