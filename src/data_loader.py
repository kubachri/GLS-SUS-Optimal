# src/data_loader.py

import os
import re
import pandas as pd
import numpy as np

# directory containing all the .inc files
INC_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'inc_data_GLS')
)

def read_set(name):
    """
    Read a GAMS set from <INC_DIR>/<name>.inc of the form
        Set name / A1 A2 A3 … / ;
    and return a Python list of the elements.
    """
    path = os.path.join(INC_DIR, f"{name}.inc")
    text = open(path, encoding='utf-8').read()
    m = re.search(r"/\s*(.+?)\s*/", text, re.DOTALL)
    if not m:
        raise ValueError(f"Could not parse set {name}.inc")
    return [tok for tok in re.split(r"[,\s]+", m.group(1).strip()) if tok]


def load_areas():
    """Wrapper around read_set for the Area set."""
    return read_set("Area")


def load_techdata():
    """
    Parse Techdata.inc into a pandas.DataFrame.
    Header line starts with 'Minimum' and columns are whitespace‐separated tokens.
    Data rows follow until the first blank line. Index = technology names.
    """
    path = os.path.join(INC_DIR, "Techdata.inc")
    with open(path, encoding='utf-8') as f:
        lines = f.readlines()

    header_idx = next(i for i, L in enumerate(lines) if L.strip().startswith("Minimum"))
    end_idx = next((i for i in range(header_idx+1, len(lines)) if not lines[i].strip()), len(lines))

    header_line = lines[header_idx].rstrip("\n")
    cols = header_line.split()
    starts = [header_line.index(c) for c in cols] + [len(header_line)]

    techs = []
    data_rows = []
    for raw in lines[header_idx+1 : end_idx]:
        if not raw.strip():
            continue
        techs.append(raw[:starts[0]].strip())
        row = []
        for j in range(len(cols)):
            piece = raw[starts[j] : starts[j+1]].strip()
            row.append(np.nan if piece=="" else float(piece))
        data_rows.append(row)

    df = pd.DataFrame(data_rows, index=techs, columns=cols)
    return df.fillna(0.0)


def load_carriermix():
    """
    Parse CarrierMix.inc into a DataFrame.
    Header has tokens like 'Import.Electricity', 'Export.Hydrogen', etc.
    Rows give fractions for each technology.
    """
    path = os.path.join(INC_DIR, "CarrierMix.inc")
    with open(path, encoding='utf-8') as f:
        lines = f.readlines()

    header_idx = next(i for i, L in enumerate(lines) if "Import." in L and "Export." in L)
    end_idx    = next((i for i in range(header_idx+1, len(lines)) if not lines[i].strip()), len(lines))

    header_line = lines[header_idx].rstrip("\n")
    cols        = header_line.split()
    starts      = [header_line.index(c) for c in cols] + [len(header_line)]

    techs = []
    data_rows = []
    for raw in lines[header_idx+1 : end_idx]:
        if not raw.strip():
            continue
        techs.append(raw[:starts[0]].strip())
        row = []
        for j in range(len(cols)):
            piece = raw[starts[j] : starts[j+1]].strip()
            row.append(0.0 if piece=="" else float(piece))
        data_rows.append(row)

    return pd.DataFrame(data_rows, index=techs, columns=cols)

def load_flowset():
    """
    Parse Flowset.inc → list of (area_from, area_to, energy) triples.
    """
    path = os.path.join(INC_DIR, "Flowset.inc")
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()

    # pick up both "/" and "/;" as delimiters
    boundaries = [i for i,L in enumerate(lines) if L.strip().startswith("/")]
    if len(boundaries) < 2:
        raise RuntimeError("Could not locate / … / block in Flowset.inc")
    start, end = boundaries[0]+1, boundaries[1]

    flowset = []
    for raw in lines[start:end]:
        if not raw.strip():
            continue
        parts = raw.split()
        # drop the literal "." tokens
        toks = [p for p in parts if p != "."]
        flowset.append(tuple(toks[:3]))

    return flowset

def load_demand():
    """
    Parse Demand.inc into a pandas.DataFrame.
    Columns   = DK1.<energy> …
    Index     = Hour-1, Hour-2, …
    All blanks → 0.0 floats.
    """
    path = os.path.join(INC_DIR, "Demand.inc")
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    # 1) find the real header line (first non‐comment containing 'DK1.')
    header_idx = next(
        i for i, L in enumerate(lines)
        if not L.lstrip().startswith("*") and "DK1." in L
    )
    header_line = lines[header_idx].rstrip("\n")
    columns     = header_line.strip().split()

    # 2) compute slice positions
    col_starts = [header_line.index(col) for col in columns] + [len(header_line)]

    # 3) gather data‐lines until first blank
    data_lines = []
    for raw in lines[header_idx+1:]:
        if not raw.strip():
            break
        data_lines.append(raw.rstrip("\n"))

    # 4) parse out times + numeric matrix
    times  = []
    matrix = []
    for raw in data_lines:
        times.append(raw[:col_starts[0]].strip())
        row = []
        for j in range(len(columns)):
            piece = raw[col_starts[j] : col_starts[j+1]].strip()
            row.append(0.0 if piece == "" else float(piece))
        matrix.append(row)

    return pd.DataFrame(matrix, index=times, columns=columns)

def load_interconnector_capacity():
    """
    Parse InterconnectorCapacity.inc into a pandas.DataFrame.
    Index = Hour-1, Hour-2, …
    Columns = <Area>.<Energy> (e.g. 'DK1.Electricity')
    """
    import os, re, pandas as pd

    path = os.path.join(INC_DIR, "InterconnectorCapacity.inc")
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()

    # find header
    header_idx = next(
        i for i, L in enumerate(lines)
        if not L.lstrip().startswith("*") and re.search(r"\w+\.\w+", L)
    )
    header_line = lines[header_idx].rstrip("\n")
    cols        = header_line.strip().split()
    starts      = [header_line.index(c) for c in cols] + [len(header_line)]

    # collect data lines
    times, data_rows = [], []
    for raw in lines[header_idx+1:]:
        if not raw.strip() or raw.strip().startswith("/"):
            break
        times.append(raw[:starts[0]].strip())
        row = []
        for j in range(len(cols)):
            piece = raw[starts[j]:starts[j+1]].strip()
            row.append(0.0 if piece=="" else float(piece))
        data_rows.append(row)

    return pd.DataFrame(data_rows, index=times, columns=cols)

def load_location_entries():
    """
    Read Location.inc and return a list of (area, tech) tuples.
    Location.inc looks like:

      SET Location(area,tech)
      /
      Skive  .  WindTurbine
      Skive  .  SolarPV
      …
      /;
    """
    path = os.path.join(INC_DIR, "Location.inc")
    lines = open(path, encoding="utf-8").read().splitlines()

    # find the two “/” delimiters
    start = next(i for i, L in enumerate(lines) if L.strip() == "/")
    end   = next(i for i in range(start+1, len(lines)) if lines[i].strip().startswith("/"))

    entries = []
    for raw in lines[start+1:end]:
        parts = raw.split()
        # expect exactly ["Area", ".", "Tech"]
        if len(parts) >= 3 and parts[1] == ".":
            area, tech = parts[0], parts[2]
            entries.append((area, tech))
    return entries

def load_price():
    """
    Parse Price.inc into a pandas.DataFrame.
    Columns look like 'DK1.Electricity.Import', 'DK1.Heat.Export', etc.
    Rows are Hour-1 … Hour-T.  Blanks → 0.0.
    """
    path = os.path.join(INC_DIR, "Price.inc")
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    # 1) header line is the first non-comment containing both 'Import' and 'Export'
    header_idx = next(
        i for i, L in enumerate(lines)
        if not L.lstrip().startswith("*") and "Import" in L and "Export" in L
    )
    header_line = lines[header_idx].rstrip("\n")
    cols = header_line.strip().split()

    # 2) compute slice positions
    starts = [header_line.index(c) for c in cols] + [len(header_line)]

    # 3) gather data lines until blank
    data_lines = []
    for raw in lines[header_idx+1:]:
        if not raw.strip():
            break
        data_lines.append(raw.rstrip("\n"))

    # 4) parse times + values
    times = []
    matrix = []
    for raw in data_lines:
        times.append(raw[:starts[0]].strip())
        row = []
        for j, col in enumerate(cols):
            piece = raw[starts[j]:starts[j+1]].strip()
            row.append(0.0 if piece == "" else float(piece))
        matrix.append(row)

    return pd.DataFrame(matrix, index=times, columns=cols)

def load_profile():
    """
    Parse Profile.inc → DataFrame indexed by Hour-*, columns = each tech name.
    """
    path = os.path.join(INC_DIR, "Profile.inc")
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    # find the header
    hdr = next(
        i for i, L in enumerate(lines)
        if L.strip()
        and not L.lstrip().startswith("*")
        and not L.lstrip().startswith("Table")
    )
    header_line = lines[hdr].rstrip("\n")
    cols        = header_line.split()
    starts      = [header_line.index(c) for c in cols] + [len(header_line)]

    times     = []
    data_rows = []
    for raw in lines[hdr+1:]:
        if not raw.strip() or raw.lstrip().startswith("*") or raw.strip().startswith("/"):
            break
        times.append(raw[:starts[0]].strip())
        row = []
        for j in range(len(cols)):
            piece = raw[starts[j]:starts[j+1]].strip()
            row.append(0.0 if piece == "" else float(piece))
        data_rows.append(row)

    return pd.DataFrame(data_rows, index=times, columns=cols)

def load_units():
    """
    Parse Units.inc (a GAMS SET of output‐field names) into a Python list.
    """
    path = os.path.join(INC_DIR, "Units.inc")
    text = open(path, encoding="utf-8").read()
    m = re.search(r"/\s*(.+?)\s*/", text, re.DOTALL)
    if not m:
        raise ValueError("Could not parse Units.inc")
    return [tok for tok in re.split(r"[,\s]+", m.group(1).strip()) if tok]

def load_resultset():
    return read_set("Resultset")

def load_data():
    """
    Glue‐together all of the above into the dict your Pyomo model expects:
      A, T, FlowSet,
      G, F, G_s,
      Capacity, Pmin, Rup, Rdown, Cvar, Cstart, SOC_max, SOC_init,
      sigma_in, sigma_out,
      Demand
    """
    data = {}

    # sets
    data['A']       = load_areas()
    data['T']       = read_set("Time")
    data['FlowSet'] = load_flowset()

     # --- location mapping from Location.inc ---
    loc = load_location_entries()
    # tech → area
    data['location'] = loc

    # area → list of tech: ensure every area appears, even if empty
    G_a = { a: [] for a in data['A'] }
    for area, tech in loc:
        if area not in G_a:
            # just in case but data['A'] should contain all areas
            G_a[area] = []
        G_a[area].append(tech)
    data['G_a'] = G_a

    # tech + fuels
    tech_df = load_techdata()
    cm_df   = load_carriermix()

    data['G']   = list(tech_df.index)
    data['F']   = sorted({c.split('.',1)[1] for c in cm_df.columns})
    data['G_s'] = [g for g in data['G'] if "Storage" in g]

    profile_df = load_profile()
    data['Profile'] = {
        (tech, time): float(profile_df.at[time, tech])
        for tech in profile_df.columns
        for time in profile_df.index
    }

    # techno‐economic params
    capacity = tech_df["Capacity"].to_dict()
    data['capacity']  = capacity
    # --- set up Pmin only when Minimum is a meaningful fraction (0 < Min < 1) ---
    data['Pmin'] = {}
    for g in data['G']:
        raw_min = tech_df.at[g, "Minimum"]  # the “Minimum” value from Techdata.inc
        if 0.0 < raw_min < 1.0:
            # treat as fraction of capacity
            data['Pmin'][g] = raw_min * capacity[g]
        else:
            # no enforced minimum for this technology
            data['Pmin'][g] = 0.0
    data['Rup']   = {g: tech_df.at[g,"RampRate"]  * capacity[g] for g in data['G']}
    data['Rdown'] = data['Rup'].copy()
    data['Cvar']   = tech_df["VariableOmcost"].to_dict()
    data['Cstart'] = tech_df["StartupCost"].to_dict()
    data['SOC_max']  = {g: tech_df.at[g,"StorageCap"]     * capacity[g] for g in data['G_s']}
    data['SOC_init'] = {g: tech_df.at[g,"InitialVolume"]  * capacity[g] for g in data['G_s']}

    # carrier‐mix → sigma_in / sigma_out
    sig_in, sig_out = {}, {}
    for tech in cm_df.index:
        for fuel in data['F']:
            imp = f"Import.{fuel}"
            exp = f"Export.{fuel}"
            sig_in [(tech,fuel)] = cm_df.at[tech,imp] if imp in cm_df.columns else 0.0
            sig_out[(tech,fuel)] = cm_df.at[tech,exp] if exp in cm_df.columns else 0.0
    data['sigma_in']  = sig_in
    data['sigma_out'] = sig_out

    dem_df = load_demand()
    demand = {}
    for col in dem_df.columns:
        area, energy = col.split(".", 1)
        for t in dem_df.index:
            demand[(area, energy, t)] = float(dem_df.at[t, col])
    data['Demand'] = demand

    data['Xcap'] = {}
    ic_df = load_interconnector_capacity()
    for col in ic_df.columns:
        area, energy = col.split(".", 1)
        for t in ic_df.index:
            data['Xcap'][(area, energy, t)] = float(ic_df.at[t, col])

    # 7) price_buy / price_sell from Price.inc
    price_df = load_price()
    price_buy  = {}
    price_sell = {}
    for col in price_df.columns:
        area, energy, direction = col.split(".", 2)
        for t in price_df.index:
            price = float(price_df.at[t, col])
            if direction == "Import":
                price_buy[(area, energy, t)] = price
            else:  # "Export"
                price_sell[(area, energy, t)] = price

    data["price_buy"]  = price_buy
    data["price_sell"] = price_sell

    data['Units'] = load_units()

    # map each storage unit to its fuel
    storage_fuel = {}
    for g in data['G_s']:
        candidates = [
            f for f in data['F']
            if data['sigma_in'].get((g,f),0.0)>0
            or data['sigma_out'].get((g,f),0.0)>0
        ]
        if not candidates:
            raise KeyError(f"No stored‐fuel found for storage tech '{g}'")
        storage_fuel[g] = candidates[0]

    data['storage_fuel'] = storage_fuel

    return data
