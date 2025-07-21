# src/data/loader.py

import os
import re
import pandas as pd
import numpy as np

# Will be set at runtime by load_data(cfg)
INC_DIR = None


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

    techs, data_rows = [], []
    for raw in lines[header_idx+1 : end_idx]:
        if not raw.strip():
            continue
        techs.append(raw[:starts[0]].strip())
        row = [
            np.nan if raw[starts[j]:starts[j+1]].strip()=="" else float(raw[starts[j]:starts[j+1]].strip())
            for j in range(len(cols))
        ]
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

    # find header and end
    header_idx = next(i for i, L in enumerate(lines) 
                      if "Import." in L and "Export." in L)
    end_idx = next((i for i in range(header_idx+1, len(lines)) 
                    if not lines[i].strip()), len(lines))

    # grab the raw header line
    header_line = lines[header_idx].rstrip("\n")

    # --- NEW: find every run of non-whitespace and its start position ---
    matches = list(re.finditer(r'\S+', header_line))
    cols    = [m.group() for m in matches]
    starts  = [m.start()    for m in matches] + [len(header_line)]

    techs, data_rows = [], []
    for raw in lines[header_idx+1 : end_idx]:
        if not raw.strip():
            continue
        # tech name is everything to the first column start
        techs.append(raw[:starts[0]].strip())
        row = []
        for j in range(len(cols)):
            piece = raw[starts[j] : starts[j+1]].strip()
            row.append(0.0 if piece == "" else float(piece))
        data_rows.append(row)

    return pd.DataFrame(data_rows, index=techs, columns=cols)

# def load_carriermix():
#     """
#     Parse CarrierMix.inc into a DataFrame.
#     Header has tokens like 'Import.Electricity', 'Export.Hydrogen', etc.
#     Rows give fractions for each technology.
#     """
#     path = os.path.join(INC_DIR, "CarrierMix.inc")
#     with open(path, encoding='utf-8') as f:
#         lines = f.readlines()

#     header_idx = next(i for i, L in enumerate(lines) if "Import." in L and "Export." in L)
#     end_idx    = next((i for i in range(header_idx+1, len(lines)) if not lines[i].strip()), len(lines))

#     header_line = lines[header_idx].rstrip("\n")
#     cols        = header_line.split()
#     starts      = [header_line.index(c) for c in cols] + [len(header_line)]

#     techs, data_rows = [], []
#     for raw in lines[header_idx+1 : end_idx]:
#         if not raw.strip(): continue
#         techs.append(raw[:starts[0]].strip())
#         row = []
#         for j in range(len(cols)):
#             piece = raw[starts[j] : starts[j+1]].strip()
#             row.append(0.0 if piece=="" else float(piece))
#         data_rows.append(row)

#     return pd.DataFrame(data_rows, index=techs, columns=cols)


def load_flowset():
    """
    Parse Flowset.inc → list of (area_from, area_to, energy) triples.
    """
    path = os.path.join(INC_DIR, "Flowset.inc")
    lines = open(path, encoding='utf-8').read().splitlines()

    boundaries = [i for i,L in enumerate(lines) if L.strip().startswith("/")]
    if len(boundaries) < 2:
        raise RuntimeError("Could not locate / … / block in Flowset.inc")
    start, end = boundaries[0]+1, boundaries[1]

    flowset = []
    for raw in lines[start:end]:
        if not raw.strip(): continue
        parts = raw.split()
        toks = [p for p in parts if p != "."]
        flowset.append(tuple(toks[:3]))

    return flowset


def load_demand():
    """
    Parse Demand.inc into a pandas.DataFrame (Area.Energy columns, Hour-* index).
    Blanks become 0.0 floats.
    """
    path = os.path.join(INC_DIR, "Demand.inc")
    with open(path, encoding='utf-8') as f:
        lines = f.readlines()

    header_idx = next(i for i, L in enumerate(lines) if not L.lstrip().startswith("*") and "DK1." in L)
    header_line = lines[header_idx].rstrip("\n")
    columns     = header_line.strip().split()
    col_starts  = [header_line.index(col) for col in columns] + [len(header_line)]

    times, matrix = [], []
    for raw in lines[header_idx+1:]:
        if not raw.strip(): break
        times.append(raw[:col_starts[0]].strip())
        row = [
            0.0 if raw[col_starts[j]:col_starts[j+1]].strip()=="" else float(raw[col_starts[j]:col_starts[j+1]].strip())
            for j in range(len(columns))
        ]
        matrix.append(row)

    return pd.DataFrame(matrix, index=times, columns=columns)


def load_interconnector_capacity():
    """
    Parse InterconnectorCapacity.inc into a pandas.DataFrame.
    Index = Hour-*, Columns = Area.Energy
    """
    path = os.path.join(INC_DIR, "InterconnectorCapacity.inc")
    with open(path, encoding='utf-8') as f:
        lines = f.read().splitlines()

    header_idx = next(i for i, L in enumerate(lines) if not L.lstrip().startswith("*") and re.search(r"\w+\.\w+", L))
    header_line = lines[header_idx].rstrip("\n")
    cols        = header_line.strip().split()
    starts      = [header_line.index(c) for c in cols] + [len(header_line)]

    times, data_rows = [], []
    for raw in lines[header_idx+1:]:
        if not raw.strip() or raw.strip().startswith("/"): break
        times.append(raw[:starts[0]].strip())
        row = []
        for j in range(len(cols)):
            piece = raw[starts[j]:starts[j+1]].strip()
            row.append(0.0 if piece=="" else float(piece))
        data_rows.append(row)

    return pd.DataFrame(data_rows, index=times, columns=cols)


def load_location_entries():
    """
    Read Location.inc and return list of (area, tech) tuples.
    """
    path = os.path.join(INC_DIR, "Location.inc")
    lines = open(path, encoding='utf-8').read().splitlines()

    start = next(i for i, L in enumerate(lines) if L.strip() == "/")
    end   = next(i for i in range(start+1, len(lines)) if lines[i].strip().startswith("/"))

    entries = []
    for raw in lines[start+1:end]:
        parts = raw.split()
        if len(parts) >= 3 and parts[1] == ".":
            entries.append((parts[0], parts[2]))
    return entries


def load_price():
    """
    Parse Price.inc → DataFrame of import/export prices.
    """
    path = os.path.join(INC_DIR, "Price.inc")
    with open(path, encoding='utf-8') as f:
        lines = f.readlines()

    header_idx = next(i for i, L in enumerate(lines) if not L.lstrip().startswith("*") and "Import" in L and "Export" in L)
    header_line = lines[header_idx].rstrip("\n")
    cols        = header_line.strip().split()
    starts      = [header_line.index(c) for c in cols] + [len(header_line)]

    times, matrix = [], []
    for raw in lines[header_idx+1:]:
        if not raw.strip(): break
        times.append(raw[:starts[0]].strip())
        row = [
            0.0 if raw[starts[j]:starts[j+1]].strip()=="" else float(raw[starts[j]:starts[j+1]].strip())
            for j in range(len(cols))
        ]
        matrix.append(row)

    return pd.DataFrame(matrix, index=times, columns=cols)


def load_profile():
    """
    Parse Profile.inc → DataFrame indexed by Hour-*, columns = each tech.
    Handles wrapped rows via stitching, and computes true column spans
    via regex rather than header.index().
    """
    path = os.path.join(INC_DIR, "Profile.inc")
    with open(path, encoding='utf-8') as f:
        lines = f.readlines()

    # 1) Find header line
    hdr = next(i for i, L in enumerate(lines)
               if L.strip() and not L.lstrip().startswith("*") and not L.lstrip().startswith("Table"))
    header_line = lines[hdr].rstrip("\n")

    # 2) Build cols + exact start/end spans via regex
    matches = list(re.finditer(r"\S+", header_line))
    cols   = [m.group() for m in matches]
    starts = [m.start() for m in matches]
    ends   = [m.end()   for m in matches] + [len(header_line)]

    # 3) Stitch wrapped rows
    times          = []
    data_str_rows  = []
    current_time   = None
    current_buffer = ""

    for raw in lines[hdr+1:]:
        if not raw.strip() or raw.lstrip().startswith("*") or raw.strip().startswith("/"):
            break

        prefix = raw[:starts[0]]
        rest   = raw[starts[0]:].rstrip("\n")

        if prefix.strip():
            # new logical row
            if current_time is not None:
                data_str_rows.append(current_buffer)
            current_time   = prefix.strip()
            times.append(current_time)
            current_buffer = rest
        else:
            # continuation
            current_buffer += rest

    # append last
    if current_time is not None:
        data_str_rows.append(current_buffer)

    # 4) Slice each logical row into floats
    data_rows = []
    for row_str in data_str_rows:
        row = []
        for j in range(len(cols)):
            cell = row_str[starts[j]-starts[0] : ends[j]-starts[0]].strip()
            row.append(0.0 if cell == "" else float(cell))
        data_rows.append(row)

    return pd.DataFrame(data_rows, index=times, columns=cols)

def load_units():
    """
    Parse Units.inc into a Python list of names.
    """
    path = os.path.join(INC_DIR, "Units.inc")
    text = open(path, encoding='utf-8').read()
    m = re.search(r"/\s*(.+?)\s*/", text, re.DOTALL)
    if not m:
        raise ValueError("Could not parse Units.inc")
    return [tok for tok in re.split(r"[,\s]+", m.group(1).strip()) if tok]


def load_data(cfg):
    """
    Glue together all loads into the dict expected by the model.
    First sets INC_DIR based on cfg.data_dir.
    """
    global INC_DIR
    INC_DIR = cfg.data_dir
    if not os.path.isdir(INC_DIR):
        raise FileNotFoundError(f"Could not find data folder: {INC_DIR}")

    data = {}
    # sets
    data['A']       = load_areas()
    data['T']       = read_set("Time")
    data['FlowSet'] = load_flowset()

    # location mapping
    data['location'] = load_location_entries()

    # tech & fuels
    tech_df = load_techdata()
    cm_df   = load_carriermix()
    data['G']   = list(tech_df.index)
    data['F']   = sorted({c.split('.',1)[1] for c in cm_df.columns})
    data['G_s'] = [g for g in data['G'] if "Storage" in g]

    # time-series
    prof_df = load_profile()
    data['Profile'] = {
        (tech, time): float(prof_df.at[time, tech])
        for tech in prof_df.columns
        for time in prof_df.index
    }

    dem_df = load_demand()
    data['Demand'] = {
        (area, fuel, t): float(dem_df.at[t, col])
        for col in dem_df.columns
        for (area, fuel) in [col.split('.',1)]
        for t in dem_df.index
    }

    price_df = load_price()
    data['price_buy'], data['price_sell'] = {}, {}
    for col in price_df.columns:
        area, energy, direction = col.split('.',2)
        for t in price_df.index:
            price = float(price_df.at[t, col])
            if direction == 'Import':
                data['price_buy'][(area, energy, t)] = price
            else:
                data['price_sell'][(area, energy, t)] = price

    ic_df = load_interconnector_capacity()
    data['Xcap'] = {
        (area, energy, t): float(ic_df.at[t, col])
        for col in ic_df.columns
        for (area, energy) in [col.split('.',1)]
        for t in ic_df.index
    }

    # techno-economic
    data['capacity'] = tech_df['Capacity'].to_dict()
    data['Cvar']     = tech_df['VariableOmcost'].to_dict()
    data['Cstart']   = tech_df['StartupCost'].to_dict()

    # carrier-mix
    sig_in, sig_out = {}, {}
    for tech in cm_df.index:
        for fuel in data['F']:
            imp = f"Import.{fuel}"
            exp = f"Export.{fuel}"
            sig_in[(tech, fuel)]  = cm_df.at[tech, imp] if imp in cm_df.columns else 0.0
            sig_out[(tech, fuel)] = cm_df.at[tech, exp] if exp in cm_df.columns else 0.0
    data['sigma_in'], data['sigma_out'] = sig_in, sig_out

    # SOC init/max from tech_df
    data['SOC_init'] = {g: tech_df.at[g, 'InitialVolume'] for g in data['G_s']}
    data['SOC_max']  = {g: tech_df.at[g, 'StorageCap']     for g in data['G_s']}

    # units & storage fuel mapping
    data['Units'] = load_units()
    data['storage_fuel'] = {
        g: next(f for f in data['F'] if sig_in.get((g,f),0)+sig_out.get((g,f),0)>0)
        for g in data['G_s']
    }

    return data
