"""
Assignment 4 – Weather Analysis using MapReduce
================================================
Finds the hottest and coolest year from weather station data using a
**MapReduce** pipeline implemented in pure Python.

Data source
-----------
A sample CSV dataset (`weather_data.csv`) is bundled with the assignment.
It contains yearly max/min temperatures from multiple stations (1990-2023).

For live internet data, run with  --download  to fetch NOAA Global Surface
Summary of Day data and cache it locally.

Usage
-----
    python weather_mapreduce.py                    # use bundled sample data
    python weather_mapreduce.py --file my.csv      # custom CSV file
    python weather_mapreduce.py --download         # fetch live NOAA data
    python weather_mapreduce.py --workers 4        # parallel map phase
    python weather_mapreduce.py --help

Requirements
------------
    pip install numpy requests
"""

import argparse
import csv
import io
import multiprocessing
import os
import sys
from collections import defaultdict
from functools import reduce
from typing import List, Tuple, Dict


# ---------------------------------------------------------------------------
# Map phase
# ---------------------------------------------------------------------------

def mapper(record: dict) -> List[Tuple[int, Tuple[float, float]]]:
    """
    Input:  one CSV row dict with keys YEAR, MAX_TEMP_C, MIN_TEMP_C.
    Output: list of (year, (max_temp, min_temp)) key-value pairs.
    """
    try:
        year = int(record["YEAR"])
        max_t = float(record["MAX_TEMP_C"])
        min_t = float(record["MIN_TEMP_C"])
        return [(year, (max_t, min_t))]
    except (ValueError, KeyError):
        return []  # skip malformed records


# ---------------------------------------------------------------------------
# Shuffle & Sort phase (group by key)
# ---------------------------------------------------------------------------

def shuffle_sort(mapped: List[Tuple[int, Tuple[float, float]]]) \
        -> Dict[int, List[Tuple[float, float]]]:
    """Group values by year key."""
    grouped: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
    for key, value in mapped:
        grouped[key].append(value)
    return dict(grouped)


# ---------------------------------------------------------------------------
# Reduce phase
# ---------------------------------------------------------------------------

def reducer(year: int,
            readings: List[Tuple[float, float]]) -> Tuple[int, float, float]:
    """
    Input:  year, list of (max_temp, min_temp) tuples.
    Output: (year, yearly_avg_max, yearly_avg_min).
    """
    avg_max = sum(r[0] for r in readings) / len(readings)
    avg_min = sum(r[1] for r in readings) / len(readings)
    return (year, avg_max, avg_min)


# ---------------------------------------------------------------------------
# MapReduce runner
# ---------------------------------------------------------------------------

def run_mapreduce(records: List[dict], workers: int = 1) \
        -> List[Tuple[int, float, float]]:
    """Execute Map → Shuffle/Sort → Reduce pipeline."""

    # ---- Map phase ---------------------------------------------------------
    if workers > 1:
        with multiprocessing.Pool(processes=workers) as pool:
            map_results_nested = pool.map(mapper, records)
    else:
        map_results_nested = [mapper(r) for r in records]

    # Flatten list of lists
    mapped = [pair for sublist in map_results_nested for pair in sublist]

    # ---- Shuffle & Sort ----------------------------------------------------
    grouped = shuffle_sort(mapped)

    # ---- Reduce phase ------------------------------------------------------
    reduced = [reducer(year, readings)
               for year, readings in sorted(grouped.items())]

    return reduced


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv(filepath: str) -> List[dict]:
    """Load weather CSV into a list of row dicts."""
    records = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(dict(row))
    return records


def download_noaa_data(output_path: str = "weather_data_live.csv") -> str:
    """
    Download a small sample of NOAA Global Historical Climatology Network data.
    Falls back to bundled sample if network is unavailable.

    Returns the path to the data file to use.
    """
    try:
        import requests
        # NOAA public endpoint – annual temperature data for a few stations
        url = (
            "https://www.ncei.noaa.gov/data/global-summary-of-the-year/"
            "access/USW00094728.csv"
        )
        print(f"  Downloading NOAA data from:\n  {url}")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()

        # Parse NOAA CSV and reformat to our schema
        reader = csv.DictReader(io.StringIO(resp.text))
        rows = []
        for row in reader:
            year_val = row.get("DATE", "")
            tmax = row.get("TMAX", "")
            tmin = row.get("TMIN", "")
            if year_val and tmax and tmin:
                try:
                    rows.append({
                        "YEAR": str(int(year_val[:4])),
                        "STATION": row.get("STATION", "NOAA"),
                        "MAX_TEMP_C": str(round(float(tmax) / 10.0, 1)),  # tenths°C → °C
                        "MIN_TEMP_C": str(round(float(tmin) / 10.0, 1)),
                    })
                except (ValueError, TypeError):
                    continue

        if not rows:
            raise ValueError("No usable rows in NOAA response.")

        with open(output_path, "w", newline="", encoding="utf-8") as out:
            writer = csv.DictWriter(
                out, fieldnames=["YEAR", "STATION", "MAX_TEMP_C", "MIN_TEMP_C"]
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Saved {len(rows)} records to {output_path}")
        return output_path

    except Exception as exc:
        print(f"  [WARN] Download failed ({exc}). Using bundled sample data.")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Weather MapReduce – find hottest and coolest year."
    )
    parser.add_argument("--file", default=None,
                        help="Path to weather CSV (default: bundled weather_data.csv)")
    parser.add_argument("--download", action="store_true",
                        help="Download live NOAA data before processing")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel map workers (default: 1 = serial)")
    args = parser.parse_args()

    print("=" * 60)
    print("Weather Analysis – MapReduce Pipeline")
    print("=" * 60)

    # Determine data file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_csv = os.path.join(script_dir, "weather_data.csv")

    data_file = args.file or default_csv
    if args.download:
        live_path = download_noaa_data()
        if live_path:
            data_file = live_path

    print(f"\n[1] Loading data from: {data_file}")
    records = load_csv(data_file)
    print(f"    Loaded {len(records)} records.")

    print(f"\n[2] Running MapReduce (workers={args.workers}) …")
    results = run_mapreduce(records, workers=args.workers)

    # ---- Display per-year summary
    print(f"\n{'Year':>6}  {'Avg Max °C':>10}  {'Avg Min °C':>10}")
    print("-" * 32)
    for year, avg_max, avg_min in results:
        print(f"{year:>6}  {avg_max:>10.2f}  {avg_min:>10.2f}")

    # ---- Find hottest / coolest year
    hottest = max(results, key=lambda x: x[1])
    coolest = min(results, key=lambda x: x[1])

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"  🌡  Hottest Year: {hottest[0]}  (avg max = {hottest[1]:.2f} °C)")
    print(f"  ❄   Coolest Year: {coolest[0]}  (avg max = {coolest[1]:.2f} °C)")
    print()


if __name__ == "__main__":
    main()
