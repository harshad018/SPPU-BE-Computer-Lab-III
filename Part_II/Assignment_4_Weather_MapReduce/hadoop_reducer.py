#!/usr/bin/env python3
"""
Hadoop Streaming Reducer – Weather MaxTemp per Year
Reads sorted key-value pairs from stdin (year TAB max_temp)
and outputs: year TAB avg_max_temp

Usage (standalone pipe test):
    cat weather_data.csv | python hadoop_mapper.py | sort | python hadoop_reducer.py
"""
import sys
from collections import defaultdict

totals = defaultdict(list)
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    parts = line.split("\t")
    if len(parts) != 2:
        continue
    try:
        year = int(parts[0])
        max_temp = float(parts[1])
        totals[year].append(max_temp)
    except ValueError:
        pass

for year in sorted(totals):
    vals = totals[year]
    avg = sum(vals) / len(vals)
    print(f"{year}\t{avg:.2f}")
