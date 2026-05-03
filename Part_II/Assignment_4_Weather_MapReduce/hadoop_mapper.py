#!/usr/bin/env python3
"""
Hadoop Streaming Mapper – Weather MaxTemp per Year
Usage (standalone pipe test):
    cat weather_data.csv | python hadoop_mapper.py
Usage (Hadoop streaming):
    hadoop jar hadoop-streaming.jar -mapper hadoop_mapper.py ...
"""
import sys
import csv

reader = csv.DictReader(sys.stdin)
for row in reader:
    try:
        year = int(row["YEAR"])
        max_temp = float(row["MAX_TEMP_C"])
        print(f"{year}\t{max_temp}")
    except (ValueError, KeyError):
        pass
