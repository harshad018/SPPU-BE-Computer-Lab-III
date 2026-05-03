# Assignment 4 – Weather Analysis using MapReduce

## Overview
Finds the **hottest** and **coolest** year from weather station data using a
**MapReduce** pipeline implemented in pure Python.

The pipeline consists of three canonical phases:
```
Records (CSV rows)
    │
    ▼  Map phase     – extract (year, (max_temp, min_temp)) key-value pairs
    │
    ▼  Shuffle/Sort  – group all readings by year
    │
    ▼  Reduce phase  – compute average max/min per year
    │
    ▼  Analysis      – find hottest / coolest year
```
An optional `--workers` flag enables **parallel map phase** via `multiprocessing.Pool`.

## Data
- **Bundled sample** (`weather_data.csv`): 1990–2023, 3 stations × 34 years = 102 records.
- **Live NOAA data**: use `--download` to fetch a real NOAA station CSV (requires internet).

## Requirements
```
numpy
requests   (only needed for --download)
```
Install with:
```bash
pip install numpy requests
```

## Usage
```bash
# Default: use bundled sample data
python weather_mapreduce.py

# Use a custom CSV file
python weather_mapreduce.py --file /path/to/my_data.csv

# Download live NOAA data then process
python weather_mapreduce.py --download

# Parallel map phase with 4 workers
python weather_mapreduce.py --workers 4

# All options
python weather_mapreduce.py --help
```

## CSV Format
```
YEAR,STATION,MAX_TEMP_C,MIN_TEMP_C
1990,STATION_A,38.2,15.3
...
```

## Expected Output
```
============================================================
Weather Analysis – MapReduce Pipeline
============================================================

[1] Loading data from: weather_data.csv
    Loaded 102 records.

[2] Running MapReduce (workers=1) …

  Year  Avg Max °C  Avg Min °C
--------------------------------
  1990       37.47       15.13
  1991       36.40       14.50
  ...

============================================================
Results
============================================================
  🌡  Hottest Year: 2003  (avg max = 43.63 °C)
  ❄   Coolest Year: 1993  (avg max = 33.47 °C)
```

## Hadoop Streaming (optional)
For running on a real Hadoop cluster with Python streaming:
```bash
# mapper
cat weather_data.csv | python hadoop_mapper.py | sort | python hadoop_reducer.py

# or via Hadoop streaming jar
hadoop jar $HADOOP_HOME/share/hadoop/tools/lib/hadoop-streaming-*.jar \
  -mapper  "python hadoop_mapper.py" \
  -reducer "python hadoop_reducer.py" \
  -input   hdfs:///weather/input/ \
  -output  hdfs:///weather/output/
```
