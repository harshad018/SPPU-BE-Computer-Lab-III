#!/usr/bin/env python3
"""
Character Count Reducer  (Hadoop Streaming compatible)
======================================================
Aggregates per-character counts from sorted mapper output.

Usage – standalone test:
    cat sample_input.txt | python char_mapper.py | sort | python char_reducer.py

Usage – with Hadoop Streaming: see char_mapper.py header.
"""

import sys
from itertools import groupby


def main():
    """Aggregate character counts from sorted mapper output."""
    data = []
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            char, count = line.split("\t", 1)
            data.append((char, int(count)))
        except ValueError:
            continue

    data.sort(key=lambda x: x[0])

    for char, group in groupby(data, key=lambda x: x[0]):
        total = sum(item[1] for item in group)
        print(f"{char!r}\t{total}")


if __name__ == "__main__":
    main()
