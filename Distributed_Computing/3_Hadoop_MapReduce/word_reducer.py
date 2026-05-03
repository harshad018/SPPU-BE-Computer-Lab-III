#!/usr/bin/env python3
"""
Word Count Reducer  (Hadoop Streaming compatible)
=================================================
Reads sorted (word <TAB> count) pairs from stdin (the shuffle-sort phase
delivers them pre-sorted by key in Hadoop) and emits aggregated counts.

Usage – standalone test:
    cat sample_input.txt | python word_mapper.py | sort | python word_reducer.py

Usage – with Hadoop Streaming: see word_mapper.py header.
"""

import sys
from itertools import groupby


def main():
    """Aggregate word counts from sorted mapper output."""
    data = []
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            word, count = line.split("\t", 1)
            data.append((word, int(count)))
        except ValueError:
            continue

    # Sort by word (Hadoop's shuffle-sort does this automatically)
    data.sort(key=lambda x: x[0])

    for word, group in groupby(data, key=lambda x: x[0]):
        total = sum(item[1] for item in group)
        print(f"{word}\t{total}")


if __name__ == "__main__":
    main()
