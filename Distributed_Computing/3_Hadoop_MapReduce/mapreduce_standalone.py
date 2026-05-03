"""
Practical 3: Hadoop MapReduce – Standalone Python Simulation
============================================================
Implements the MapReduce programming model in pure Python to count
word occurrences and character occurrences in a text file.

This standalone script reproduces the Map → Shuffle/Sort → Reduce
pipeline without requiring a Hadoop cluster, making it easy to run
and understand in a lab environment.

For real Hadoop Streaming (requires a running Hadoop cluster):
    # Word count
    cat sample_input.txt | python word_mapper.py | sort | python word_reducer.py

    # Character count
    cat sample_input.txt | python char_mapper.py | sort | python char_reducer.py

Usage:
    python mapreduce_standalone.py                  # uses sample_input.txt
    python mapreduce_standalone.py path/to/file.txt

Expected output (partial):
    Word Count top entries: 'the' -> 6, 'distributed' -> 4, ...
    Char Count top entries: 'a' -> 72, 'e' -> 68, ...

Dependencies: None (Python stdlib only)
"""

import os
import sys
from itertools import groupby


# ---------------------------------------------------------------------------
# Mapper functions
# ---------------------------------------------------------------------------

def word_mapper(line: str):
    """Emit (word, 1) for every word in the line.

    Args:
        line: A raw text line.

    Yields:
        (word, 1) tuples.
    """
    for word in line.strip().lower().split():
        word = word.strip('.,!?;:"\'-()[]{}')
        if word:
            yield (word, 1)


def char_mapper(line: str):
    """Emit (char, 1) for every non-whitespace character in the line.

    Args:
        line: A raw text line.

    Yields:
        (char, 1) tuples.
    """
    for char in line.strip():
        if not char.isspace():
            yield (char, 1)


# ---------------------------------------------------------------------------
# Reducer function
# ---------------------------------------------------------------------------

def sum_reducer(key, values):
    """Sum all values for the given key.

    Args:
        key:    The grouping key (word or character).
        values: Iterable of integer counts.

    Returns:
        (key, total_count) tuple.
    """
    return (key, sum(values))


# ---------------------------------------------------------------------------
# MapReduce engine
# ---------------------------------------------------------------------------

def run_mapreduce(lines, mapper, reducer):
    """Execute a single MapReduce job in three phases.

    Phases:
        1. Map    – apply mapper to every input line, collect (key, value) pairs.
        2. Shuffle/Sort – sort all pairs by key.
        3. Reduce – group by key and apply reducer.

    Args:
        lines:   Iterable of input text lines.
        mapper:  Callable(line) -> iterable of (key, value) pairs.
        reducer: Callable(key, values_iter) -> (key, result).

    Returns:
        Sorted list of (key, result) tuples.
    """
    # --- Map phase ---
    mapped = []
    for line in lines:
        for kv in mapper(line):
            mapped.append(kv)

    # --- Shuffle & Sort phase ---
    mapped.sort(key=lambda x: x[0])

    # --- Reduce phase ---
    results = []
    for key, group in groupby(mapped, key=lambda x: x[0]):
        values = (v for _, v in group)
        results.append(reducer(key, values))

    return sorted(results, key=lambda x: x[0])


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_table(title: str, rows, col1: str, col2: str, top_n: int = None):
    """Print results as a formatted table.

    Args:
        title:  Section heading.
        rows:   Iterable of (key, count) pairs.
        col1:   Header label for the key column.
        col2:   Header label for the count column.
        top_n:  If set, only show the top_n most frequent entries.
    """
    print("\n" + "=" * 52)
    print(f"  {title}")
    print("=" * 52)
    if top_n:
        rows = sorted(rows, key=lambda x: x[1], reverse=True)[:top_n]
        rows = sorted(rows, key=lambda x: x[0])  # re-sort alphabetically
    print(f"  {col1:<28} {col2:>6}")
    print(f"  {'-'*36}")
    for key, count in rows:
        print(f"  {str(key):<28} {count:>6}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run word-count and character-count MapReduce jobs on an input file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = (sys.argv[1]
                  if len(sys.argv) > 1
                  else os.path.join(script_dir, "sample_input.txt"))

    print(f"--- Hadoop MapReduce Simulation ---")
    print(f"Input file: {input_file}\n")

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)

    # --- Job 1: Word Count ---
    word_counts = run_mapreduce(lines, word_mapper, sum_reducer)
    print_table("MapReduce Job 1: Word Count (all words)", word_counts,
                "Word", "Count")
    print(f"\n  Total unique words : {len(word_counts)}")
    print(f"  Total word tokens  : {sum(c for _, c in word_counts)}")

    # Show top-10 most frequent words
    top10 = sorted(word_counts, key=lambda x: x[1], reverse=True)[:10]
    print("\n  Top-10 most frequent words:")
    for word, count in top10:
        print(f"    {word:<25} {count}")

    # --- Job 2: Character Count ---
    char_counts = run_mapreduce(lines, char_mapper, sum_reducer)
    print_table("MapReduce Job 2: Character Count", char_counts,
                "Character", "Count")
    print(f"\n  Total unique characters : {len(char_counts)}")
    print(f"  Total character tokens  : {sum(c for _, c in char_counts)}")


if __name__ == "__main__":
    main()
