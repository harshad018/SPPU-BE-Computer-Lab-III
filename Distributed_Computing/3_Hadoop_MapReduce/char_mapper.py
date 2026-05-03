#!/usr/bin/env python3
"""
Character Count Mapper  (Hadoop Streaming compatible)
=====================================================
Reads lines from stdin and emits one (char <TAB> 1) pair for every
non-whitespace character encountered.

Usage – standalone test:
    echo "hello" | python char_mapper.py

Usage – with Hadoop Streaming:
    hadoop jar $HADOOP_HOME/share/hadoop/tools/lib/hadoop-streaming-*.jar \\
        -input  /user/hadoop/input \\
        -output /user/hadoop/char_output \\
        -mapper  char_mapper.py \\
        -reducer char_reducer.py \\
        -file    char_mapper.py \\
        -file    char_reducer.py
"""

import sys


def main():
    """Emit (char, 1) for every non-whitespace character read from stdin."""
    for line in sys.stdin:
        for char in line.strip():
            if not char.isspace():
                print(f"{char}\t1")


if __name__ == "__main__":
    main()
