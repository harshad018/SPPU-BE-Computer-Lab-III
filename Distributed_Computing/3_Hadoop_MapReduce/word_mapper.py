#!/usr/bin/env python3
"""
Word Count Mapper  (Hadoop Streaming compatible)
================================================
Reads lines from stdin, tokenises each line, and emits one
(word <TAB> 1) key-value pair per word to stdout.

Usage – standalone test:
    echo "hello world hello" | python word_mapper.py

Usage – with Hadoop Streaming:
    hadoop jar $HADOOP_HOME/share/hadoop/tools/lib/hadoop-streaming-*.jar \\
        -input  /user/hadoop/input \\
        -output /user/hadoop/output \\
        -mapper  word_mapper.py \\
        -reducer word_reducer.py \\
        -file    word_mapper.py \\
        -file    word_reducer.py
"""

import sys


def map_words(line: str):
    """Tokenise a line and emit (word, 1) key-value pairs.

    Args:
        line: A raw input line of text.
    """
    for word in line.strip().lower().split():
        # Strip common punctuation so "word," and "word" count as the same
        word = word.strip('.,!?;:"\'-()[]{}')
        if word:
            print(f"{word}\t1")


def main():
    """Read lines from stdin and emit word-count pairs to stdout."""
    for line in sys.stdin:
        map_words(line)


if __name__ == "__main__":
    main()
