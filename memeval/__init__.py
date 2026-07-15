"""memeval — measure an agent memory layer before you trust it.

Rank and adversarial evals over pluggable adapters; every run is recorded to a
JSONL log that the replay TUI, charts, and runs listing all consume. Five
adapters ship (hindsight, mem0, letta, zep, example-rest); writing a new one is
~40 lines — see memeval/adapters/base.py and the README.
"""

__version__ = "0.2.0"
