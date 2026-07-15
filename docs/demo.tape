# NOTE: examples/corpus.jsonl is now 110+ docs — re-recording this tape needs a much longer Sleep after the rank command (retain + extraction takes minutes on stores with LLM extraction).
Output docs/demo.gif
Set FontSize 15
Set Width 1100
Set Height 620
Set Padding 12
Type "memeval rank --adapter hindsight --corpus examples/corpus.jsonl --gold examples/gold.jsonl"
Enter
Sleep 12s
Type "memeval replay $(ls -t runs/*.jsonl | head -1)"
Enter
Sleep 2s
Space
Sleep 1.5s
Space
Sleep 1.5s
Space
Sleep 1.5s
Space
Sleep 1.5s
Space
Sleep 2s
Type "q"
Sleep 1s
