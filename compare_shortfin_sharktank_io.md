python -m sharktank.examples.paged_llm_v1 --gguf-file=/tmp/sharktank/llama/open-llama-3b-v2-f16.gguf "1 2 3 4 5" --tokenizer-config-json /tmp/sharktank/llama/tokenizer_config.json

(./sharktank/sharktank/examples/paged_llm_v1.py)

this one works. generates 1 2 3 4 5 6 7 8 9 10 11 etc

shortfin/python/shortfin_apps/llm/server.py

this one doesn't work. generates </nd---------------------

I wonder if there's a difference between the tokenizer each one uses or the way the kv cache is managed or the way data is passed between python and IREE. Both are using IREE.

Could you help me figure out the dependencies of each one and how they connect to IREE? The goal is to eventually insert some debug statements to dump the tensors as NPY from the python code. Please issue one shell command at a time.

What have you learned so far? Include file paths and code snippets. Construct a dependency graph in mermaid if you'd like. Write them to /tmp/convo.md in preparation for our next conversation. Don't forget to include a description of the task at the end.

cat /tmp/convo.md


===========tokenizer comparison
Testing Sharktank Tokenizer:
Traceback (most recent call last):
  File "/home/xidaren2/SHARK-Platform/convos/tokenizer_test.py", line 40, in <module>
    test_sharktank_tokenizer(test_text)
  File "/home/xidaren2/SHARK-Platform/convos/tokenizer_test.py", line 19, in test_sharktank_tokenizer
    tokenizer = cli.get_tokenizer(cli.parse_args([]))
                                  ^^^^^^^^^^^^^^
AttributeError: module 'sharktank.utils.cli' has no attribute 'parse_args'
