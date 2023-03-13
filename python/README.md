## Building the Python bindings

### macOS

`brew install pybind11`

## Install python package

```
poetry install
python python/demo/testllama.py
```

## Get the model weights

You will need to obtain the weights for LLaMA yourself. There are a few torrents floating around as well as some huggingface repositories (e.g https://huggingface.co/nyanko7/LLaMA-7B/). Once you have them, copy them into the models folder.

```
ls ./models
65B 30B 13B 7B tokenizer_checklist.chk tokenizer.model
```

Convert the weights to GGML format using `convert-pth-to-ggml.py` and use the `llamacpp-quantize` command to quantize them into INT4. For example, for the 7B parameter model, run

```
python3 convert-pth-to-ggml.py models/7B/ 1
llamacpp-quantize ./models/7B/
```
## ToDo

[x] Use poetry to build package
[x] Add command line entry point for quantize script
[x] Publish wheel to PyPI
[ ] Add chat interface based on tinygrad
