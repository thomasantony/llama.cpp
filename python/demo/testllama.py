import llamacpp
import os

curdir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(curdir, "../../models/7B/ggml-model-q4_0.bin")

params = llamacpp.gpt_params(model_path,
    "Hi, I'm a llama.",
    4096,
    40,
    0.1,
    0.7,
    2.0)
model = llamacpp.PyLLAMA(model_path, params)
model.predict("Hello, I'm a llama.", 10)
