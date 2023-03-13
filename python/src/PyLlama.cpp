#include "ggml.h"
#include "llama.h"
#include "utils.h"
#include <pybind11/pybind11.h>
#include <csignal>



void catch_signals() {
  auto handler = [](int code) { throw std::runtime_error("SIGNAL " + std::to_string(code)); };
  signal(SIGINT, handler);
  signal(SIGTERM, handler);
  signal(SIGKILL, handler);
}

namespace py = pybind11;

class PyLLAMA {
    llama_model model;
    gpt_vocab vocab;
    gpt_params params;
    std::mt19937 rng;
public:
    PyLLAMA() {
    }
    ~PyLLAMA() {
    }
    static PyLLAMA init(std::string model_path, gpt_params params) {
        PyLLAMA ret;
        if (!llama_model_load(model_path, ret.model, ret.vocab, 512)) {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, model_path.c_str());
            throw std::runtime_error("Failed to load model");
        }
        if (params.seed < 0) {
            params.seed = time(NULL);
        }
        printf("%s: seed = %d\n", __func__, params.seed);
        ret.rng = std::mt19937(params.seed);
        ret.params = params;

        return ret;
    }
    void predict(const std::string& text, int32_t n_predict = 128) {
        catch_signals();
        std::vector<gpt_vocab::id> embd_inp = ::llama_tokenize(vocab, text, true);
        params.n_predict = n_predict;
        llama_inference(model, rng, vocab, embd_inp, params);
    }
};

gpt_params init_params(
    std::string model,
    std::string prompt,
    int32_t n_predict,
    int32_t top_k,
    float top_p,
    float temp,
    float repeat_penalty
) {
    gpt_params params;
    params.n_predict = n_predict;
    params.top_k = top_k;
    params.top_p = top_p;
    params.temp = temp;
    params.repeat_penalty = repeat_penalty;
    params.model = model;
    params.prompt = prompt;
    return params;
}

// std::vector<gpt_vocab::id> llama_tokenize(const gpt_vocab & vocab, const std::string & text, bool bos);
PYBIND11_MODULE(llamacpp, m) {
    m.doc() = "Python bindings for C++ implementation of the LLaMA language model";
    py::class_<gpt_params>(m, "gpt_params")
        .def(py::init<>(&init_params))
        .def_readwrite("seed", &gpt_params::seed)
        .def_readwrite("n_threads", &gpt_params::n_threads)
        .def_readwrite("n_predict", &gpt_params::n_predict)
        .def_readwrite("repeat_last_n", &gpt_params::repeat_last_n)
        .def_readwrite("top_k", &gpt_params::top_k)
        .def_readwrite("top_p", &gpt_params::top_p)
        .def_readwrite("temp", &gpt_params::temp)
        .def_readwrite("repeat_penalty", &gpt_params::repeat_penalty)
        .def_readwrite("n_batch", &gpt_params::n_batch)
        .def_readwrite("model", &gpt_params::model)
        .def_readwrite("prompt", &gpt_params::prompt);
        
    py::class_<PyLLAMA>(m, "PyLLAMA")
        .def(py::init<>(&PyLLAMA::init))
        .def("init", &PyLLAMA::init, "Initialize the LLaMA model")
        .def("predict", &PyLLAMA::predict, "Predict the next token");
}
