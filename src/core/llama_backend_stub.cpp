// Fallback compiled when the llama.cpp submodule is absent: the project
// still builds, but loading a real model fails with a clear message.

#include "sovranx/core/llama_backend.hpp"
#include "sovranx/core/model.hpp"

namespace sovranx {

std::unique_ptr<LlamaBackend> make_llama_backend(const ModelParams&) {
    throw ModelError(
        "sovranx was built without llama.cpp; initialize the submodule "
        "(git submodule update --init --depth 1 third_party/llama.cpp) "
        "and rebuild");
}

}  // namespace sovranx
