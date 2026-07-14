#pragma once

#include <memory>
#include <string>
#include <vector>

namespace reame::arca {

// Turns text into a vector in the embedding model's space. The real
// implementation loads a small embedding GGUF (mean-pooled); a mock drives
// the tests. Pure interface so the semantic cache carries no llama.cpp
// dependency of its own.
class Embedder {
public:
    virtual ~Embedder() = default;
    virtual std::vector<float> embed(const std::string& text) = 0;
};

// Loads an embedding model (pooling = mean) and returns an Embedder over
// it. Defined in the real backend TU; throws on load failure.
std::unique_ptr<Embedder> make_llama_embedder(const std::string& model_path,
                                              int threads);

}  // namespace reame::arca
