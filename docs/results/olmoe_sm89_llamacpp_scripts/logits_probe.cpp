// Greedy decode with llama.cpp, printing the top-5 logits at every step.
// usage: logits_probe MODEL N_GEN PROMPT_ID...
// The prompt is one batch and every later token one decode call, as in
// llama-completion, so the kernels (and numerics) are those of generation.
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

static std::string piece(const llama_vocab *vocab, llama_token token) {
    char buf[128];
    int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
    std::string out;
    for (int i = 0; i < std::max(n, 0); i++) out += buf[i] == '\n' ? std::string("\\n") : std::string(1, buf[i]);
    return out;
}

int main(int argc, char **argv) {
    if (argc < 4) return 2;
    const int n_gen = atoi(argv[2]);
    std::vector<llama_token> tokens;
    for (int i = 3; i < argc; i++) tokens.push_back(atoi(argv[i]));
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    llama_model *model = llama_model_load_from_file(argv[1], mp);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 256;
    cp.n_batch = 256;
    llama_context *ctx = llama_init_from_model(model, cp);
    const llama_vocab *vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<int> idx(n_vocab);
    llama_token token;
    llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t)tokens.size());
    for (int step = 0; step < n_gen; step++) {
        if (llama_decode(ctx, batch)) { fprintf(stderr, "decode failed\n"); return 1; }
        const float *logits = llama_get_logits_ith(ctx, -1);
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                          [&](int a, int b) { return logits[a] > logits[b]; });
        printf("%d", step);
        for (int j = 0; j < 5; j++) printf("\t%d|%.4f|%s", idx[j], logits[idx[j]], piece(vocab, idx[j]).c_str());
        printf("\n");
        token = idx[0];
        batch = llama_batch_get_one(&token, 1);
    }
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
