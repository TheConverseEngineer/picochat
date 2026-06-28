#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include "bpe.hpp"
#include "tokenizer.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <ranges>

void train_tokenizer(const std::string& corpus, const std::string& output_path, int vocab_size) {
    std::string raw_text = bpe::read_file(corpus);
    bpe::BPETrainer trainer(raw_text, vocab_size); 
    trainer.vocab.write_to_file(output_path);
    std::cout << "Wrote output to " << output_path << "!" << std::endl;
}

void tokenize(const std::string& vocab_path, const std::string& text_path, const std::string& output_path) {
    tokenizer::Vocabulary vocab(vocab_path);
    std::string input_string = bpe::read_file(text_path);

    std::vector<std::vector<bpe::token_t>> tokens;
    vocab.pretokenize(input_string, tokens);
    std::cout << "Finished pretokenizing " << tokens.size() << " words!\n";

    for (auto& pretoken : tokens) vocab.tokenize_pretoken(pretoken);
    auto flattened = tokens | std::views::join | std::ranges::to<std::vector<bpe::token_t>>();
    
    std::cout << "Tokenized into " << flattened.size() << " tokens!\n";

    std::ofstream output_file(output_path);
    output_file.write(reinterpret_cast<const char*>(flattened.data()), flattened.size() * sizeof(bpe::token_t));
}

std::string untokenize(const std::string& vocab_path, const std::vector<bpe::token_t>& tokens) {
    tokenizer::Vocabulary vocab(vocab_path);

    return vocab.untokenize(tokens);
}

NB_MODULE(_core, m) {
    m.def("train_tokenizer", &train_tokenizer);
    m.def("tokenize", &tokenize);
    m.def("untokenize", &untokenize);
}