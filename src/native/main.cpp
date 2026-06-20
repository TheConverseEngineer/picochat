#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include "bpe.hpp"
#include <iostream>
#include <string>

void train_tokenizer(const std::string& corpus, const std::string& output_path, int vocab_size) {
    std::string raw_text = bpe::read_file(corpus);
    bpe::BPETrainer trainer(raw_text, vocab_size); 
    trainer.vocab.write_to_file(output_path);
    std::cout << "Wrote output to " << output_path << "!" << std::endl;
}

NB_MODULE(_core, m) {
    m.def("train_tokenizer", &train_tokenizer);
}