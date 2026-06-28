#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <boost/unordered/unordered_flat_map.hpp>
#include <bpe.hpp>
#include <tuple>
#include <string>
#include <vector>
#include <ranges>

namespace tokenizer {

    struct MergeOutput {
        bpe::token_t output;
        unsigned int merge_priority;
    };

    class Vocabulary {
        boost::unordered_flat_map<bpe::token_pair_t, MergeOutput> merge_rules;
        boost::unordered_flat_map<std::string, bpe::token_t> string_to_token;
        std::vector<std::string> token_to_string;

    public:
        Vocabulary(const std::string& filename);

        void tokenize_pretoken(std::vector<bpe::token_t>& pretoken);

        void pretokenize(const std::string& raw_string, std::vector<std::vector<bpe::token_t>>& pretokens);
        
        std::string untokenize(std::span<bpe::token_t> tokens);            
    };

}

#endif