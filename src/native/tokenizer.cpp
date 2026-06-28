
#include "lib/ctre/ctre-unicode.hpp"
#include "tokenizer.hpp"
#include <iostream>
#include <fstream>

namespace tokenizer {

void replace_all(std::string& str, const std::string& from, const std::string& to) {
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length(); 
    }
}

Vocabulary::Vocabulary(const std::string& filename) {
    std::ifstream vocab_file(filename);

    if (!vocab_file.is_open()) {
        std::cerr << "[FATAL] vocab file not found" << std::endl;
    }

    for (bpe::token_t i = 0; i < 256; i++) {
        string_to_token[std::string(1, (char)i)] = i;
        token_to_string.push_back(std::string(1, (char)i));
    }

    std::string line;
    while (std::getline(vocab_file, line)) {
        std::istringstream iss(line);
        std::string tstr_a, tstr_b;
        iss >> tstr_a >> tstr_b;

        replace_all(tstr_a, "Ġ", " ");
        replace_all(tstr_b, "Ġ", " ");
        replace_all(tstr_a, "ġ", "\n");
        replace_all(tstr_b, "ġ", "\n");
        
        bpe::token_t token_a = string_to_token[tstr_a];
        bpe::token_t token_b = string_to_token[tstr_b];

        merge_rules.emplace_back(std::make_pair(token_a, token_b), token_to_string.size());
        std::string new_tstr = tstr_a + tstr_b;
        string_to_token[new_tstr] = token_to_string.size();
        token_to_string.push_back(new_tstr);
    }

    string_to_token["<|endofstring|>"] = token_to_string.size();
    token_to_string.push_back("<|endofstring|>");
}   


void Vocabulary::tokenize_pretoken(std::vector<bpe::token_t>& pretoken) {
    for (const auto& merge_rule : merge_rules) {
        for (size_t i = 0; i + 1 < pretoken.size(); i++) {
            if (std::make_pair(pretoken[i], pretoken[i+1]) == merge_rule.first) {
                pretoken[i+1] = token_to_string.size();
                pretoken[i] = merge_rule.second;
            }
        }
        std::erase(pretoken, token_to_string.size());
    }
}


void Vocabulary::pretokenize(const std::string& raw_string, std::vector<std::vector<bpe::token_t>>& pretokens) {
    for (auto segment : std::views::split(raw_string, "<|endoftext|>")) {
        for (auto& tokens : ctre::tokenize<PRETOKENIZE_REGEX>(segment)) {
            auto view = tokens.to_view();
            if (view.empty()) continue;

            pretokens.emplace_back(view.size());
            for (size_t i = 0; i < view.size(); i++) {
                pretokens.back()[i] = string_to_token[std::string(1, view[i])];
            }
        }
        pretokens.emplace_back(1, token_to_string.size() - 1);
    }
}


std::string Vocabulary::untokenize(std::span<bpe::token_t> tokens) {
    std::ostringstream oss;
    for (const bpe::token_t& token : tokens) {
        oss << token_to_string[token];
    }

    return oss.str();
}

}