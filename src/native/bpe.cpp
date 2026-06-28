#include "bpe.hpp"

#include "lib/ctre/ctre-unicode.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <numeric>
#include <chrono>
#include <ranges>

namespace bpe {

BPETrainerVocabulary::BPETrainerVocabulary() {
    for (token_t i = 0; i < 256; i++) {
        token_to_word.emplace_back(1, (char)i);
    }
}


void BPETrainerVocabulary::add_word(token_t new_token, token_t a, token_t b) {
    merge_rules.emplace_back(a, b, new_token);
    if (token_to_word.size() <= new_token) {
        token_to_word.resize(new_token + 1);
    }
    token_to_word[new_token] = token_to_word[a] + token_to_word[b];
}


/** Utility method to output a string to a stream while converting all spaces to Ġ */
void write_no_space(std::ofstream& stream, const std::string& str) {
    for (char c : str) {
        if (c == ' ') {
            stream << "Ġ";
        } else if (c == '\n') {
            stream << "ġ";
        } else {
            stream << c;
        }
    }
}


void BPETrainerVocabulary::write_to_file(const std::string& filename) {
    std::ofstream output_file(filename);
    if (!output_file) {
        std::cerr << "[FATAL]: Could not create or open output file" << std::endl;
    }

    for (const auto& merge : merge_rules) {
        write_no_space(output_file, token_to_word[merge.a]);
        output_file << " ";
        write_no_space(output_file, token_to_word[merge.b]);
        output_file << "\n";
    }
}


std::string read_file(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[WARNING] No file found at " << filepath << std::endl;
        return "";
    }

    auto filesize = std::filesystem::file_size(filepath);
    std::string content(filesize, '\0');
    file.read(content.data(), filesize);
    return content;
}


BPETrainer::BPETrainer(const std::string& corpus, token_t vocab_size) {
    // Populate wid_to_word and wid_frequency
    run_pretokenizer(corpus);

    // Populate remaining fields
    train_tokenizer(vocab_size);
}


void BPETrainer::run_pretokenizer(const std::string& corpus) {
    boost::unordered_flat_map<std::string_view, unsigned int> frequency_counts(1<<17);

    for (auto segment : std::views::split(corpus, "<|endoftext|>")) {
        for (auto& tokens : ctre::tokenize<PRETOKENIZE_REGEX>(segment)) {
            auto view = tokens.to_view();
            frequency_counts[view] += 1;
        }
    }

    for (const auto &[str, counts] : frequency_counts) {
        wid_to_word.emplace_back(str.begin(), str.end());
        wid_frequency.push_back(counts);
    }
}


/** Utility method to replace a pair of tokens with a new token */
void BPETrainer::replace_pair(
    unsigned int wid,
    token_pair_t pair_to_replace,
    token_t new_token,
    std::vector<token_t>& word
) {
    unsigned int frequency = wid_frequency[wid];
    for (size_t i = 0; i < word.size() - 1; i++) {
        token_pair_t pair = std::make_pair(word[i], word[i + 1]);
        pair_to_wid[pair].erase(wid);
        pair_frequency[pair] -= frequency;
    }

    for (size_t i = 0; i < word.size() - 1; i++) {
        token_pair_t pair = std::make_pair(word[i], word[i + 1]);
        if (pair == pair_to_replace) {
            word[i] = new_token;
            word[i+1] = new_token + 1;
            i++;
        }
    }
    std::erase(word, new_token + 1);

    for (size_t i = 0; i < word.size() - 1; i++) {
        token_pair_t pair = std::make_pair(word[i], word[i+1]);
        pair_to_wid[pair].emplace(wid);
        pair_frequency[pair] += frequency;
    }
}


void BPETrainer::train_tokenizer(token_t vocab_size) {
   
    for (int wid = 0; const auto &word_tokens : wid_to_word) {
        unsigned int frequency = wid_frequency[wid];
        for (size_t i = 0; i < word_tokens.size() - 1; i++) {
            token_pair_t pair = std::make_pair(word_tokens[i], word_tokens[i + 1]);
            pair_to_wid[pair].emplace(wid);
            pair_frequency[pair] += frequency;
        }
        wid += 1;
    }

    for (token_t new_token = 256; new_token < vocab_size; new_token++) {
        // Find most common pair
        auto mx = std::max_element(pair_frequency.begin(), pair_frequency.end(),
            [](const auto &p1, const auto &p2) {
                if (p1.second == p2.second) return p1.first < p2.first;
                return p1.second < p2.second;
            }
        );
        if (mx == pair_frequency.end()) {
            std::cout << "[Warning] Out of pairs to match!" << std::endl;
            break;
        }
        
        vocab.add_word(new_token, mx->first.first, mx->first.second);
        // Update words containing this pair
        auto pairs_to_update = pair_to_wid[mx->first];
        for (const auto &wid : pairs_to_update) {
            replace_pair(wid, mx->first, new_token, wid_to_word[wid]);
        }
    }
}

};
