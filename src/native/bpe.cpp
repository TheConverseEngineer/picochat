#include "bpe.hpp"

#include <boost/unordered/unordered_flat_map.hpp>
#include <boost/unordered/unordered_flat_set.hpp>
#include "lib/ctre/ctre-unicode.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <numeric>
#include <chrono>
#include <ranges>

struct Timer {
    std::chrono::time_point<std::chrono::high_resolution_clock> st;
    std::chrono::duration<long long, std::nano> elapsed;
    Timer() {  }
    ~Timer() {
        std::cout << "Algorithm took: " << ((double)elapsed.count()) / 1'000'000'000.0 << "s\n";
    }
    void stop() {
        auto ct = std::chrono::high_resolution_clock::now();
        elapsed += ct - st;
    }
    void start() {
        st = std::chrono::high_resolution_clock::now();
    }
};

#define PRETOKENIZE_REGEX R"('s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)"

namespace bpe {

using string_counter_t = boost::unordered_flat_map<std::string_view, unsigned int>; 
using token_pair_t = std::pair<token_t, token_t>;


Vocabulary::Vocabulary() {
    for (token_t i = 0; i < 256; i++) {
        token_to_word.emplace_back(1, (char)i);
    }
}


void Vocabulary::add_word(token_t new_token, token_t a, token_t b) {
    merge_rules.emplace_back(a, b, new_token);
    if (token_to_word.size() <= new_token) {
        token_to_word.resize(new_token + 1);
    }
    token_to_word[new_token] = token_to_word[a] + token_to_word[b];
}


void write_no_space(std::ofstream& stream, const std::string& str) {
    for (char c : str) {
        if (c == ' ') {
            stream << "Ġ";
        } else {
            stream << c;
        }
    }
}


void Vocabulary::write_to_file(const std::string& filename) {
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

/** Utility method to quickly read the raw contents of a file into a string.
    * Returns the empty string if the file cannot be found */
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


string_counter_t pretokenize(const std::string& corpus) {
    // TODO: add the option to use differnt pretokenization regex's instead of
    // hardcoding the GPT-2 regex
    // TODO: deal with special characters (check opencode)

    std::cout << "starting\n";
    string_counter_t frequency_counts(1<<17);

    for (auto segment : std::views::split(corpus, "<|endoftext|>")) {
        for (auto& tokens : ctre::tokenize<R"('s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)">(segment)) {
            auto view = tokens.to_view();
            frequency_counts[view] += 1;
        }
    }
    return frequency_counts;
}


Vocabulary train_tokenizer(string_counter_t word_counts, token_t vocab_size) {
    Timer timer;
    timer.start();

    /** Map integer id -> "word" from training data (as a list of tokens)*/
    std::vector<std::vector<token_t>> wid_to_word; wid_to_word.reserve(word_counts.size());
    /** Map integer id -> number of times that "word" appears in training data*/
    std::vector<unsigned int> wid_frequency; wid_frequency.reserve(word_counts.size());
    for (const auto& [str, counts] : word_counts) {
        wid_to_word.emplace_back(str.begin(), str.end());
        wid_frequency.push_back(counts);
    }
    std::cout << "Total wid's: " << wid_to_word.size() << "\n";

    /** Maps token pair to word id's that contain that token */
    boost::unordered_flat_map<token_pair_t, boost::unordered_flat_set<int>> pair_to_wid(1<<16);
    /** Maps token pair to total appearances in training set */
    boost::unordered_flat_map<token_pair_t, unsigned int> pair_frequency(1 << 16);
    for (int wid = 0; const auto& word_tokens : wid_to_word) {
        unsigned int frequency = wid_frequency[wid];
        for (size_t i = 0; i < word_tokens.size() - 1; i++) {
            token_pair_t pair = std::make_pair(word_tokens[i], word_tokens[i+1]);
            pair_to_wid[pair].emplace(wid);
            pair_frequency[pair] += frequency;
        }

        wid += 1;
    }    
    
    Vocabulary vocab;
    for (token_t new_token = 256; new_token < vocab_size; new_token++) {
        // Find most common pair
        timer.start();
        auto mx = std::max_element(pair_frequency.begin(), pair_frequency.end(), 
            [](const auto& p1, const auto& p2) {
                if (p1.second == p2.second) {
                    if (p1.first.first == p2.first.first) {
                        return p1.first.second > p2.first.second;
                    } else return p1.first > p2.first;
                } else return p1.second < p2.second;
            }
        );
        timer.stop();

        if (mx == pair_frequency.end()) {
            std::cout << "[Warning] Out of pairs to match!" << std::endl;
            break;
        }
        vocab.add_word(new_token, mx->first.first, mx->first.second);
        
        // Update words containing this pair
        for (const auto& wid : pair_to_wid[mx->first]) {
            auto word = &wid_to_word[wid];
            unsigned int frequency = wid_frequency[wid];
            for (size_t i = 0; i < word->size() - 1; i++) {
                token_pair_t pair = std::make_pair(word->operator[](i), word->operator[](i+1));
                pair_to_wid[pair].erase(wid);
                pair_frequency[pair] -= frequency;
            } 

            for (size_t i = 0; i < word->size() - 1; i++) { 
                token_pair_t pair = std::make_pair(word->operator[](i), word->operator[](i+1)); 
                if (pair == mx->first) {
                    word->operator[](i) = new_token; 
                    word->operator[](i+1) = new_token + 1;
                }
            } 
            std::erase(*word, new_token+1);

            for (size_t i = 0; i < word->size() - 1; i++) {
                token_pair_t pair = std::make_pair(word->operator[](i), word->operator[](i+1));
                pair_to_wid[pair].emplace(wid);
                pair_frequency[pair] += frequency;
            }
        }
    }

    timer.stop();
    return vocab;
}

void run(const std::string filename) {
    std::string raw_text = read_file(filename);
    std::cout << "Finished reading file!" << std::endl;
    string_counter_t pretoken_counts = pretokenize(raw_text);

    std::cout << "Finished pretokenizing!" << std::endl;
    auto vocab = train_tokenizer(pretoken_counts, 500);

    std::cout << "Finished training tokenizer!" << std::endl;
    vocab.write_to_file("vocab.txt");
}

};
