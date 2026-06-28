#ifndef BPE_H
#define BPE_H

#include <string>
#include <vector>
#include <boost/unordered/unordered_flat_map.hpp>
#include <boost/unordered/unordered_flat_set.hpp>

// TODO: have a mechanism for supporting multiple regex's here?
#define PRETOKENIZE_REGEX R"('s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)"

namespace bpe {
    /** Datatype must be large enough to store all unique id's in vocab (ie. max vocab size <= MAX_INT) */
    using token_t = std::uint16_t;

    using token_pair_t = std::pair<token_t, token_t>;

    /** Represents a byte-pair encoding merge operation (a + b -> c) */
    struct Merge {
        token_t a, b, c;
    };

    /** Represents a token mapping generated using byte-pair encoding 
     * 
     * Note: this class is specifically designed for use with sequential tokens, as the 
     * underlying map is a vector, not a hashtable.
    */
    class BPETrainerVocabulary {
        /** Map of token -> string it represents */
        std::vector<std::string> token_to_word;
        /** Chronological record of merge rules to apply in order to convert input text into tokens */
        std::vector<Merge> merge_rules;

    public:
        /** Constructor for the vocabulary class
         * Automatically initializes the first 256 valid bytes as direct-mapped tokens */
        BPETrainerVocabulary();

        /** Add a new token to the vocabulary as the merge rule (a + b) -> new_token */
        void add_word(token_t new_token, token_t a, token_t b);

        /** Write vocabulary to a file */
        void write_to_file(const std::string& filename);
    };

    /** Utility class for training a byte-pair encoding tokenizer */
    class BPETrainer {
        /** Map integer id -> "word" from training data (as a list of tokens)*/
        std::vector<std::vector<token_t>> wid_to_word; 
        /** Map integer id -> number of times that "word" appears in training data*/
        std::vector<unsigned int> wid_frequency; 

        /** Maps token pair to word id's that contain that token */
        boost::unordered_flat_map<token_pair_t, boost::unordered_flat_set<int>> pair_to_wid;
        /** Maps token pair to total appearances in training set */
        boost::unordered_flat_map<token_pair_t, unsigned int> pair_frequency;

        /** Utility method to be used once (and only once) during initializations
         * Pretokenizes the given string by removing all EOS markers and splitting the 
         * remaining text into pretokens 
         *
         * Populates wid_to_word and wid_frequency
         */
        void run_pretokenizer(const std::string& corpus);

        /** Utility method to be used once (and only once) during initializations
         * Trains a byte-pair encoding tokenizer given pretoken counts and a desired vocab size 
         * 
         * Populates pair_to_wid and pair_frequency
         */
        void train_tokenizer(token_t vocab_size);
        
        
        void replace_pair(unsigned int wid, token_pair_t pair_to_replace, token_t new_token, std::vector<token_t>& word);

    public:
        /** Train a byte-pair encoding tokenizer on the given corpus with a specified vocab size
         *
         * Notes:
         *  - The vocab size includes the 256 base bytes
         *  - Any <|endoftext|> markers in the corpus will be stripped out and used as separators
         *  - No special tokens will be included in the generated vocabulary
         */
        BPETrainer(const std::string& corpus, token_t vocab_size);

        /** Generated vocab after training */
        BPETrainerVocabulary vocab;
    };

    /** Utility method to quickly read a file into a single string */
    std::string read_file(const std::string& filepath);

    
}
#endif