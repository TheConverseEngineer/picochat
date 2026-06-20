#ifndef BPE_H
#define BPE_H

#include <string>
#include <vector>

namespace bpe {
    /** Datatype must be large enough to store all unique id's in vocab (ie. max vocab size <= MAX_INT) */
    using token_t = std::uint16_t;

    /** Represents a byte-pair encoding merge operation (a + b -> c) */
    struct Merge {
        token_t a, b, c;
    };

    /** Represents a token mapping generated using byte-pair encoding 
     * 
     * Note: this class is specifically designed for use with sequential tokens, as the 
     * underlying map is a vector, not a hashtable.
    */
    class Vocabulary {
        /** Map of token -> string it represents */
        std::vector<std::string> token_to_word;
        /** Chronological record of merge rules to apply in order to convert input text into tokens */
        std::vector<Merge> merge_rules;

    public:
        /** Constructor for the vocabulary class
         * Automatically initializes the first 256 valid bytes as direct-mapped tokens */
        Vocabulary();

        /** Add a new token to the vocabulary as the merge rule (a + b) -> new_token */
        void add_word(token_t new_token, token_t a, token_t b);

        /** Write vocabulary to a file */
        void write_to_file(const std::string& filename);
    };

    void run(const std::string filename);
}
#endif