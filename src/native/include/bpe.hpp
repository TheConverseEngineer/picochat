#ifndef BPE_H
#define BPE_H

#include <string>

namespace bpe {
    /** Datatype must be large enough to store all unique id's in vocab (ie. max vocab size <= MAX_INT) */
    using token_t = std::uint16_t;

    void run(const std::string filename);
}
#endif