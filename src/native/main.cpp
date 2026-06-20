#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include "bpe.hpp"

void run_bpe_training(std::string filename) {
    bpe::run(filename);
}

NB_MODULE(_core, m) {
    m.def("run_bpe_training", &run_bpe_training);
}