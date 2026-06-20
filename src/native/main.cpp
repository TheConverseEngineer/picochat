#include <nanobind/nanobind.h>

int add(int a, int b) {
    return a + b + 105;
}

NB_MODULE(_core, m) {
    m.def("add", &add);
}