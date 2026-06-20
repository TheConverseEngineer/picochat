cmake_minimum_required(VERSION 3.15)
set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")

set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS_DEBUG "-O0")

project(native LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Python 3.12 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(Boost CONFIG REQUIRED COMPONENTS headers)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_DIR
)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_core 
    src/native/main.cpp
    src/native/bpe.cpp
)

target_include_directories(_core PRIVATE
    src/native/include
    src/lib/ctre
)
target_link_libraries(_core PRIVATE Boost::boost)
target_compile_options(_core PRIVATE -Wunknown-escape-sequence)

install(TARGETS _core LIBRARY DESTINATION native)