cmake_minimum_required(VERSION 3.15)
project(native LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Python 3.12 COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_DIR
)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_core src/native/main.cpp)
install(TARGETS _core LIBRARY DESTINATION native)