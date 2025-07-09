cmake_minimum_required(VERSION 3.4)
project(my_module)

# Find pybind11
execute_process(
    COMMAND python3 -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(pybind11_DIR ${pybind11_cmake_dir})

find_package(pybind11 REQUIRED)

pybind11_add_module(my_module main.cpp)
