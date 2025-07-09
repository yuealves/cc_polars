cmake_minimum_required(VERSION 3.4)
project(arrow_demo)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# Find pybind11
execute_process(
    COMMAND python3 -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(pybind11_DIR ${pybind11_cmake_dir})
find_package(pybind11 REQUIRED)

# Find Arrow
find_package(Arrow REQUIRED)

# Add the pybind11 module
pybind11_add_module(my_module cmath.cc)

# Add the main executable
add_executable(main main.cc)

# Link Arrow to the executable
target_link_libraries(main PRIVATE Arrow::arrow_shared)
