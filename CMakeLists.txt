cmake_minimum_required(VERSION 3.4)
project(arrow_demo)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")


# Find pybind11
execute_process(
    COMMAND python3 -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(pybind11_DIR ${pybind11_cmake_dir})
find_package(pybind11 REQUIRED)

# Find Arrow & PyArrow
find_package(Arrow REQUIRED)
execute_process(COMMAND python3 -c "import pyarrow; print(pyarrow.get_include())"
    OUTPUT_VARIABLE PyArrow_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND python3 -c "import pyarrow; print(';'.join(pyarrow.get_library_dirs()))"
    OUTPUT_VARIABLE PyArrow_LIBRARY_DIRS
    OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND python3 -c "import pyarrow; print(';'.join(pyarrow.get_libraries()))"
    OUTPUT_VARIABLE PyArrow_LIBRARIES
    OUTPUT_STRIP_TRAILING_WHITESPACE)

# Add the pybind11 module
pybind11_add_module(my_module cmath.cc)

# Add the arrow_utils module
pybind11_add_module(arrow_utils arrow_utils.cc)
target_include_directories(arrow_utils PRIVATE ${PyArrow_INCLUDE_DIR} ${Arrow_INCLUDE_DIRS})
target_link_directories(arrow_utils PRIVATE ${PyArrow_LIBRARY_DIRS})
target_link_libraries(arrow_utils PRIVATE ${PyArrow_LIBRARIES})


# Add the main executable
add_executable(main main.cc)

# Link Arrow to the executable
target_link_libraries(main PRIVATE Arrow::arrow_shared)

# Include Arrow headers
target_include_directories(main PRIVATE ${Arrow_INCLUDE_DIRS})