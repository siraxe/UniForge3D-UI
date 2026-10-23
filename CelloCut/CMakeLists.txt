cmake_minimum_required(VERSION 3.18)
project(PROJ_CPP LANGUAGES CXX CUDA)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

# CGAL/GMP/MPFR root: -DCONDA_DIR=... or the CONDA_PREFIX environment var
# (conda-forge layout: <root>/Library/{include,lib} on Windows).
if(NOT DEFINED CONDA_DIR)
    if(DEFINED ENV{CONDA_PREFIX})
        set(CONDA_DIR "$ENV{CONDA_PREFIX}")
    else()
        set(CONDA_DIR "")
    endif()
endif()
if(WIN32 AND CONDA_DIR)
    set(CONDA_INCLUDE_DIR "${CONDA_DIR}/Library/include")
    set(CONDA_LIB_DIR "${CONDA_DIR}/Library/lib")
    list(APPEND CMAKE_PREFIX_PATH "${CONDA_DIR}/Library")
else()
    set(CONDA_INCLUDE_DIR "${CONDA_DIR}/include")
    set(CONDA_LIB_DIR "${CONDA_DIR}/lib")
    if(CONDA_DIR)
        list(APPEND CMAKE_PREFIX_PATH "${CONDA_DIR}")
    endif()
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# torch 2.13 headers need C++20; CUDA 13 dropped pre-Turing architectures and
# this machine's GPU is sm_120.
set(CMAKE_CUDA_ARCHITECTURES 86;90;120)
set(CMAKE_BUILD_TYPE Release)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(CGAL REQUIRED)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; from torch.utils.cpp_extension import include_paths; print(';'.join(include_paths()))"
    OUTPUT_VARIABLE TORCH_INCLUDE_DIRS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"
    OUTPUT_VARIABLE TORCH_LIB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

set(PYTHON_INCLUDE_DIR ${Python3_INCLUDE_DIRS})

find_library(GMP_LIB NAMES gmp HINTS "${CONDA_LIB_DIR}")
find_library(MPFR_LIB NAMES mpfr HINTS "${CONDA_LIB_DIR}")
find_library(TORCH_PYTHON_LIB NAMES torch_python HINTS ${TORCH_LIB_DIR} NO_DEFAULT_PATH)
find_library(TORCH_LIB        NAMES torch        HINTS ${TORCH_LIB_DIR} NO_DEFAULT_PATH)
find_library(TORCH_CPU_LIB    NAMES torch_cpu    HINTS ${TORCH_LIB_DIR} NO_DEFAULT_PATH)
find_library(TORCH_CUDA_LIB   NAMES torch_cuda   HINTS ${TORCH_LIB_DIR} NO_DEFAULT_PATH)
find_library(C10_LIB          NAMES c10          HINTS ${TORCH_LIB_DIR} NO_DEFAULT_PATH)
find_library(C10_CUDA_LIB     NAMES c10_cuda     HINTS ${TORCH_LIB_DIR} NO_DEFAULT_PATH)


# The bundled third_party/eigen (3.3.x) predates CUDA 13 and C++20 and does
# not compile under this toolchain; use the conda-forge Eigen instead.
set(ALL_INCLUDES
    ${PYTHON_INCLUDE_DIR}
    ${TORCH_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
    ${CONDA_INCLUDE_DIR}/eigen3
    ${PROJECT_SOURCE_DIR}/third_party/libigl/include
    ${PROJECT_SOURCE_DIR}
    ${CONDA_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

file(GLOB SRC_CU  ${PROJECT_SOURCE_DIR}/src/*.cu)
add_library(cuda_objs OBJECT ${SRC_CU})

target_include_directories(cuda_objs PRIVATE ${ALL_INCLUDES})

set_target_properties(cuda_objs PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(cuda_objs PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo -O3 -Xcompiler=/W0>
)

file(GLOB SRC_CPP ${PROJECT_SOURCE_DIR}/src/*.cpp)
pybind11_add_module(cppmodules ${SRC_CPP})
target_sources(cppmodules PRIVATE $<TARGET_OBJECTS:cuda_objs>)

target_include_directories(cppmodules PRIVATE ${ALL_INCLUDES})

target_link_libraries(cppmodules PRIVATE
    ${CUDAToolkit_LIBRARIES}
    ${GMP_LIB}
    ${MPFR_LIB}
    ${TORCH_PYTHON_LIB}
    ${TORCH_LIB}
    ${TORCH_CPU_LIB}
    ${TORCH_CUDA_LIB}
    ${C10_LIB}
    ${C10_CUDA_LIB}
    CGAL::CGAL
)

set_target_properties(cppmodules PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "86;90;120"
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build
    ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build
)

if(MSVC)
    target_compile_options(cppmodules PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W0 /O2>)
else()
    target_compile_options(cppmodules PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-w -O3>)
endif()