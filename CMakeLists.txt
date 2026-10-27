cmake_minimum_required(VERSION 3.18)

project(MCML
  VERSION 1.4.0
  DESCRIPTION "High-Performance Monte Carlo Photon Transport in Multi-Layered Turbid Media (OpenMP & CUDA)"
  LANGUAGES C CXX
)

# Standard & Compiler Configuration
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ENABLE_OPENMP "Enable OpenMP multi-threading support" ON)
option(ENABLE_CUDA "Enable NVIDIA CUDA GPU acceleration if nvcc is available" ON)
option(BUILD_TESTING "Build unit and integration test suite" ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (Release, Debug, RelWithDebInfo)" FORCE)
endif()

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native -ffast-math -Wall -Wextra")
  set(CMAKE_C_FLAGS_DEBUG "-O0 -g -Wall -Wextra")
endif()

# Dependencies
find_library(MATH_LIBRARY m REQUIRED)

if(ENABLE_OPENMP)
  find_package(OpenMP)
endif()

# Optional NVIDIA CUDA Detection (Targets RTX 2060 Turing sm_75 by default)
set(MCML_CUDA_ENABLED FALSE)
if(ENABLE_CUDA)
  if(NOT DEFINED CMAKE_CUDA_COMPILER AND EXISTS "/usr/local/cuda/bin/nvcc")
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
  endif()
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES 75)
    endif()
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math")
    set(MCML_CUDA_ENABLED TRUE)
    message(STATUS "CUDA support enabled with compiler: ${CMAKE_CUDA_COMPILER} (arch: sm_${CMAKE_CUDA_ARCHITECTURES})")
  else()
    message(STATUS "CUDA compiler not found; building CPU-only (OpenMP) targets.")
  endif()
endif()

# Core Physics, Convolution & I/O Library
set(MCML_CORE_SOURCES
  src/mcml/mcmlgo.c
  src/mcml/mcmlio.c
  src/mcml/mcmlnr.c
  src/conv/convconv.c
  src/conv/extra.c
)

if(MCML_CUDA_ENABLED)
  list(APPEND MCML_CORE_SOURCES src/mcml/mcmlgpu.cu)
endif()

add_library(mcml_core STATIC ${MCML_CORE_SOURCES})
target_include_directories(mcml_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(mcml_core PUBLIC ${MATH_LIBRARY})

if(ENABLE_OPENMP AND OpenMP_C_FOUND)
  target_link_libraries(mcml_core PUBLIC OpenMP::OpenMP_C)
endif()

if(MCML_CUDA_ENABLED)
  target_compile_definitions(mcml_core PUBLIC ENABLE_CUDA=1)
  set_target_properties(mcml_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# Main Simulation & Finite-Beam Convolution Executables
add_executable(mcml src/mcml/mcmlmain.c)
target_link_libraries(mcml PRIVATE mcml_core)

add_executable(conv src/conv/convmain.c)
target_link_libraries(conv PRIVATE mcml_core)

# Automated Test Suite (CTest)
if(BUILD_TESTING)
  enable_testing()

  add_executable(test_physics tests/test_physics.c)
  target_link_libraries(test_physics PRIVATE mcml_core)

  add_test(
    NAME unit_physics_tests
    COMMAND test_physics
  )

  add_test(
    NAME integration_smoke_test_cpu
    COMMAND mcml --cpu --threads 4 ${CMAKE_CURRENT_SOURCE_DIR}/data/test.mci
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  if(MCML_CUDA_ENABLED)
    add_test(
      NAME integration_smoke_test_gpu
      COMMAND mcml --gpu ${CMAKE_CURRENT_SOURCE_DIR}/data/test.mci
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
  endif()

  add_test(
    NAME integration_finite_beam_conv
    COMMAND conv test.mco test_convolved.dat --beam gaussian --power 1.0 --radius 0.1
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_tests_properties(integration_finite_beam_conv PROPERTIES DEPENDS integration_smoke_test_cpu)
endif()
