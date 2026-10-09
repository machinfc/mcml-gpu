cmake_minimum_required(VERSION 3.10)

project(MCML VERSION 1.2.2 LANGUAGES C)

# HPC Compilation Flags
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Optimization flags (Release by default)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_C_COMPILER_ID MATCHES "Clang")
    set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native -ffast-math -Wall -Wextra")
    set(CMAKE_C_FLAGS_DEBUG "-O0 -g -Wall -Wextra")
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Math library dependency
find_library(MATH_LIBRARY m)

# MCML Executable
set(MCML_SOURCES
    src/mcml/mcmlmain.c
    src/mcml/mcmlgo.c
    src/mcml/mcmlio.c
    src/mcml/mcmlnr.c
)
add_executable(mcml ${MCML_SOURCES})
target_link_libraries(mcml ${MATH_LIBRARY})

# Conv utilities were incomplete, omitted from this build.
