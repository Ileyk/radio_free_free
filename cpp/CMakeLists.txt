cmake_minimum_required(VERSION 3.21)
project(radio_free_free LANGUAGES CXX)

find_package(Kokkos CONFIG REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import numpy"
  RESULT_VARIABLE RFF_PYTHON_NUMPY_RESULT
  OUTPUT_QUIET
  ERROR_QUIET
)
if(NOT RFF_PYTHON_NUMPY_RESULT EQUAL 0)
  message(FATAL_ERROR
    "The selected Python interpreter (${Python3_EXECUTABLE}) must be able to import NumPy. "
    "Configure with -DPython3_EXECUTABLE=/path/to/python3.")
endif()

add_executable(rff-transfer src/main.cpp)
target_compile_features(rff-transfer PRIVATE cxx_std_17)
target_link_libraries(rff-transfer PRIVATE Kokkos::kokkos Python3::Python)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(rff-transfer PRIVATE -Wall -Wextra -Wpedantic)
endif()
