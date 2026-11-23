cmake_minimum_required(VERSION 3.21)
project(radio_free_free LANGUAGES CXX)

find_package(Kokkos CONFIG REQUIRED)

add_executable(rff-transfer src/main.cpp)
target_compile_features(rff-transfer PRIVATE cxx_std_17)
target_link_libraries(rff-transfer PRIVATE Kokkos::kokkos)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(rff-transfer PRIVATE -Wall -Wextra -Wpedantic)
endif()
