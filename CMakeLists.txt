project(dummy NONE)
# comment out due to the incompatibility with Xcode in homebrew
# if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
#     set(CMAKE_CXX_COMPILER g++-13)
# endif()
cmake_minimum_required(VERSION 3.14)
project(benchmark_eigen CXX)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

message(PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
cmake_path(GET PROJECT_SOURCE_DIR PARENT_PATH PARENT_DIR)
message(PARENT_DIRECTORY="${PARENT_DIR}")
cmake_path(GET PARENT_DIR PARENT_PATH ROOT_DIR)
message(ROOT_DIRECTORY="${ROOT_DIR}")
cmake_path(APPEND INCLUDE_DIR ${ROOT_DIR} include)
message(INCLUDE_DIR="${INCLUDE_DIR}")
cmake_path(APPEND SOURCE_DIR ${ROOT_DIR} src)
message(SOURCE_DIR="${SOURCE_DIR}")

add_executable(${PROJECT_NAME} main.cpp)

target_compile_options(${PROJECT_NAME} PRIVATE -O3 -Wall -Wextra -Wno-psabi)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)

include(FetchContent)
find_package(Eigen3)
if(Eigen3_FOUND)
    message(STATUS "Eigen3 exists")
    message(EIGEN3_INCLUDE_DIR="${EIGEN3_INCLUDE_DIR}")
else()
    message(STATUS "Fetching Eigen3")
    FetchContent_Declare(
        Eigen3
        GIT_REPOSITORY  https://gitlab.com/libeigen/eigen.git
        GIT_TAG         master
    )
    FetchContent_MakeAvailable(Eigen3)
endif()
target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen)
