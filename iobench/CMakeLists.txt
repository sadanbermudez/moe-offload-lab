cmake_minimum_required(VERSION 3.20)
project(iobench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(iobench iobench.cpp)
target_compile_options(iobench PRIVATE -Wall -Wextra)
find_package(Threads REQUIRED)
target_link_libraries(iobench PRIVATE Threads::Threads)
