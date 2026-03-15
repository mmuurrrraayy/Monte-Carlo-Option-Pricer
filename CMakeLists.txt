cmake_minimum_required(VERSION 3.10)
project(Monte-Carlo-Option-Pricer)

set(CMAKE_CXX_STANDARD 17)

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(Monte-Carlo-Option-Pricer ${SOURCES})
target_include_directories(Monte-Carlo-Option-Pricer PRIVATE src)