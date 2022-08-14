cmake_minimum_required(VERSION 3.23)
project(NeuralNet)

set(CMAKE_CXX_STANDARD 14)
include_directories(matrix)
add_executable(NeuralNet main.cpp matrix/Matrix.h matrix/Test.h)
