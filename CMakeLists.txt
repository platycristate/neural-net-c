cmake_minimum_required(VERSION 3.23)
project(NeuralNet)

set(CMAKE_CXX_STANDARD 14)

include_directories(matrix neural_net)

add_executable(${PROJECT_NAME}
        main.cpp matrix
        neural_net neural_net/Network.h
        neural_net/Optimizer.h neural_net/Load_data.h)
