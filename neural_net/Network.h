#pragma once
#include <iostream>
#include <vector>
#include <tuple>
#include <Modules.h>
#include <Matrix_c.h>

struct Network {
    std::vector<std::vector<int>> dims;
    LinearLayer * layers;
    explicit Network(std::vector<std::vector<int>> &dims_) {
        dims = dims_;
        layers = (LinearLayer *) malloc(sizeof(LinearLayer) * dims.size());
        for (int i=0; i < dims.size(); i++) {
            LinearLayer l(dims[i][0], dims[i][1]);
            *(layers + i) = l;
        }
    }
    void print_structure() {
        for (int i=0; i < dims.size(); i++)
            std::cout << "Layer " << i << ": " <<
            dims[i][0] << ", " << dims[i][1] << std::endl;
    }
    matrix forward(matrix &input) const {
        matrix x = layers[0].forward(input);
        for (int i=1; i < dims.size(); i++) {
            x = ReLU::forward(x);
            x = layers[i].forward(x);
        }
        return x;
    }
    void backward(matrix &grad_output) {
        for (unsigned int l=dims.size()-1; l >= 1; l--) {
            layers[l].backward(grad_output);
            grad_output = ReLU::backward(layers[l].grad_input, layers[l].input_vec);
            std::cout << "DESTRUCTOR" << std::endl;
            free(layers[l].grad_input.data);
            std::cout << "DESTRUCTOR" << std::endl;
            free(layers[l].input_vec.data);
        }
        layers[0].backward(grad_output);
        std::cout << "DESTRUCTOR" << std::endl;
        free(grad_output.data);
        grad_output = layers[0].grad_input;
    }
};