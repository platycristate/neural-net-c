#pragma once
#include <iostream>
#include <vector>
#include <tuple>
#include <Modules.h>
#include <Matrix.h>

struct Network {
    std::vector<LinearLayer> layers;
    std::vector<std::vector<int>> dims;
    explicit Network(std::vector<std::vector<int>> &dims_) {
        dims = dims_;
        for (int i=0; i < dims.size(); i++) {
            LinearLayer l(dims[i][0], dims[i][1]);
            layers.push_back(l);
        }
    }
    void print_structure() {
        for (int i=0; i < dims.size(); i++)
            std::cout << "Layer " << i << ": " <<
            dims[i][0] << ", " << dims[i][1] << std::endl;
    }
    Matrix forward(Matrix &input) {
        Matrix x = layers[0].forward(input);
        for (int i=1; i < layers.size(); i++) {
            x = ReLU::forward(x);
            x = layers[i].forward(x);
        }
        return x;
    }
    void backward(Matrix &grad_output) {
        for (int l=dims.size()-1; l >= 1; l--) {
            layers[l].backward(grad_output);
            grad_output = layers[l].grad_input;
            grad_output = ReLU::backward(grad_output, layers[l].input_vec);
        }
        layers[0].backward(grad_output);
        grad_output = layers[0].grad_input;
    }
};