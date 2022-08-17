#pragma once
#include <iostream>
#include <vector>
#include <Matrix.h>
#include <Network.h>

struct Optimizer {
    double learning_rate;
    std::vector<LinearLayer *> layer_ptrs;

    explicit Optimizer( Network &net, double lr) {
        learning_rate = lr;
        LinearLayer * layer_ptr;
        unsigned int n_layers = net.dims.size();
        std::cout << n_layers << std::endl;
        for (int i=0; i < n_layers; i++) {
            layer_ptr = &(net.layers[i]);
            std::cout << "layer " << i << ": "<< layer_ptr << std::endl;
            layer_ptrs.push_back(layer_ptr);
        }
    }
    void gradient_step() {
        Matrix change{1, 1};
        for (auto & layer_ptr : layer_ptrs) {
            change = layer_ptr->grad_weight.scalarMul(learning_rate);
            layer_ptr->weight = layer_ptr->weight - change;
            change = layer_ptr->grad_bias.scalarMul(learning_rate);
            layer_ptr->bias = layer_ptr->bias - change;
        }
    }


};