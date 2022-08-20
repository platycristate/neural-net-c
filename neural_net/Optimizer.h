#pragma once
#include <iostream>
#include <vector>
#include <Matrix_c.h>
#include <Network.h>

struct Optimizer {
    double learning_rate;
    int n_ptrs;
    LinearLayer ** layer_ptrs;

    explicit Optimizer( Network &net, double lr) {
        learning_rate = lr;
        LinearLayer * layer_ptr;
        n_ptrs = net.dims.size();
        layer_ptrs = (LinearLayer **) malloc(sizeof(LinearLayer*) * n_ptrs);
        for (int i=0; i < n_ptrs; i++) {
            layer_ptr = &(net.layers[i]);
            *(layer_ptrs + i) = layer_ptr;
        }
    }
    void gradient_step() {
        matrix change{1, 1};
        LinearLayer * layer_ptr;
        for (int i=0; i < n_ptrs; i++) {
            layer_ptr = layer_ptrs[i];
            change = layer_ptr->grad_weight.scalarMul(learning_rate);
            layer_ptr->weight = layer_ptr->weight - change;
            change = layer_ptr->grad_bias.scalarMul(learning_rate);
            layer_ptr->bias = layer_ptr->bias - change;
        }
    }

};