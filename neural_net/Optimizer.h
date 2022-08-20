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
        LinearLayer * layer_ptr;
        for (int i=0; i < n_ptrs; i++) {
            layer_ptr = layer_ptrs[i];
            matrix change_weight = layer_ptr->grad_weight.scalarMul(learning_rate);
            matrix new_weight = layer_ptr->weight - change_weight;
            free(layer_ptr->weight.data);
            layer_ptr->weight = new_weight;
            matrix change_bias = layer_ptr->grad_bias.scalarMul(learning_rate);
            matrix new_bias = layer_ptr->bias - change_bias;
            free(layer_ptr->bias.data);
            layer_ptr->bias = new_bias;
            free(change_weight.data);
            free(change_bias.data);
        }
    }

};