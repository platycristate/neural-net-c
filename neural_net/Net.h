#pragma once
#include <iostream>
#include <vector>
#include <random>
#include <Matrix.h>

std::random_device rd{};
std::mt19937 generator{rd()};
std::normal_distribution<> distribution{0,1};

double randDouble() {
    double number = distribution(generator);
    return number;
}
struct LinearLayer {
    int n_neurons;
    int n_links;
    Matrix weight{1, 1};

    LinearLayer(int n_neurons_, int n_links_) {
        n_neurons = n_neurons_;
        n_links = n_links_;
        weight.resize(n_neurons, n_links);
        normal_initialization();
    }
    void normal_initialization() {
        double value;
        for (int i=0; i < n_neurons; i++) {
            for (int j=0; j < n_links; j++) {
                value = randDouble();
                weight.data[i][j] = value;
            }
        }
    }
};

struct Net{
    int rows = 2;
    int cols = 2;
    std::vector<Matrix> layers;
    Net(int n_rows_, int n_cols_){
        Matrix m(n_rows_, n_cols_);
        m.printArray();
        rows = n_rows_;
        cols = n_cols_;
    }
};