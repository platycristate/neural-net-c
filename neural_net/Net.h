#pragma once
#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <Matrix.h>

std::random_device rd{};
std::mt19937 generator{rd()};
std::normal_distribution<> distribution{0,1};

double randDouble() {
    double number = distribution(generator);
    return number;
}
Matrix ReLU(const Matrix &m) {
    Matrix res(m.n_rows, m.n_cols);
    double value;
    for (int i=0; i < m.n_rows; i++) {
        for (int j=0; j < m.n_cols; j++) {
            value = m.data[i][j];
            if (value > 0)
                res.data[i][j] = value;
        }
    }
    return res;
}

Matrix ReLU_backward(const Matrix &output) {
    Matrix grad(output.n_rows, output.n_cols);
    for (int i = 0; i < output.n_rows; i++) {
        for (int j = 0; j < output.n_cols; j++) {
            if (output.data[i][j] > 0)
                grad.data[i][j] = 1.0;
        }
    }
    return grad;
}


struct LinearLayer {
    int n_neurons;
    int n_links;
    Matrix weight{1, 1};
    Matrix grad{1, 1};

    LinearLayer(int n_neurons_, int n_links_) {
        n_neurons = n_neurons_;
        n_links = n_links_;
        weight.resize(n_neurons, n_links);
        grad.resize(n_neurons, n_links);
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
    Matrix forward(Matrix &input) {
        assert(input.n_rows == weight.n_cols);
        assert(input.n_cols == 1);
        Matrix res = weight * input;
        return res;
    }
    Matrix backward_wrt_weights(Matrix const &inputs, Matrix const &outputs) {
        Matrix transposed_inputs = inputs.transpose();
        Matrix grad_weights(weight.n_rows, weight.n_cols);
        for (int row=0; row < weight.n_rows; row++) {
            for (int col=0; col < weight.n_cols; col++)
                grad_weights.data[row][col] = transposed_inputs.data[0][col];
        }
        return grad_weights;
    }
    Matrix backward_wrt_inputs() {
        return weight;
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