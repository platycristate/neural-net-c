#pragma once
#include <iostream>
#include <tuple>
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
    Matrix input_vec{1, 1};
    Matrix grad{1, 1};

    LinearLayer(int n_neurons_, int n_links_) {
        n_neurons = n_neurons_;
        n_links = n_links_;
        weight.resize(n_neurons, n_links);
        input_vec.resize(n_links, 1);
        normal_initialization();
    }
    Matrix forward(Matrix &input) {
        input_vec = input;
        assert(input.n_rows == weight.n_cols);
        assert(input.n_cols == 1);
        Matrix res = weight * input;
        return res;
    }
    std::tuple<Matrix, Matrix> backward(Matrix &grad_output) const {
        assert(grad_output.n_rows == 1 && grad_output.n_cols == weight.n_rows);
        Matrix grad_output_input = grad_output * weight;
        Matrix grad_output_weight(weight.n_rows, weight.n_cols);
        double grad_w_row_col;
        for (int row=0; row < weight.n_rows; row++) {
            for (int col=0; col < weight.n_cols; col++) {
                grad_w_row_col = input_vec.data[col][0] * grad_output.data[0][row];
                grad_output_weight.data[row][col] = grad_w_row_col;
            }
        }
        return {grad_output_input, grad_output_weight};
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