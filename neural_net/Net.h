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

struct ReLU {
    static Matrix forward(const Matrix &input) {
        Matrix output(input.n_rows, input.n_cols);
        double value;
        for (int i=0; i < input.n_rows; i++) {
            for (int j=0; j < input.n_cols; j++) {
                value = input.data[i][j];
                if (value > 0)
                    output.data[i][j] = value;
            }
        }
        return output;
    }

    static std::tuple<Matrix> backward(Matrix const &grad_output, Matrix const &output){
        Matrix grad_output_input(grad_output.n_rows,
                                 grad_output.n_cols);
        for (int row=0; row < grad_output.n_rows; row++) {
            for (int col=0; col < grad_output.n_cols; col++) {
                if (output.data[row][col] > 0.0) {
                    grad_output_input.data[row][col] = 1.0;
                } else {
                    grad_output_input.data[row][col] = 0.0;
                }
            }
        }
        return {grad_output_input};
    }
};

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