#pragma once
#include <iostream>
#include <tuple>
#include <vector>
#include <random>
#include <cassert>
#include <cmath>
#include <Matrix.h>

std::random_device rd{};
std::mt19937 generator{rd()};
std::uniform_real_distribution<> distribution{-0.5, 0.5};

struct SoftMax {
    static Matrix forward(const Matrix &input) {
        Matrix output(input.n_rows, input.n_cols);
        double value;
        double denominator;
        for (int row=0; row < input.n_rows; row++) {
            for (int col=0; col < input.n_cols; col++) {
                value = std::exp(input.data[row][col]);
                output.data[row][col] = value;
                denominator += value;
            }
        }
        for (int row=0; row < input.n_rows; row++) {
            for (int col=0; col < input.n_cols; col++) {
                output.data[row][col] /= denominator;
            }
        }
        return output;
    }
};

struct Sigmoid {
    static Matrix forward(const Matrix &input) {
        Matrix output(input.n_rows, input.n_cols);
        double value;
        for (int row=0; row < input.n_rows; row++) {
            for (int col=0; col < input.n_cols; col++) {
                value = 1 / (1 + std::exp(-input.data[row][col]));
                output.data[row][col] = value;
            }
        }
        return output;
    }

    static Matrix backward(const Matrix &grad_output, const Matrix &output) {
        assert(output.n_rows == grad_output.n_cols && output.n_cols == 1 && grad_output.n_rows == 1);
        Matrix grad_output_input(grad_output.n_rows,
                                 grad_output.n_cols);
        double value;
        Matrix output_t = output.transpose();
        for (int row = 0; row < grad_output.n_rows; row++) {
            for (int col = 0; col < grad_output.n_cols; col++) {
                value = grad_output.data[row][col] * output_t.data[row][col] * (1 - output_t.data[row][col]);
                grad_output_input.data[row][col] = value;
            }
        }
        return grad_output_input;
    }

};
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

    static Matrix backward(Matrix const &grad_output, Matrix const &output) {
        assert(output.n_rows == grad_output.n_cols && output.n_cols == 1 && grad_output.n_rows == 1);
        Matrix grad_output_input(grad_output.n_rows,
                                 grad_output.n_cols);
        for (int row=0; row < grad_output.n_rows; row++) {
            for (int col=0; col < grad_output.n_cols; col++) {
                if (output.data[col][row] > 0.0) {
                    grad_output_input.data[row][col] = 1.0 * grad_output.data[row][col];
                } else {
                    grad_output_input.data[row][col] = 0.0;
                }
            }
        }
        return grad_output_input;
    }
};

struct LinearLayer {
    unsigned int n_neurons;
    unsigned int n_links;
    Matrix weight{1, 1};
    Matrix bias{1, 1};
    Matrix input_vec{1, 1};
    Matrix grad_weight{1, 1};
    Matrix grad_bias{1, 1};
    Matrix grad_input{1, 1};

    LinearLayer(unsigned int n_neurons_, unsigned int n_links_) {
        n_neurons = n_neurons_;
        n_links = n_links_;
        weight.resize(n_neurons, n_links);
        input_vec.resize(n_links, 1);
        bias.resize(n_neurons, 1);
        grad_weight.resize(n_neurons, n_links);
        grad_bias.resize(n_neurons, 1);
        grad_input.resize(1, n_links);
        normal_initialization();
    }
    Matrix forward(Matrix &input) {
        input_vec = input;
        assert(input.n_rows == weight.n_cols);
        assert(input.n_cols == 1);
        Matrix res = weight * input;
        res = res + bias;
        return res;
    }
    void backward(Matrix &grad_output) {
        assert(grad_output.n_rows == 1 && grad_output.n_cols == weight.n_rows);
        Matrix grad_output_input = grad_output * weight;
        Matrix grad_output_weight(weight.n_rows, weight.n_cols);
        Matrix grad_output_bias(bias.n_rows, bias.n_cols);
        double grad_w_row_col;
        for (int row=0; row < weight.n_rows; row++) {
            grad_output_bias.data[row][0] = grad_output.data[0][row];
            for (int col=0; col < weight.n_cols; col++) {
                grad_w_row_col = input_vec.data[col][0] * grad_output.data[0][row];
                grad_output_weight.data[row][col] = grad_w_row_col;
            }
        }
        grad_weight = grad_output_weight;
        grad_bias = grad_output_bias;
        grad_input = grad_output_input;
    }
    void normal_initialization() {
        generator.seed(259);
        for (int i=0; i < n_neurons; i++) {
            bias.data[i][0] = distribution(generator);
            for (int j=0; j < n_links; j++) {
                weight.data[i][j] = distribution(generator);
            }
        }
    }

};