#pragma once
#include <iostream>
#include <tuple>
#include <vector>
#include <random>
#include <cassert>
#include <cmath>
#include <Matrix_c.h>

std::random_device rd{};
std::mt19937 generator{rd()};
std::normal_distribution<> distribution{0, 1.};

double max(matrix const &input) {
    double max_el = input.get(0, 0);
    for (int row=0; row < input.n_rows; row++) {
        if (input.get(row, 0) > max_el)
            max_el = input.get(row,0);
    }
    return max_el;
}
struct SoftMax {
    static matrix forward(const matrix &input) {
        matrix output(input.n_rows, input.n_cols);
        double value;
        double denominator, x;
        double max_el = max(input);

        for (int row=0; row < input.n_rows; row++) {
            for (int col=0; col < input.n_cols; col++) {
                x = input.get(row,col) - max_el;
                value = std::exp(x);
                *output.get_ptr(row,col) = value;
                denominator += value;
            }
        }
        for (int row=0; row < input.n_rows; row++) {
            for (int col=0; col < input.n_cols; col++) {
                *output.get_ptr(row, col) /= denominator;
            }
        }
        return output;
    }

    static matrix backward(const matrix &grad_output, const matrix &output) {
        assert(output.n_rows == grad_output.n_cols && output.n_cols == 1 && grad_output.n_rows == 1);
        int n_elements = grad_output.n_cols;
        matrix D(grad_output.n_cols, grad_output.n_cols);
        for (int row=0; row < n_elements; row++) {
            for (int col=0; col < n_elements; col++) {
                if (row == col) {
                    *D.get_ptr(row ,col) =
                            output.get(col,0) * (1 - output.get(col,0));
                } else {
                    *D.get_ptr(row, col) = -output.get(row,0) * output.get(col, 0);
                }
            }
        }
        matrix grad_output_input = grad_output * D;
        std::cout << "DESTRUCTOR" << std::endl;
        free(D.data);
        return grad_output_input;

    }
};

struct Sigmoid {
    static matrix forward(const matrix &input) {
        matrix output(input.n_rows, input.n_cols);
        double value;
        for (int row=0; row < input.n_rows; row++) {
            for (int col=0; col < input.n_cols; col++) {
                value = 1 / (1 + std::exp(-input.get(row, col)));
                *output.get_ptr(row, col) = value;
            }
        }
        return output;
    }

    static matrix backward(const matrix &grad_output, const matrix &output) {
        assert(output.n_rows == grad_output.n_cols && output.n_cols == 1 && grad_output.n_rows == 1);
        matrix grad_output_input(grad_output.n_rows,
                                 grad_output.n_cols);
        double value;
        matrix output_t = output.transpose();
        for (int row = 0; row < grad_output.n_rows; row++) {
            for (int col = 0; col < grad_output.n_cols; col++) {
                value = grad_output.get(row, col) * output_t.get(row, col) * (1 - output_t.get(row, col));
                *grad_output_input.get_ptr(row, col) = value;
            }
        }
        return grad_output_input;
    }
};
struct ReLU {
    static matrix forward(const matrix &input) {
        matrix output(input.n_rows, input.n_cols);
        double value;
        for (int i=0; i < input.n_rows; i++) {
            for (int j=0; j < input.n_cols; j++) {
                value = input.get(i, j);
                if (value > 0)
                    *output.get_ptr(i, j) = value;
            }
        }
        return output;
    }

    static matrix backward(matrix const &grad_output, matrix const &output) {
        assert(output.n_rows == grad_output.n_cols && output.n_cols == 1 && grad_output.n_rows == 1);
        matrix grad_output_input(grad_output.n_rows,
                                 grad_output.n_cols);
        for (int row=0; row < grad_output.n_rows; row++) {
            for (int col=0; col < grad_output.n_cols; col++) {
                if (output.get(col, row) > 0.0) {
                    *grad_output_input.get_ptr(row, col) = 1.0 * grad_output.get(row, col);
                } else {
                    *grad_output_input.get_ptr(row, col) = 0.0;
                }
            }
        }
        return grad_output_input;
    }
};

struct LinearLayer {
    unsigned int n_neurons;
    unsigned int n_links;
    matrix weight{1, 1 };
    matrix bias{1, 1 };
    matrix input_vec{1, 1 };
    matrix grad_weight{1, 1 };
    matrix grad_bias{1, 1 };
    matrix grad_input{1, 1};

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
    matrix forward(matrix &input) {
        input_vec = input;
        assert(input.n_rows == weight.n_cols);
        assert(input.n_cols == 1);
        matrix res1 = weight * input;
        matrix res2 = res1 + bias;
        std::cout << "DESTRUCTOR" << std::endl;
        free(res1.data);
        return res2;
    }
    void backward(matrix &grad_output) {
        assert(grad_output.n_rows == 1 && grad_output.n_cols == weight.n_rows);
        grad_input = grad_output * weight;
        double grad_w_row_col;
        for (int row=0; row < weight.n_rows; row++) {
            *grad_bias.get_ptr(row, 0) = grad_output.get(0, row);
            for (int col=0; col < weight.n_cols; col++) {
                grad_w_row_col = input_vec.get(col, 0) * grad_output.get(0, row);
                *grad_weight.get_ptr(row, col) = grad_w_row_col;
            }
        }
    }
    void normal_initialization() const {
        generator.seed(259);
        for (int i=0; i < n_neurons; i++) {
            *bias.get_ptr(i,0) = distribution(generator);
            for (int j=0; j < n_links; j++) {
                *weight.get_ptr(i,j) = distribution(generator);
            }
        }
    }

};