#pragma once
#include <cassert>
#include <cmath>
#include <Matrix.h>

struct CrossEntropyLoss {
    static Matrix forward(Matrix &pred, Matrix &target ) {
        assert(target.n_rows == pred.n_rows);
        double result = 0;
        double value;
        for (int i=0; i < pred.n_rows; i++) {
            value = target.data[i][0] * log(pred.data[i][0]);
            result -= value;
        }
        return Matrix{1, 1, result};
    }
    static Matrix backward(
            Matrix &pred,
            Matrix &target,
            Matrix &output) {
        Matrix grad_output_input(pred.n_rows, pred.n_cols);
        double grad_w;
        for (int row=0; row < pred.n_rows; row++) {
            grad_w = -target.data[row][0] * 1 / (pred.data[row][0] + 1e-8);
            grad_output_input.data[row][0] = grad_w;
        }
        return grad_output_input.transpose();
    }
};