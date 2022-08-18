#pragma once
#include <cassert>
#include <cmath>
#include <Matrix.h>
#include <Modules.h>

struct CrossEntropyLoss {
    static Matrix forward(Matrix &pred, Matrix &target ) {
        assert(target.n_rows == pred.n_rows);
        double result = 0;
        double value;
        Matrix sm_activated = SoftMax::forward(pred);
        for (int i=0; i < pred.n_rows; i++) {
            value = target.data[i][0] * log(sm_activated.data[i][0]);
            result -= value;
        }
        return Matrix{1, 1, result};
    }
    static Matrix backward(
            Matrix &pred,
            Matrix &target,
            Matrix &output) {
        Matrix grad_output_input(pred.n_rows, pred.n_cols, 0);
        Matrix sm_activated = SoftMax::forward(pred);
        for (int row=0; row < pred.n_rows; row++)
            grad_output_input.data[row][0] = -target.data[row][0] * (1 / sm_activated.data[row][0] + 1e-8);

        Matrix sm_grad = SoftMax::backward(
                grad_output_input.transpose(),
                    sm_activated).transpose();
        return sm_grad.transpose();
    }
};