#pragma once
#include <cassert>
#include <cmath>
//#include <Matrix.h>
#include <Matrix_c.h>
#include <Modules.h>

struct CrossEntropyLoss {
    static matrix forward(matrix &pred, matrix &target ) {
        assert(target.n_rows == pred.n_rows);
        double result = 0;
        double value;
        matrix sm_activated = SoftMax::forward(pred);
        for (int i=0; i < pred.n_rows; i++) {
            value = target.get(i, 0) * log(sm_activated.get(i, 0));
            result -= value;
        }
        return matrix{1, 1, result};
    }
    static matrix backward(
            matrix &pred,
            matrix &target,
            matrix &output) {
        matrix grad_output_input(pred.n_rows, pred.n_cols, 0);
        matrix sm_activated = SoftMax::forward(pred);
        for (int row=0; row < pred.n_rows; row++)
            *grad_output_input.get_ptr(row,0) = -target.get(row, 0) * (1 / sm_activated.get(row,0) + 1e-8);

        matrix sm_grad = SoftMax::backward(
                grad_output_input.transpose(),
                sm_activated).transpose();
        return sm_grad.transpose();
    }
};