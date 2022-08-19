#pragma once
#include <iostream>
#include <cassert>


struct Matrix {
    double * data;
    unsigned int n_rows;
    unsigned int n_cols;

    Matrix(double * data_,
           unsigned int n_rows_,
           unsigned int n_cols_) {
        data = data_;
        n_rows = n_rows_;
        n_cols = n_cols_;
    }
    Matrix (unsigned int n_rows_, unsigned int n_cols_,
            double value) {
        n_rows = n_rows_;
        n_cols = n_cols_;
        data = (double*)malloc(sizeof(double) * n_rows * n_cols);
        for (int row=0; row < n_rows; row++) {
            for (int col=0; col < n_cols; col++)
                *(data + row*n_cols + col) = value;
        }
    }
    Matrix operator + (Matrix const &mat) {
        assert(mat.n_cols == n_cols && mat.n_rows == n_rows);
        double * out = (double*) malloc(sizeof(double) * n_rows * n_cols);
        for (int row=0; row < n_rows; row++) {
            for (int col=0; col < n_cols; col++)
                *(out + row*n_cols + col) =
                *(data + row*n_cols + col) + *(mat.data + row*n_cols + col);
        }
        static Matrix out_mat(out, n_rows, n_cols);
        return out_mat;
    }

    void print() const {
        for (int row=0; row < n_rows; row++) {
            for (int col=0; col < n_cols; col++)
                std::cout << *(data + row*n_cols + col) << ' ';
            std::cout << '\n';
        }
    }
};
