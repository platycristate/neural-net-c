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
    Matrix operator * (Matrix const &mat) {
        assert(n_cols == mat.n_rows);
        double * out = (double *) malloc(sizeof(double) * n_rows * mat.n_cols);
        for (int row=0; row < n_rows; row++) {
            for (int col=0; col < mat.n_cols; col++) {
                double value = 0;
                for (int i=0; i < n_cols; i++) {
                     value += *(data + row*n_cols + i) * *(mat.data + i*mat.n_cols + col);
                }
                *(out + row*mat.n_cols + col) = value;
            }
        }
        Matrix out_mat(out, n_rows, mat.n_cols);
        return out_mat;
    }
    Matrix scalarMul(double const f) const {
        double * out = (double *) malloc(sizeof(double) * n_rows * n_cols);
        for (int row=0; row < n_rows; row++) {
            for (int col=0; col < n_cols; col++)
                *(out + row*n_cols + col) = f * *(data + row*n_cols + col);
        }
        Matrix out_mat(out, n_rows, n_cols);
        return out_mat;
    }
    Matrix operator ^ (Matrix const &mat) {
        assert(n_rows == mat.n_rows && n_cols == mat.n_cols);
        double * out = (double *) malloc(sizeof(double) * n_rows * n_cols);
        for (int row=0; row < n_rows; row++) {
            for (int col=0; col < n_cols; col++)
                *(out + row*n_cols + col) = *(data + row*n_cols + col) * *(mat.data + row*n_cols + col);
        }
        Matrix out_mat(out, n_rows, n_cols);
        return out_mat;
    }
    Matrix operator + (Matrix const &mat) {
        assert(mat.n_cols == n_cols && mat.n_rows == n_rows);
        double * out = (double *) malloc(sizeof(double) * n_rows * n_cols);
        for (int row=0; row < n_rows; row++) {
            for (int col=0; col < n_cols; col++)
                *(out + row*n_cols + col) =
                *(data + row*n_cols + col) + *(mat.data + row*n_cols + col);
        }
        static Matrix out_mat(out, n_rows, n_cols);
        return out_mat;
    }
    Matrix transpose() {
        double * out = (double *) malloc(sizeof(double) * n_rows * n_cols);
        for (int row=0; row < n_rows; row++) {
            for (int col=0; col < n_cols; col++) {
                *(out + row*n_rows + col) = *(data + col*n_cols + row);
                *(out + col*n_rows + row) = *(data + row*n_cols + col);
            }
        }
        Matrix out_mat(out, n_cols, n_rows);
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
