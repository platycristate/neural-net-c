#pragma once
#include <iostream>
#include <cassert>


struct matrix {
    double * data;
    unsigned int n_rows;
    unsigned int n_cols;

    matrix(double * data_,
           unsigned int n_rows_,
           unsigned int n_cols_) {
        data = data_;
        n_rows = n_rows_;
        n_cols = n_cols_;
    }
    matrix (unsigned int n_rows_, unsigned int n_cols_) {
        n_rows = n_rows_;
        n_cols = n_cols_;
        data = (double*)malloc(sizeof(double) * n_rows * n_cols);
    }
    matrix (unsigned int n_rows_, unsigned int n_cols_,
            double value) {
        n_rows = n_rows_;
        n_cols = n_cols_;
        data = (double*)malloc(sizeof(double) * n_rows * n_cols);
        for (int row=0; row < n_rows; row++) {
            for (int col=0; col < n_cols; col++)
                *(data + row*n_cols + col) = value;
        }
    }
    matrix operator * (matrix const &mat) const {
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
        matrix out_mat(out, n_rows, mat.n_cols);
        return out_mat;
    }
    matrix scalarMul(double const f) const {
        double * out = (double *) malloc(sizeof(double) * n_rows * n_cols);
        for (int row=0; row < n_rows; row++) {
            for (int col=0; col < n_cols; col++)
                *(out + row*n_cols + col) = f * *(data + row*n_cols + col);
        }
        matrix out_mat(out, n_rows, n_cols);
        return out_mat;
    }
    matrix operator ^ (matrix const &mat) const {
        assert(n_rows == mat.n_rows && n_cols == mat.n_cols);
        double * out = (double *) malloc(sizeof(double) * n_rows * n_cols);
        for (int row=0; row < n_rows; row++) {
            for (int col=0; col < n_cols; col++)
                *(out + row*n_cols + col) = *(data + row*n_cols + col) * *(mat.data + row*n_cols + col);
        }
        matrix out_mat(out, n_rows, n_cols);
        return out_mat;
    }
    matrix operator + (matrix const &mat) const {
        assert(mat.n_cols == n_cols && mat.n_rows == n_rows);
        double * out = (double *) malloc(sizeof(double) * n_rows * n_cols);
        for (int row=0; row < n_rows; row++) {
            for (int col=0; col < n_cols; col++)
                *(out + row*n_cols + col) =
                *(data + row*n_cols + col) + *(mat.data + row*n_cols + col);
        }
        static matrix out_mat(out, n_rows, n_cols);
        return out_mat;
    }
    matrix operator - (matrix const &mat) const {
        assert(mat.n_cols == n_cols && mat.n_rows == n_rows);
        matrix out_mat = *(this) + mat.scalarMul(-1);
        return out_mat;
    }
    double get(int const i, int const j) const {
        assert(i < n_rows && j < n_cols);
        return *(data + n_cols*i + j);
    }
    double * get_ptr(int const i, int const j) const {
        assert(i < n_rows && j < n_cols);
        return data + n_cols*i + j;
    }
    matrix transpose() const {
        double * out = (double *) malloc(sizeof(double) * n_rows * n_cols);
        for (int row=0; row < n_rows; row++) {
            for (int col=0; col < n_cols; col++) {
                *(out + row*n_rows + col) = *(data + col*n_cols + row);
                *(out + col*n_rows + row) = *(data + row*n_cols + col);
            }
        }
        matrix out_mat(out, n_cols, n_rows);
        return out_mat;
    }
    void resize(int n_rows_, unsigned int n_cols_) {
        data = (double *) malloc(sizeof(double) * n_rows_ * n_cols_);
        n_rows = n_rows_;
        n_cols = n_cols_;
    }
    void shape() const {
        std::cout << "(" << n_rows << ", " << n_cols << ")\n";
    }
    void print() const {
        for (int row=0; row < n_rows; row++) {
            for (int col=0; col < n_cols; col++)
                std::cout << *(data + row*n_cols + col) << ' ';
            std::cout << '\n';
        }
    }
};
