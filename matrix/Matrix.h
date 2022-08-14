#pragma once
#include <iostream>
#include <vector>
#include <cstring>
#include <cassert>
#include <vector>
using namespace std;

struct Matrix {
    vector<vector<float>> data;
    int n_rows;
    int n_cols;

    Matrix( vector<vector<float>> data_) {
        data = data_;
        n_rows = data.size();
        n_cols = data[0].size();
    }
    Matrix( int n_rows_, int n_cols_) {
        vector<vector<float>> data_(
                n_rows_,
                vector<float> (n_cols_, 0));
        data = data_;
        assert(n_rows_ == data.size() && n_cols_ == data[0].size());
        n_rows = data.size();
        n_cols = data[0].size();
    }

    void printArray() const {
        for (int i=0; i < n_rows; i++) {
            for (int j=0; j < n_cols; j++)
                cout << data[i][j] << " ";
            cout << "\n";
        }
    }

    Matrix add(Matrix &m) const {
        assert(n_rows == m.n_rows && n_cols == m.n_cols);
        Matrix res(n_rows, n_cols);
        for (int i=0; i < n_rows; i++){
            for (int j=0; j < n_cols; j++)
                res.data[i][j] = data[i][j] + m.data[i][j];
        }
        return res;
    }

    Matrix mul(Matrix const &m) const {
        assert(n_cols == m.n_rows);
        Matrix res(n_rows, m.n_cols);
        for (int row=0; row < n_rows; row++){
            for (int col=0; col < m.n_cols; col++){
                float val = 0;
                for (int i=0; i < n_cols; i++)
                    val += data[row][i] * m.data[i][col];
                res.data[row][col] = val;
            }
        }
        return res;
    }
    Matrix operator + (Matrix &m) {
        Matrix res = this->add(m);
        return res;
    }

    Matrix scalarMul(float const &f){
        Matrix res(this->n_rows, this->n_cols);
        for (int i=0; i<n_rows; i++){
            for (int j=0; j<n_cols; j++)
                res.data[i][j] = f * data[i][j];
        }
        return res;
    }
    Matrix operator * (Matrix const &m) {
        Matrix res = this->mul(m);
        return res;
    }
    Matrix transpose() const {
        Matrix res(n_cols, n_rows);
        for (int i=0; i < n_rows; i++){
            for (int j=0; j < n_cols; j++)
                res.data[j][i] = data[i][j];
        }
        return res;
    }
};
