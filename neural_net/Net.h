#pragma once
#include <iostream>
#include <vector>
#include <Matrix.h>

struct Net{
    int rows = 2;
    int cols = 2;
    std::vector<Matrix> layers;
    Net(int n_rows_, int n_cols_){
        Matrix m(n_rows_, n_cols_);
        m.printArray();
        rows = n_rows_;
        cols = n_cols_;
    }
};