#include <iostream>
#include "Matrix_c.h"


int main() {
    double data[6] = {1.0, 2.0, 3, 4, 5, 6};
    double data2[6] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    unsigned int n_rows = 2;
    unsigned int n_cols = 3;
    Matrix mat1(data, n_rows, n_cols);
    Matrix mat2(data2, n_rows, n_cols);
    Matrix mat3 = mat1 + mat2;
    mat3.print();
    return 0;
}