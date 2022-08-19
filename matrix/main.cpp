#include <iostream>
#include "Matrix_c.h"


int main() {
    double data1[3] = {-1, 1, 3};
    Matrix mat1(data1, 3, 1);
    mat1.print();
    std::cout << '\n';
    Matrix mat2 = mat1.transpose();
    mat2.print();
    return 0;
}