#pragma once
#include <iostream>
#include <cstring>
#include <cassert>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include "Matrix.h"

bool compareMatrices(const Matrix &a, const Matrix &b) {
    assert(a.n_rows == b.n_rows && a.n_cols == b.n_cols);
    for (int i=0; i < a.n_rows; i++) {
        for (int j=0; j < a.n_cols; j++)
            if (a.data[i][j] != b.data[i][j])
                return false;
    }
    return true;
}

void readData(){
    std::ifstream file;
    file.open("../tests.txt");
    std::string line;
    // Read a file
    if (file.is_open()) {
        while (file.good()){
            std::getline(file, line);
            std::istringstream iss(line);
            std::vector<float> array;
            float element;
            while (iss.good()){
                iss >> element;
                array.push_back(element);
            }
            std::cout << array.size() << std::endl;

        }
    }
    file.close();
}

void Test(){
//    std::cout << "Testing addition" << std::endl;
    std::vector<std::vector<float>> data = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    std::vector<std::vector<float>> data2 = {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
    std::vector<std::vector<float>> data_cor = {{2, 3, 4}, {5, 6, 7}, {8, 9, 10}};
    // Addition Test 1
    Matrix mat1(data), mat2(data2), res_cor(data_cor);
    Matrix res = mat1 + mat2;
    assert(compareMatrices(res, res_cor) == true);

    // Addition Test 2
    Matrix mat3(3, 3);
    res = mat1 + mat3;
    res_cor = mat1;
    assert(compareMatrices(res, res_cor) == true);

    // Addition Test 3
    std::vector<std::vector<float>> data4 = {{1}};
    std::vector<std::vector<float>> data5 = {{3}};
    std::vector<std::vector<float>> data_cor2 = {{4}};
    Matrix mat4(data4), mat5(data5), res_cor2(data_cor2);
    Matrix res2 = mat4 + mat5;
    assert(compareMatrices(res2, res_cor2) == true);
//    std::cout << "Finished addition" << std::endl;

//    std::cout << "Testing transpositions" << std::endl;
    std::vector<std::vector<float>> data6 = {{1}};
    Matrix mat6(data6);
    Matrix mat7 = mat6.transpose();
    assert(compareMatrices(mat7, mat6) == true);

    std::vector<std::vector<float>> data8 = {{1, 2}};
    std::vector<std::vector<float>> data_cor8 = {{1}, {2}};
    Matrix mat8(data8), res_cor8(data_cor8);
    Matrix res8 = mat8.transpose();
    assert(compareMatrices(res8, res_cor8) == true);
//    std::cout << "Finished Testing" << std::endl;

    // Tests from file
    readData();
}