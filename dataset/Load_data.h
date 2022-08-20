#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <Matrix_c.h>

matrix one_hot(unsigned int label) {
    unsigned int n_classes = 10;
    matrix output{n_classes, 1};
    *output.get_ptr(label,0) = 1;
    return output;
}

std::tuple<matrix *, matrix *, unsigned int *> load_data() {
    std::fstream file;
    file.open("../data/mnist_train.csv");
    std::string line;
    double data[1000][784];
    unsigned int labels[1000];
    std::getline(file, line);
    double num;
    unsigned int label;
    matrix * inputs = (matrix *) calloc( 1000, sizeof(matrix));
    matrix * targets = (matrix *) calloc(1000, sizeof(matrix));
    for (int i=0; i < 1000; i++) {
        std::getline(file, line);
        std::stringstream str_stream(line);
        str_stream >> label;
        labels[i] = label;
        int j=0;
        while (str_stream.good()){
            str_stream >> num;
            data[i][j] = num / 255;
            j++;
        }
        matrix x = matrix(data[i], 784, 1);
        *(inputs + i) = x;
        *(targets + i) = one_hot(label);
    }
    file.close();
    return {inputs, targets, labels};
}

int extract_label(matrix const &prob) {
    int label=0;
    for (int i=1; i < prob.n_rows; i++) {
        if (prob.get(i, 0) > prob.get(label, 0))
            label = i;
    }
    return label;
}

