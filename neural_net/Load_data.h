#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

std::tuple<std::vector<std::vector<double>>, std::vector<unsigned int>> load_data() {
    fstream file;
    file.open("../data/mnist_train.csv");
    std::string line;
    std::vector<std::vector<double>> data;
    std::vector<unsigned int> labels;
    std::getline(file, line);
    double num;
    unsigned int label;

    for (int i=0; i < 5000; i++){
        std::getline(file, line);
        stringstream str_stream(line);
        str_stream >> label;
        std::vector<double> data_example;
        labels.push_back(label);
        while (str_stream.good()){
            str_stream >> num;
            num /= 255;
            data_example.push_back(num);
        }
        data.push_back(data_example);
    }
    file.close();
    return {data, labels};
}

Matrix prepare_input(std::vector<double> vector) {
    Matrix output{static_cast<unsigned int>(vector.size()), 1};
    for (int row=0; row < vector.size(); row++)
        output.data[row][0] = vector[row];
    return output;
}

Matrix one_hot(unsigned int label) {
    unsigned int n_classes = 10;
    Matrix output{n_classes, 1};
    output.data[label][0] = 1;
    return output;
}