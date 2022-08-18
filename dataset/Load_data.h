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

    for (int i=0; i < 1000; i++){
        std::getline(file, line);
        stringstream str_stream(line);
        str_stream >> label;
        std::vector<double> data_example;
        labels.push_back(label);
        int j=0;
        while (str_stream.good()){
            str_stream >> num;
            num /= 255;
            data_example.push_back(num);
            j++;
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

std::vector<Matrix> preprocess_data(std::vector<std::vector<double>> const &raw_data) {
    std::vector<Matrix> data;
    for (const auto & i : raw_data)
        data.push_back(prepare_input(i));
    return data;
}
Matrix one_hot(unsigned int label) {
    unsigned int n_classes = 10;
    Matrix output{n_classes, 1};
    output.data[label][0] = 1;
    return output;
}

std::vector<Matrix> preprocess_labels(std::vector<unsigned int> const &raw_labels) {
    std::vector<Matrix> targets;
    int label;
    for (int i=0; i < raw_labels.size(); i++) {
        label = raw_labels[i];
        targets.push_back(one_hot(label));
    }
    return targets;
}


int extract_label(Matrix const &prob) {
    int label=0;
    for (int i=1; i < prob.n_rows; i++) {
        if (prob.data[i] > prob.data[label])
            label = i;
    }
    return label;
}


