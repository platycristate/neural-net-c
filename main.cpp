#include <Matrix.h>
#include <Net.h>
#include <vector>
#include <tuple>

int main() {
    std::vector<std::vector<double>> data = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    std::vector<std::vector<double>> data2 = {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
    std::vector<std::vector<double>> input_data = {{4}, {-2}};
    Matrix input(input_data);

    LinearLayer layer1(8, 2);
    LinearLayer layer2(1, 8);

    Matrix x1 = layer1.forward(input);
    Matrix x2 = layer2.forward(x1);

    x1.shape();
    x2.shape();
    Matrix grad_output(1, 1, 1);
    grad_output.shape();
    std::tuple<Matrix, Matrix> grads = layer2.backward(grad_output);
    Matrix grad0 = std::get<0>(grads);
    Matrix grad1 = std::get<1>(grads);
    grad0.shape();
    grad1.shape();
    grad0.printArray();
    grad1.printArray();
    return 0;
}
