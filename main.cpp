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
    Matrix x1_act = ReLU::forward(x1);
    Matrix x2 = layer2.forward(x1);

    // Initialize
    Matrix grad_output(1, 1, 1);
    std::tuple<Matrix, Matrix> grads = layer2.backward(grad_output);
    Matrix grad_output2 = std::get<0>(grads);

    std::tuple<Matrix> grads2 = ReLU::backward(grad_output2, x1_act);
    Matrix grad_output3 = std::get<0>(grads2);

    std::tuple<Matrix, Matrix> grads3 = layer1.backward(grad_output3);
    Matrix grad_output4 = std::get<0>(grads3);
    Matrix grad_weights1 = std::get<1>(grads3);
    Matrix grad_weights2 = std::get<1>(grads);

    std::cout << "grad_weights1:\n";
    grad_weights1.shape();
    grad_weights1.printArray();

    return 0;
}
