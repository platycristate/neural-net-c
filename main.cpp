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
    std::cout << "grad_output2:\n";
    grad_output2.shape();
    grad_output2.printArray();

    std::tuple<Matrix> grads2 = ReLU::backward(grad_output2, x2);
    Matrix grad_output3 = std::get<0>(grads2);
    std::cout << "grad_output3:\n";
    grad_output3.shape();
    grad_output3.printArray();
    return 0;
}
