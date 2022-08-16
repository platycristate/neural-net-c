#include <Matrix.h>
#include <vector>
#include <Network.h>
#include <Loss_functions.h>

int main() {
    std::vector<std::vector<double>> input_data = {{4}, {-2}};
    std::vector<std::vector<double>> target_data = {{0, 1, 1, 1, 0, 0, 1, 0}};
    std::vector<std::vector<int>> dims = {{8, 2}, {8, 8}};

    Matrix input(input_data);
    Matrix target(target_data);
    target = target.transpose();
    Network net(dims);

    // Forward part
    Matrix pred = net.forward(input);
    pred = ReLU::forward(pred);

    // Backward part
    Matrix loss = CrossEntropyLoss::forward(pred, target);
    Matrix grad_output = CrossEntropyLoss::backward(pred, target, loss);
    grad_output.printArray();
    grad_output.shape();
    grad_output = ReLU::backward(grad_output, pred);
    net.backward(grad_output);
    net.layers[1].weight.printArray();
    net.layers[1].weight.shape();
    return 0;
}
