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
    Matrix prob = Sigmoid::forward(pred);
    Matrix loss = CrossEntropyLoss::forward(prob, target);

    // Backward part
    Matrix grad_output = CrossEntropyLoss::backward(prob, target, loss);
    grad_output = Sigmoid::backward(grad_output, prob);
    net.backward(grad_output);

    net.layers[0].grad_bias.printArray();
    return 0;
}
