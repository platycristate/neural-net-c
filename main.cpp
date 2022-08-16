#include <Matrix.h>
#include <Modules.h>
#include <vector>
#include <tuple>
#include <Network.h>

int main() {
    std::vector<std::vector<double>> input_data = {{4}, {-2}};
    Matrix input(input_data);
    std::vector<std::vector<int>> dims = {{8, 2}, {1, 8}};
    Network net(dims);
    Matrix res = net.forward(input);
    Matrix grad_output(1, 1, 1);
    net.backward(grad_output);
    res.printArray();
    net.layers[0].grad_weight.printArray();
    return 0;
}
