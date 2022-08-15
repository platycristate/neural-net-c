#include <Matrix.h>
#include <Net.h>
#include <vector>

int main() {
    std::vector<std::vector<double>> data = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    std::vector<std::vector<double>> data2 = {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
    std::vector<std::vector<double>> input_data = {{4}, {-2}};
    Matrix input(input_data);

    LinearLayer layer1(8, 2);
    LinearLayer layer2(8, 8);
    LinearLayer layer3(8, 8);
    LinearLayer layer4(1, 8);

    Matrix x1 = layer1.forward(input);
    Matrix x2 = ReLU(x1);
    Matrix x3 = layer2.forward(x2);
    Matrix x4 = ReLU(x3);
    Matrix x5 = layer3.forward(x4);
    Matrix x6 = ReLU(x5);
    Matrix x7 = layer4.forward(x6);

    Matrix m = layer2.backward_wrt_weights(x2, x3);
    m.shape();
    m.printArray();

    x7.shape();
    x7.printArray();
    return 0;
}
