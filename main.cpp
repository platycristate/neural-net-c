#include <Matrix.h>
#include <Net.h>
#include <vector>

int main() {
    std::vector<std::vector<double>> data = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    std::vector<std::vector<double>> data2 = {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
    std::vector<std::vector<double>> input_data = {{4}, {-2}};
    Matrix input(input_data);
    LinearLayer layer(3, 2);
    Matrix res = layer.forward(input);
    std::cout << '\n';
    return 0;
}
