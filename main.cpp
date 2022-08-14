#include <Test_matrix.h>
#include <Net.h>
#include <vector>

int main() {
    std::vector<std::vector<float>> data = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    std::vector<std::vector<float>> data2 = {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
    Test();
    Net net1(2, 3);
    return 0;
}
