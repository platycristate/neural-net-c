#include <vector>
#include <tuple>
#include <Matrix.h>
#include <Network.h>
#include <Loss_functions.h>
#include <Optimizer.h>
#include <Load_data.h>

Matrix training_step(Network &net,
                     Optimizer &opt,
                     Matrix &input,
                     Matrix &target){

    // Forward part
    Matrix pred = net.forward(input);
    Matrix prob = Sigmoid::forward(pred);
    Matrix loss = CrossEntropyLoss::forward(prob, target);

    // Backward part
    Matrix grad_output = CrossEntropyLoss::backward(prob, target, loss);
    grad_output = Sigmoid::backward(grad_output, prob);
    net.backward(grad_output);

    // Gradient descent
    opt.gradient_step();

    return loss;
}

int main() {
    tuple<std::vector<std::vector<double>>, std::vector<unsigned int>> training_data;
    training_data = load_data();
    std::vector<std::vector<double>> inputs = std::get<0>(training_data);
    std::vector<unsigned int> labels = std::get<1>(training_data);
    int n_epochs = 40;
    // Initialisation
    std::vector<std::vector<int>> dims = {{32,  785},
                                          {32,  32},
                                          {32,  32},
                                          {10, 32}};
    Network net(dims);
    Optimizer opt(net, 5e-5);
    Matrix input{1, 1};
    Matrix target{1, 1};
    Matrix loss{1, 1, 0};
    for (int epoch=0; epoch < n_epochs; epoch++) {
        for (int step=1; step < inputs.size(); step++) {
            input = prepare_input(inputs[step]);
            target = one_hot(labels[step]);
            loss = training_step(net, opt, input, target);
        }
        loss.printArray();
    }
    return 0;
}
