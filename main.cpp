#include <vector>
#include <algorithm>
#include <tuple>
#include <Matrix.h>
#include <Network.h>
#include <Loss_functions.h>
#include <Optimizer.h>
#include <Load_data.h>

int training_step(Network &net,
                     Optimizer &opt,
                     Matrix &input,
                     Matrix &target){

    // Forward part
    Matrix pred = net.forward(input);
    Matrix loss = CrossEntropyLoss::forward(pred, target);

    // Backward part
    Matrix grad_output = CrossEntropyLoss::backward(pred, target, loss);
    net.backward(grad_output);

    // Gradient descent
    opt.gradient_step();
    int predicted_label = extract_label(pred);
    return predicted_label;
}

int main() {

    std::random_device rd2;
    std::mt19937 g(rd2());
    g.seed(2895);
    tuple<std::vector<std::vector<double>>, std::vector<unsigned int>> training_data;
    training_data = load_data();
    std::vector<std::vector<double>> inputs = std::get<0>(training_data);
    std::vector<unsigned int> labels = std::get<1>(training_data);

    uniform_int_distribution<int> uni(0, labels.size()-1);

    int n_epochs = 40;
    // Initialisation
    std::vector<std::vector<int>> dims = {{32,  784},
                                          {32, 32},
                                          {10,  32}};
    Network net(dims);
    Optimizer opt(net, 1e-4);
    Matrix input{1, 1};
    Matrix target{1, 1};
    double num_of_examples = labels.size();
    int target_label, index;

    for (int epoch=0; epoch < n_epochs; epoch++) {
        double correctly_classified = 0;
        for (int step=1; step < num_of_examples; step++) {
            index = uni(g);
            target_label = labels[index];
            input = prepare_input(inputs[index]);
            target = one_hot(target_label);
            int pred_label = training_step(net, opt, input, target);
            if (pred_label == target_label)
                correctly_classified += 1;
        }
        std::cout << "Epoch " << epoch << ": ";
        std::cout << correctly_classified / num_of_examples << std::endl;

    }
    return 0;
}
