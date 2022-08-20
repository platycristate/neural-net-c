#include <algorithm>
#include <tuple>
#include <Matrix_c.h>
#include <Network.h>
#include <Loss_functions.h>
#include <Optimizer.h>
#include <Load_data.h>

int training_step(Network &net,
                     Optimizer &opt,
                     matrix &input,
                     matrix &target){

    // Forward part
    matrix pred = net.forward(input);
    matrix loss = CrossEntropyLoss::forward(pred, target);

    // Backward part
    matrix grad_output = CrossEntropyLoss::backward(pred, target, loss);
    net.backward(grad_output);

    // Gradient descent
    opt.gradient_step();
    int predicted_label = extract_label(pred);

    free(pred.data);
    free(grad_output.data);
    free(loss.data);
    return predicted_label;
}

int main() {

    std::random_device rd2;
    std::mt19937 g(rd2());
    g.seed(2895);

    double num_of_examples = 1000;
    std::tuple<matrix *, matrix *, unsigned int *> training_data = load_data((unsigned int)num_of_examples);
    matrix * inputs = std::get<0>(training_data);
    matrix * targets = std::get<1>(training_data);
    unsigned int * labels = std::get<2>(training_data);

    // Initialisation
    std::vector<std::vector<int>> dims = {{128,  784},
                                          {10,  128}};
    Network net(dims);
    Optimizer opt(net, 1e-4);
    int target_label, index;
    int n_epochs = 20;
    std::uniform_int_distribution<int> uni(0, (int)num_of_examples-1);

    for (int epoch=0; epoch < n_epochs; epoch++) {
        double correctly_classified = 0;
        for (int step=1; step < num_of_examples; step++) {
            index = uni(g);
            target_label = (int)*(labels + index);
            matrix input =  *(inputs + index);
            matrix target = *(targets + index);
            int pred_label = training_step(net, opt, input, target);
            if (pred_label == target_label)
                correctly_classified += 1;
        }
        std::cout << "Epoch " << epoch << ": ";
        std::cout << correctly_classified / num_of_examples << std::endl;
    }
    return 0;
}
