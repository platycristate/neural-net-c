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

    std::cout << "__------------------------------------------------_________________\n" ;
    // Backward part
    matrix grad_output = CrossEntropyLoss::backward(pred, target, loss);
    std::cout << "#####################################################################################\n";
    net.backward(grad_output);
    std::cout << "-------------------------------------------------------------------------\n";
    // Gradient descent
    opt.gradient_step();
    int predicted_label = extract_label(pred);

    ////std::cout \<< "DESTRUCTOR" << std::endl;
//    free(pred.data);
    //std::cout \<< "DESTRUCTOR" << std::endl;
//    free(grad_output.data);
    ////std::cout \<< "DESTRUCTOR" << std::endl;
//    free(loss.data);
    return predicted_label;
}

int main() {

    std::random_device rd2;
    std::mt19937 g(rd2());
    g.seed(2895);

    std::tuple<matrix*, matrix*, unsigned int*> training_data = load_data();
    matrix * inputs = std::get<0>(training_data);
    matrix * targets = std::get<1>(training_data);
    unsigned int * labels = std::get<2>(training_data);
    std::uniform_int_distribution<int> uni(0, 990);

    // Initialisation
    std::vector<std::vector<int>> dims = {{32,  784},
                                          {10,  32}};
    Network net(dims);
    Optimizer opt(net, 1e-4);
    double num_of_examples = 1000;
    int target_label, index;
    int n_epochs = 6;

    std::cout << "--------------------------------------------------" << std::endl;
    for (int epoch=0; epoch < n_epochs; epoch++) {
        double correctly_classified = 0;
        for (int step=1; step < num_of_examples; step++) {
            index = uni(g);
            target_label = *(labels + index);
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
