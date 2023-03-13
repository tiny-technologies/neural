#include "lib.c"

// SUBCOMMANDS

int train()
{
    Dataset dataset = load_mnist_dataset("mnist/train-labels-idx1-ubyte", "mnist/train-images-idx3-ubyte");
    printf("loaded dataset with %d images\n", dataset.size);

    int dims[] = {dataset.rows * dataset.cols, 128, 64, 10};
    Network network = network_create(4, dims);
    for (int i = 0; i < 20; i++)
    {
        printf("Epoch: %d\n", i);
        epoch(network, dataset, 100);
    }

    int predicted_correctly = 0;
    for (int i = 0; i < dataset.size; i++)
    {
        forward(network, dataset.images[i].data);
        predicted_correctly += arg_max(network.neurons[network.ndim - 1]) == arg_max(dataset.images[i].label);
    }
    printf("predicted: %d, accurarcy: %f\n", predicted_correctly, ((double)predicted_correctly) / dataset.size);

    network_destroy(network);

    return 0;
}

int run()
{
    printf("not implemented yet\n");
    return 0;
}

// USAGE

void print_usage_main()
{
    printf("Usage:\n\n");
    printf("    %sneural <command> [<args>]%s\n\n", BOLD, RESET);
    printf("These are common commands:\n\n");
    printf("    %srun%s    Run inference using a trained network\n", BOLD, RESET);
    printf("    %strain%s  Train a new network and store it to disk\n", BOLD, RESET);
    printf("\n");
}

// MAIN

int main(int argc, char *argv[])
{
    srand(0);

    if (argc == 1 || strcmp(argv[1], "--help") == 0)
    {
        print_usage_main();
        return argc == 1;
    }

    if (strcmp(argv[1], "run") == 0)
    {
        return run();
    }
    else if (strcmp(argv[1], "train") == 0)
    {
        return train();
    }
    else
    {
        printf("neural: '%s' is not a neural command. See 'neural --help'.\n", argv[1]);
        return 1;
    }

    return 0;
}
