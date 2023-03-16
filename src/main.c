#include "lib.c"

// SUBCOMMANDS

int train(int batch_size, int ndim, int *dims_hidden, int epochs, double learning_rate)
{
    Dataset dataset = load_mnist_dataset("mnist/train-labels-idx1-ubyte", "mnist/train-images-idx3-ubyte");
    printf("loaded dataset with %d images\n", dataset.size);

    int dims[ndim];
    dims[0] = dataset.rows * dataset.cols;
    for (int i = 1; i < ndim - 1; i++)
    {
        dims[i] = dims_hidden[i];
    }
    dims[ndim - 1] = 10;

    printf("initialized network with layers:");
    for (int i = 0; i < ndim; i++)
        printf(" %d", dims[i]);
    printf("\n");

    Network network = network_create(ndim, dims);
    for (int i = 0; i < epochs; i++)
    {
        printf("Epoch: %d\n", i);
        epoch(network, dataset, batch_size);
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

int print_usage_main()
{
    printf("Usage:\n\n");
    printf("    %sneural <command> [<args>]%s\n\n", BOLD, RESET);
    printf("Commands:\n\n");
    printf("    %srun%s    Run inference using a trained network\n\n", BOLD, RESET);
    printf("    %strain%s  Train a new network and store it to disk\n", BOLD, RESET);
    printf("      %s-b, --batch-size <INT>%s      samples per batch (default: 200)\n", BOLD, RESET);
    printf("      %s-d, --dims <INT,INT,..>%s     dimensions of hidden layers (default: 16,16)\n", BOLD, RESET);
    printf("      %s-e, --epochs <INT>%s          number of epochs (default: 10)\n", BOLD, RESET);
    printf("      %s-l, --learning-rate <REAL>%s  step size of parameter update (default: 0.001)\n\n", BOLD, RESET);
    printf("    %shelp%s   Show this message and exit\n", BOLD, RESET);
    printf("\n");

    return 0;
}

// MAIN

int main(int argc, char *argv[])
{
    srand(0);

    if (argc == 1 || strcmp(argv[1], "help") == 0 || strcmp(argv[1], "--help") == 0)
    {
        print_usage_main();
        return argc == 1;
    }

    else if (strcmp(argv[1], "run") == 0)
    {
        return run();
    }

    else if (strcmp(argv[1], "train") == 0)
    {
        // default values
        int batch_size = 200;
        int dims_hidden[8] = {16};
        int ndim = 4;
        int epochs = 10;
        double learning_rate = 0.001;

        // Parse optional flags
        char c;
        for (int i = 2; i < argc; i++)
        {

            if (strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--batch-size") == 0)
            {
                // Check if batch size is an integer
                if (i + 1 >= argc)
                {
                    printf("%serror:%s expected batch size after '%s' flag\n", RED, RESET, argv[i]);
                    exit(1);
                }
                if (sscanf(argv[++i], "%d%c", &batch_size, &c) != 1)
                {
                    printf("%serror:%s invalid batch size '%s'\n", RED, RESET, argv[i]);
                    exit(1);
                }
            }

            else if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--dims") == 0)
            {

                // Check if dimensions are comma-separated integers
                if (i + 1 >= argc)
                {
                    printf("%serror:%s expected dimensions after '%s' flag\n", RED, RESET, argv[i]);
                    exit(1);
                }

                char *token = strtok(argv[++i], ",");
                for (ndim = 1; ndim < 9 && token != NULL; ndim++)
                {
                    if (sscanf(token, "%d%c", &dims_hidden[ndim], &c) != 1)
                    {
                        printf("%serror:%s invalid dimensions '%s'\n", RED, RESET, token);
                        exit(1);
                    }
                    token = strtok(NULL, ",");
                }

                if (token != NULL)
                {
                    printf("%serror:%s not more than 8 hidden layers allowed", RED, RESET);
                    exit(1);
                }

                ++ndim; // output layer
            }

            else if (strcmp(argv[i], "-e") == 0 || strcmp(argv[i], "--epochs") == 0)
            {
                // Check if batch size is an integer
                if (i + 1 >= argc)
                {
                    printf("%serror:%s expected epochs after '%s' flag\n", RED, RESET, argv[i]);
                    exit(1);
                }
                if (sscanf(argv[++i], "%d%c", &epochs, &c) != 1)
                {
                    printf("%serror:%s invalid epochs '%s'\n", RED, RESET, argv[i]);
                    exit(1);
                }
            }

            else if (strcmp(argv[i], "-l") == 0 || strcmp(argv[i], "--learning-rate") == 0)
            {
                // Check if learning rate is a double
                if (i + 1 >= argc)
                {
                    printf("%serror:%s expected learning rate after '%s' flag\n", RED, RESET, argv[i]);
                    exit(1);
                }

                if (sscanf(argv[++i], "%lf%c", &learning_rate, &c) != 1)
                {
                    printf("%serror:%s invalid learning rate '%s'\n", RED, RESET, argv[i]);
                    exit(1);
                }
            }

            else
            {
                printf("%serror:%s unknown flag '%s'\n", RED, RESET, argv[i]);
                exit(1);
            }
        }

        return train(batch_size, ndim, dims_hidden, epochs, learning_rate);
    }

    else
    {
        printf("neural: '%s' is not a neural command. See 'neural --help'.\n", argv[1]);
        return 1;
    }

    return 0;
}
