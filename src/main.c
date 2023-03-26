#include <ctype.h>
#include "lib.c"

// SUBCOMMANDS

int train(Network network, Dataset dataset, int batch_size, int epochs, double learning_rate, char *model_path)
{
    // training
    {
        printf("start training with learning rate of %s%.4lf%s and %s%d%s epochs\n",
               BOLD, learning_rate, RESET, BOLD, epochs, RESET);
        for (int i = 0; i < epochs; i++)
        {
            printf("%sEpoch %d%s\n", BOLD, i, RESET);
            epoch(network, dataset, batch_size, learning_rate);
        }

        destroy_dataset(dataset);
    }

    // validation
    {
        Dataset dataset = load_mnist_dataset("mnist/t10k-labels-idx1-ubyte", "mnist/t10k-images-idx3-ubyte");
        printf("loaded validation dataset with %d images\n", dataset.size);

        int predicted_correctly = 0;
        for (int i = 0; i < dataset.size; i++)
        {
            forward(network, dataset.images[i].data);
            predicted_correctly += arg_max(network.neurons[network.ndim - 1]) == arg_max(dataset.images[i].label);
        }
        printf("predicted: %d, accurarcy: %f\n", predicted_correctly, ((double)predicted_correctly) / dataset.size);

        destroy_dataset(dataset);
    }

    // persistence
    {
        FILE *file = fopen(model_path, "wb");
        serialize_network(network, file);
        printf("saved model to: '%s'\n", model_path);
        fclose(file);
    }

    network_destroy(network);

    return 0;
}

int bench()
{
    Dataset dataset = load_mnist_dataset("mnist/train-labels-idx1-ubyte", "mnist/train-images-idx3-ubyte");
    printf("loaded training dataset with %d images\n", dataset.size);

    int ndim = 5;
    int dims[] = {
        dataset.rows * dataset.cols,
        1024,
        1024,
        1024,
        10,
    };
    Network network = network_create(ndim, dims);
    printf("created network of size %d", dims[0]);
    for (int i = 1; i < ndim; i++)
    {
        printf("x%d", dims[i]);
    }
    printf("\n");

    int n_passes = 100;

    {
        printf("%sForward Pass%s \U0001f51c\n", BOLD, RESET);
        double start = timestamp();
        for (int i = 0; i < n_passes; i++)
        {
            forward(network, dataset.images[i].data);
        }
        double end = timestamp();
        printf("took: %.3f seconds (%d passes)\n", end - start, n_passes);
    }

    {
        printf("%sBackward Pass%s \U0001f519\n", BOLD, RESET);
        double start = timestamp();
        for (int i = 0; i < n_passes; i++)
        {
            backward(network, dataset.images[i].label);
        }
        double end = timestamp();
        printf("took: %.3f seconds (%d passes)\n", end - start, n_passes);
    }

    destroy_dataset(dataset);

    return 0;
}

int run(char *model_path, char *image_path)
{
    Network network = load_network(model_path);
    double *data = load_pgm_image(image_path);

    forward(network, data);
    int prediction = arg_max(network.neurons[network.ndim - 1]);

    printf("model prediction: %s%d%s\n", BOLD, prediction, RESET);

    network_destroy(network);
    free(data);

    return 0;
}

int test(char *model_path)
{
    Network network = load_network(model_path);
    Dataset dataset = load_mnist_dataset("mnist/t10k-labels-idx1-ubyte", "mnist/t10k-images-idx3-ubyte");
    printf("loaded dataset with %d images\n", dataset.size);

    int predicted_correctly = 0;
    for (int i = 0; i < dataset.size; i++)
    {
        forward(network, dataset.images[i].data);
        predicted_correctly += arg_max(network.neurons[network.ndim - 1]) == arg_max(dataset.images[i].label);
    }
    printf("predicted: %d, accurarcy: %f\n", predicted_correctly, ((double)predicted_correctly) / dataset.size);

    network_destroy(network);
    destroy_dataset(dataset);

    return 0;
}

// USAGE

int print_usage_main()
{
    printf("Usage:\n");
    printf("\n");
    printf("    %sneural <command> [<args>]%s\n", BOLD, RESET);
    printf("Commands:\n");
    printf("\n");
    printf("    %srun%s    Run inference using a trained network\n", BOLD, RESET);
    printf("      %s<path>%s                      path to model\n", BOLD, RESET);
    printf("      %s<path>%s                      path to PGM P5 image \n", BOLD, RESET);
    printf("\n");
    printf("    %stest%s   Test the accurary of a trained network\n", BOLD, RESET);
    printf("      %s<path>%s                      path to model (default: default.model)\n", BOLD, RESET);
    printf("\n");
    printf("    %strain%s  Train a new network and store it to disk\n", BOLD, RESET);
    printf("      %s-b, --batch-size <INT>%s      samples per batch (default: 200)\n", BOLD, RESET);
    printf("      %s-d, --dims <INT,INT,..>%s     dimensions of hidden layers (default: 16,16)\n", BOLD, RESET);
    printf("      %s-e, --epochs <INT>%s          number of epochs (default: 10)\n", BOLD, RESET);
    printf("      %s-l, --learning-rate <REAL>%s  step size of parameter update (default: 0.01)\n", BOLD, RESET);
    printf("      %s-i, --input <PATH>%s          path to model used as starting point (optional)\n", BOLD, RESET);
    printf("      %s-o, --output <PATH>%s         output path of the trained model (default: default.model)\n", BOLD, RESET);
    printf("\n");
    printf("    %sbench%s  Benchmark forward and backward pass\n", BOLD, RESET);
    printf("\n");
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
        if (argc != 4)
        {
            printf("%serror:%s unexpected number of arguments\n", RED, RESET);
            exit(1);
        }
        else if (argc > 4)
        {
            printf("%serror:%s unexpected argument '%s'\n", RED, RESET, argv[4]);
            exit(1);
        }

        return run(argv[2], argv[3]);
    }

    else if (strcmp(argv[1], "test") == 0)
    {
        if (argc > 3)
        {
            printf("%serror:%s unexpected argument '%s'\n", RED, RESET, argv[3]);
            exit(1);
        }

        return test(argc == 2 ? "default.model" : argv[2]);
    }

    else if (strcmp(argv[1], "train") == 0)
    {
        // default values
        int batch_size = 200;
        char *dims_string = NULL;
        int epochs = 10;
        double learning_rate = 0.01;
        char *output_path = "default.model";
        char *input_path = NULL;

        // parse optional flags
        for (int i = 2; i < argc; i++)
        {

            if (strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--batch-size") == 0)
            {
                if (i + 1 >= argc)
                {
                    printf("%serror:%s expected batch size after '%s' flag\n", RED, RESET, argv[i]);
                    exit(1);
                }

                // Check if batch size is an integer
                char c;
                if (sscanf(argv[++i], "%d%c", &batch_size, &c) != 1)
                {
                    printf("%serror:%s invalid batch size '%s'\n", RED, RESET, argv[i]);
                    exit(1);
                }
            }

            else if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--dims") == 0)
            {
                if (i + 1 >= argc)
                {
                    printf("%serror:%s expected dimensions after '%s' flag\n", RED, RESET, argv[i]);
                    exit(1);
                }

                dims_string = argv[++i];
            }

            else if (strcmp(argv[i], "-e") == 0 || strcmp(argv[i], "--epochs") == 0)
            {
                if (i + 1 >= argc)
                {
                    printf("%serror:%s expected epochs after '%s' flag\n", RED, RESET, argv[i]);
                    exit(1);
                }

                // Check if batch size is an integer
                char c;
                if (sscanf(argv[++i], "%d%c", &epochs, &c) != 1)
                {
                    printf("%serror:%s invalid epochs '%s'\n", RED, RESET, argv[i]);
                    exit(1);
                }
            }

            else if (strcmp(argv[i], "-l") == 0 || strcmp(argv[i], "--learning-rate") == 0)
            {
                if (i + 1 >= argc)
                {
                    printf("%serror:%s expected learning rate after '%s' flag\n", RED, RESET, argv[i]);
                    exit(1);
                }

                // Check if learning rate is a real number
                char c;
                if (sscanf(argv[++i], "%lf%c", &learning_rate, &c) != 1)
                {
                    printf("%serror:%s invalid learning rate '%s'\n", RED, RESET, argv[i]);
                    exit(1);
                }
            }

            else if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--input") == 0)
            {
                if (i + 1 >= argc)
                {
                    printf("%serror:%s expected path after '%s' flag\n", RED, RESET, argv[i]);
                    exit(1);
                }

                input_path = argv[++i];
            }

            else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0)
            {
                if (i + 1 >= argc)
                {
                    printf("%serror:%s expected path after '%s' flag\n", RED, RESET, argv[i]);
                    exit(1);
                }

                output_path = argv[++i];
            }

            else
            {
                printf("%serror:%s unknown flag '%s'\n", RED, RESET, argv[i]);
                exit(1);
            }
        }

        // load dataset
        Dataset dataset = load_mnist_dataset("mnist/train-labels-idx1-ubyte", "mnist/train-images-idx3-ubyte");
        printf("loaded dataset with %d images\n", dataset.size);

        // initialize network
        Network network;
        if (input_path != NULL)
        {
            if (dims_string != NULL)
            {
                printf("%serror:%s --dims and --input flags are not compatible\n", RED, RESET);
                exit(1);
            }

            network = load_network(input_path);
        }
        else
        {
            int ndim;
            if (dims_string == NULL)
            {
                ndim = 4;
            }
            else
            {
                ndim = 3;
                for (char *c = dims_string; *c != '\0'; c++)
                {
                    ndim += *c == ',';
                    if (!isdigit(*c) && *c != ',')
                    {
                        printf("%serror:%s invalid character '%c' in '%s'\n",
                               RED, RESET, *c, dims_string);
                        exit(1);
                    }
                }
            }

            int *dims = malloc(ndim * sizeof(int));

            char *c = dims_string;
            dims[0] = 784;
            for (int l = 1; l < ndim - 1; l++)
            {
                if (dims_string == NULL)
                {
                    dims[l] = 16;
                }
                else
                {
                    dims[l] = atoi(c);
                    if (!dims[l])
                    {
                        printf("%serror:%s invalid value '%s' for --dims flag\n", RED, RESET, dims_string);
                        exit(1);
                    }
                    c = strchr(c, ',') + 1;
                }
            }
            dims[ndim - 1] = 10;
            network = network_create(ndim, dims);
            free(dims);

            printf("initialized network with layers: %s%d", BOLD, network.dims[0]);
            for (int i = 1; i < network.ndim; i++)
                printf("x%d", network.dims[i]);
            printf("%s\n", RESET);
        }

        return train(network, dataset, batch_size, epochs, learning_rate, output_path);
    }

    else if (strcmp(argv[1], "bench") == 0)
    {
        if (argc == 2)
        {
            return bench();
        }
        else
        {
            printf("%serror:%s unknown flag '%s'\n", RED, RESET, argv[argc - 1]);
            exit(1);
        }
    }

    else
    {
        printf("neural: '%s' is not a neural command. See 'neural --help'.\n", argv[1]);
        return 1;
    }

    return 0;
}
