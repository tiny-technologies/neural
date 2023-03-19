#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

// TYPES

typedef struct
{
    double *label;
    double *data;
} Image;

typedef struct
{
    Image *images;
    int size;
    int rows;
    int cols;
} Dataset;

// PRINT UTILS

const char *RESET = "\x1b[0m";
const char *BOLD = "\x1b[1m";
const char *RED = "\x1b[1;31m";
const char *GREEN = "\x1b[1;32m";

void print_array(int size, double *data)
{
    for (int i = 0; i < size; i++)
    {
        printf("%.2f ", data[i]);
    }
    printf("\n");
}

void print_image(Image image)
{

    printf("LABEL:");
    for (int i = 0; i < 10; i++)
    {
        printf(" %.0f", image.label[i]);
    }
    printf("\n");
    for (int j = 0; j < 28; j++)
    {
        for (int k = 0; k < 28; k++)
        {
            printf("%3d,", (int)(image.data[j * 28 + k] * 255));
        }
        printf("\n");
    }
}

// FUNCTIONAL UTILS

int arg_max(double *prediction)
{
    double max = 0.0;
    int index = 0;
    for (int i = 0; i < 10; i++)
    {
        if (prediction[i] > max)
        {
            index = i;
            max = prediction[i];
        };
    }
    return index;
}

double *random_array(int size)
{
    double *array = malloc(size * sizeof(double));
    for (int i = 0; i < size; i++)
    {
        array[i] = (double)rand() / ((double)RAND_MAX) - 0.5;
    }
    return array;
}

int read_network_order(FILE *file)
{
    int num;
    if (fread(&num, 4, 1, file) != 1)
    {
        printf("error: failed to read file\n");
        exit(1);
    };

    return ((num >> 24) & 0xff) |
           ((num << 8) & 0xff0000) |
           ((num >> 8) & 0xff00) |
           ((num << 24) & 0xff000000);
}

double timestamp()
{
    struct timespec start;
    clock_gettime(CLOCK_REALTIME, &start);
    return 0.000000001 * (1000000000 * start.tv_sec + start.tv_nsec);
}

// NETWORK

typedef struct Network
{
    double **weights;
    double **biases;
    int *dims;
    int ndim;
} Network;

Network network_create(int ndim, int *dims)
{
    Network network = {
        .weights = malloc(ndim * sizeof(double *)),
        .biases = malloc(ndim * sizeof(double *)),
        .dims = malloc(ndim * sizeof(int)),
        .ndim = ndim,
    };
    for (int i = 1; i < ndim; i++)
    {
        network.weights[i] = random_array(dims[i] * dims[i - 1]);
        network.biases[i] = random_array(dims[i]);
    }
    for (int i = 0; i < ndim; i++)
    {
        network.dims[i] = dims[i];
    }
    return network;
}

void network_destroy(Network network)
{
    for (int i = 1; i < network.ndim; i++)
    {
        free(network.weights[i]);
        free(network.biases[i]);
    }
    free(network.weights);
    free(network.biases);
    free(network.dims);
}

// MACHINE LEARNING

double compute_loss(int size, double *label, double *a)
{
    double loss = 0;
    for (int i = 0; i < size; i++)
    {
        double tmp = a[i] - label[i];
        loss += tmp * tmp;
    }
    return loss;
}

void forward(Network network, double **a)
{
    int ndim = network.ndim;
    int *dims = network.dims;
    double **w = network.weights;
    double **b = network.biases;

    for (int l = 1; l < ndim; l++)
    {
        for (int i = 0; i < dims[l]; i++)
        {
            a[l][i] = 0;
            for (int j = 0; j < dims[l - 1]; j++)
            {
                a[l][i] += w[l][i * dims[l - 1] + j] * a[l - 1][j];
            }
            a[l][i] += b[l][i];
            a[l][i] = 1.0 / (1.0 + exp(-a[l][i]));
        }
    }
}

void backward(Network network, double *label, double **a, double **w_grads, double **b_grads)
{
    int ndim = network.ndim;
    int *dims = network.dims;
    double **w = network.weights;

    int l = ndim - 1;
    for (int i = 0; i < dims[l]; i++)
    {
        b_grads[l][i] = 2 * (a[l][i] - label[i]) * a[l][i] * (1 - a[l][i]);
        for (int j = 0; j < dims[l - 1]; j++)
        {
            w_grads[l][i * dims[l - 1] + j] = a[l - 1][j] * b_grads[l][i];
        }
    }

    for (int l = ndim - 2; l > 0; l--)
    {
        for (int i = 0; i < dims[l]; i++)
        {
            b_grads[l][i] = 0;
            for (int j = 0; j < dims[l + 1]; j++)
            {
                b_grads[l][i] += w[l + 1][j * dims[l] + i] * b_grads[l + 1][j];
            }
            b_grads[l][i] *= a[l][i] * (1 - a[l][i]);

            for (int j = 0; j < network.dims[l - 1]; j++)
            {
                w_grads[l][i * dims[l - 1] + j] = a[l - 1][j] * b_grads[l][i];
            }
        }
    }
}

double update_mini_batch(Network network, Image *images, int batch_size, double learning_rate)
{
    int ndim = network.ndim;
    int *dims = network.dims;

    // allocate variable arrays
    double **neurons = malloc(ndim * sizeof(double *));

    double **weights_grads = malloc(ndim * sizeof(double *));
    double **biases_grads = malloc(ndim * sizeof(double *));

    double **weights_grad = malloc(ndim * sizeof(double *));
    double **biases_grad = malloc(ndim * sizeof(double *));

    neurons[0] = calloc(sizeof(double), dims[0]);
    for (int l = 1; l < ndim; l++)
    {
        neurons[l] = calloc(sizeof(double), dims[l]);

        weights_grads[l] = calloc(sizeof(double), dims[l] * dims[l - 1]);
        biases_grads[l] = calloc(sizeof(double), dims[l]);

        weights_grad[l] = calloc(sizeof(double), dims[l] * dims[l - 1]);
        biases_grad[l] = calloc(sizeof(double), dims[l]);
    }

    // accumulate intermediate gradients from backpropagation
    double loss = 0;
    for (int b = 0; b < batch_size; b++)
    {
        neurons[0] = images[b].data;
        forward(network, neurons);
        loss += compute_loss(dims[ndim - 1], images[b].label, neurons[ndim - 1]);

        backward(network, images[b].label, neurons, weights_grads, biases_grads);

        for (int l = 1; l < ndim; l++)
        {
            for (int i = 0; i < dims[l]; i++)
            {
                biases_grad[l][i] += biases_grads[l][i];
                for (int j = 0; j < dims[l - 1]; j++)
                {
                    int idx = i * dims[l - 1] + j;
                    weights_grad[l][idx] += weights_grads[l][idx];
                }
            }
        }
    }

    // update weights and biases
    double factor = learning_rate / batch_size;
    for (int l = 1; l < ndim; l++)
    {
        for (int i = 0; i < dims[l]; i++)
        {
            network.biases[l][i] -= factor * biases_grad[l][i];
            for (int j = 0; j < dims[l - 1]; j++)
            {
                int idx = i * dims[l - 1] + j;
                network.weights[l][idx] -= factor * weights_grad[l][idx];
            }
        }
    }

    // free variable arrays
    for (int i = 1; i < network.ndim; i++)
    {
        free(neurons[i]);
        free(weights_grads[i]);
        free(biases_grads[i]);
        free(weights_grad[i]);
        free(biases_grad[i]);
    }

    free(neurons);
    free(weights_grads);
    free(biases_grads);
    free(weights_grad);
    free(biases_grad);

    return loss;
}

void epoch(Network network, Dataset dataset, int batch_size, double learning_rate)
{
    // todo implement loading thing
    // Working... ━━━━╺━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  10% 0:00:46
    int batches = dataset.size / batch_size;
    printf("Start epoch with %d batches (batch_size: %d)\n", batches, batch_size);
    for (int i = 0; i < batches; i++)
    {
        double loss = update_mini_batch(network, dataset.images, batch_size, learning_rate) / batch_size;
        printf("loss: %.4f\r", loss);
    }
    printf("\n");
}

// IO

Dataset load_mnist_dataset(char *path_to_labels, char *path_to_images)
{
    Dataset dataset;

    {
        FILE *file = fopen(path_to_labels, "rb");
        if (file == NULL)
        {
            printf("file not found\n");
            exit(1);
        }

        fseek(file, 4, SEEK_SET); // skip magic number
        dataset.size = read_network_order(file);
        dataset.images = malloc(sizeof(Image) * dataset.size);

        uint8_t number;
        for (int i = 0; i < dataset.size; i++)
        {
            if (fread(&number, 1, 1, file) != 1)
            {
                printf("error: failed to read label from file\n");
                exit(1);
            };
            dataset.images[i].label = calloc(sizeof(double), 10);
            dataset.images[i].label[number] = 1.0;
        }

        fclose(file);
    }
    {
        FILE *file = fopen(path_to_images, "rb");
        if (file == NULL)
        {
            printf("file not found\n");
            exit(1);
        }

        fseek(file, 4, SEEK_SET); // skip magic number
        dataset.size = read_network_order(file);
        dataset.rows = read_network_order(file);
        dataset.cols = read_network_order(file);

        int pixel = dataset.rows * dataset.cols;
        uint8_t buffer[pixel];
        for (int i = 0; i < dataset.size; i++)
        {
            dataset.images[i].data = malloc(sizeof(double) * pixel);
            if (fread(buffer, 1, pixel, file) != pixel)
            {
                exit(1);
            };
            for (int j = 0; j < pixel; j++)
            {
                dataset.images[i].data[j] = ((double)buffer[j]) / 255.0;
            }
        }

        fclose(file);
    }

    return dataset;
}

// SERIALIZATION / DESERIALIZATION

/*
 * SERIALIZATION FORMAT
 * SECTION | ndim |   dims   |          w[1]         |     b[1]    | ... |
 * SIZE    |   4  | 4 * ndim | 8 * dims[1] * dims[0] | 8 * dims[1] | ... |
 */

void serialize_network(Network network, FILE *file)
{
    int ndim = network.ndim;
    int *dims = network.dims;

    fwrite(&ndim, sizeof(int32_t), 1, file);
    for (int l = 0; l < ndim; l++)
    {
        fwrite(dims + l, sizeof(int32_t), 1, file);
    }

    for (int l = 1; l < ndim; l++)
    {
        fwrite(network.weights[l], sizeof(double), dims[l] * dims[l - 1], file);
        fwrite(network.biases[l], sizeof(double), dims[l], file);
    }
}

// todo: make this platform independent
Network deserialize_network(FILE *file)
{
    int ndim;
    int failures = fread(&ndim, sizeof(int32_t), 1, file) != 1;

    int *dims = malloc(ndim * sizeof(int32_t));

    for (int l = 0; l < ndim; l++)
    {
        failures += fread(dims + l, sizeof(int32_t), 1, file) != 1;
    }

    Network network = network_create(ndim, dims);

    for (int l = 1; l < ndim; l++)
    {
        failures += fread(network.weights[l], sizeof(double), dims[l] * dims[l - 1], file) != dims[l] * dims[l - 1];
        failures += fread(network.biases[l], sizeof(double), dims[l], file) != dims[l];
    }

    if (failures)
    {
        printf("error: failed to load read network from file %d\n", failures);
        exit(1);
    }

    return network;
}
