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
const char *CLEAR = "\x1b[A\33[2KT\r";

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

void print_progress(int progress, int max, int duration)
{
    int n_blocks = 32;
    int blocks = n_blocks * progress / max;
    int percentage = 100 * progress / max;

    printf("%s", GREEN);
    for (int j = 0; j < blocks; j++)
    {
        printf("\U00002501");
    }
    printf(progress != max ? "%s\U0000257a" : "\U00002501%s", RESET);
    for (int j = 0; j < n_blocks - blocks; j++)
    {
        printf("\U00002501");
    }

    int seconds = duration % 60;
    int minutes = duration / 60;
    printf(" %s%3d%%%s %02d:%02d\n", GREEN, percentage, RESET, minutes, seconds);
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
        printf("%serror:%s failed to read file\n", RED, RESET);
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
    double **neurons;
    double **weights;
    double **biases;
    double **weights_grad;
    double **biases_grad;
    int *dims;
    int ndim;
} Network;

Network network_create(int ndim, int *dims)
{
    Network network = {
        .neurons = malloc(ndim * sizeof(double *)),
        .weights = malloc(ndim * sizeof(double *)),
        .biases = malloc(ndim * sizeof(double *)),
        .weights_grad = malloc(ndim * sizeof(double *)),
        .biases_grad = malloc(ndim * sizeof(double *)),
        .dims = malloc(ndim * sizeof(int)),
        ndim = ndim,
    };
    for (int i = 1; i < ndim; i++)
    {
        network.neurons[i] = random_array(dims[i]);
        network.weights[i] = random_array(dims[i] * dims[i - 1]);
        network.biases[i] = random_array(dims[i]);
        network.weights_grad[i] = malloc(dims[i] * dims[i - 1] * sizeof(double));
        network.biases_grad[i] = malloc(dims[i] * sizeof(double));
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
        free(network.neurons[i]);
        free(network.weights[i]);
        free(network.biases[i]);
        free(network.weights_grad[i]);
        free(network.biases_grad[i]);
    }
    free(network.neurons);
    free(network.weights);
    free(network.biases);
    free(network.weights_grad);
    free(network.biases_grad);
    free(network.dims);
}

// MACHINE LEARNING

double compute_loss(Network network, double *label)
{
    double loss = 0;
    for (int i = 0; i < network.dims[network.ndim - 1]; i++)
    {
        double tmp = network.neurons[network.ndim - 1][i] - label[i];
        loss += tmp * tmp;
    }
    return loss;
}

void forward(Network network, double *inputs)
{
    network.neurons[0] = inputs;

    int ndim = network.ndim;
    int *dims = network.dims;
    double **a = network.neurons;
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

void backward(Network network, double *label)
{
    int ndim = network.ndim;
    int *dims = network.dims;
    double **a = network.neurons;
    double **w = network.weights;
    double **w_grad = network.weights_grad;
    double **b_grad = network.biases_grad;

    int l = ndim - 1;
    for (int i = 0; i < dims[l]; i++)
    {
        b_grad[l][i] = 2 * (a[l][i] - label[i]) * a[l][i] * (1 - a[l][i]);
        for (int j = 0; j < dims[l - 1]; j++)
        {
            w_grad[l][i * dims[l - 1] + j] = a[l - 1][j] * b_grad[l][i];
        }
    }

    for (int l = ndim - 2; l > 0; l--)
    {
        for (int i = 0; i < dims[l]; i++)
        {
            b_grad[l][i] = 0;
            for (int j = 0; j < dims[l + 1]; j++)
            {
                b_grad[l][i] += w[l + 1][j * dims[l] + i] * b_grad[l + 1][j];
            }
            b_grad[l][i] *= a[l][i] * (1 - a[l][i]);

            for (int j = 0; j < network.dims[l - 1]; j++)
            {
                w_grad[l][i * dims[l - 1] + j] = a[l - 1][j] * b_grad[l][i];
            }
        }
    }
}

double update_mini_batch(Network network, Image *images, int batch_size, double learning_rate)
{
    int ndim = network.ndim;
    int *dims = network.dims;

    // allocate arrays to accumulate gradients
    double **weights_grad = malloc(ndim * sizeof(double *));
    double **biases_grad = malloc(ndim * sizeof(double *));

    for (int l = 1; l < ndim; l++)
    {
        weights_grad[l] = calloc(sizeof(double), dims[l] * dims[l - 1]);
        biases_grad[l] = calloc(sizeof(double), dims[l]);
    }

    // accumulate intermediate gradients from backpropagation
    double loss = 0;
    for (int b = 0; b < batch_size; b++)
    {
        forward(network, images[b].data);
        loss += compute_loss(network, images[b].label);

        backward(network, images[b].label);

        for (int l = 1; l < ndim; l++)
        {
            for (int i = 0; i < dims[l]; i++)
            {
                biases_grad[l][i] += network.biases_grad[l][i];
                for (int j = 0; j < dims[l - 1]; j++)
                {
                    int idx = i * dims[l - 1] + j;
                    weights_grad[l][idx] += network.weights_grad[l][idx];
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

    // free gradient arrays
    for (int i = 1; i < network.ndim; i++)
    {
        free(weights_grad[i]);
        free(biases_grad[i]);
    }
    free(weights_grad);
    free(biases_grad);

    return loss;
}

void epoch(Network network, Dataset dataset, int batch_size, double learning_rate)
{
    double start = timestamp();
    int batches = dataset.size / batch_size;
    printf("Start epoch with %d batches (batch_size: %d)\n", batches, batch_size);
    double loss = 0;
    for (int i = 0; i < batches; i++)
    {
        printf("%sloss: %.4lf ", CLEAR, loss / (i + 1));
        print_progress(i, batches, (int)timestamp() - start);

        loss += update_mini_batch(network, dataset.images + i * batch_size, batch_size, learning_rate) / batch_size;
    }

    printf("%sloss: %.4lf ", CLEAR, loss / batches);
    print_progress(batches, batches, (int)timestamp() - start);
}

// IO

double *load_pgm_image(char *path)
{
    FILE *file = fopen(path, "rb");
    if (file == NULL)
    {
        printf("%serror:%s cannot open file\n", RED, RESET);
        exit(1);
    }

    // check magic number
    char magic[3];
    if (fread(magic, 1, 3, file) != 3)
    {
        printf("%serror:%s failed to read PGM header\n", RED, RESET);
        exit(1);
    }
    else if (!(magic[0] == 0x50 && magic[1] == 0x35 && magic[2] == 0x0a))
    {
        printf("%serror:%s file '%s' is not in PGM P5 format\n", RED, RESET, path);
        exit(1);
    };

    // skip optional comments
    int c;
    while ((c = fgetc(file)) == '#')
    {
        while (fgetc(file) != '\n')
        {
        }
    }
    ungetc(c, file);

    // check image dimensions and maxval
    int width, height, maxval;
    if (fscanf(file, "%d %d\n%d\n", &width, &height, &maxval) != 3)
    {
        printf("%serror:%s failed to parse PGM header\n", RED, RESET);
        exit(1);
    }

    if (width != 28 || height != 28)
    {
        printf("%serror:%s image dimensions must be 28x28\n", RED, RESET);
        exit(1);
    }

    if (maxval != 255)
    {
        printf("%serror:%s expected maxval of 255\n", RED, RESET);
        exit(1);
    }

    // read pixel data
    uint8_t image_data[28 * 28];
    if (fread(image_data, sizeof(uint8_t), 28 * 28, file) != 28 * 28)
    {
        printf("%serror:%s failed to read pixel data\n", RED, RESET);
        exit(1);
    }
    fclose(file);

    double *pixel = malloc(width * height * sizeof(double));
    for (int i = 0; i < 28 * 28; i++)
    {
        pixel[i] = ((double)image_data[i]) / 255.0;
    }

    return pixel;
}

Dataset load_mnist_dataset(char *path_to_labels, char *path_to_images)
{
    Dataset dataset;

    {
        FILE *file = fopen(path_to_labels, "rb");
        if (file == NULL)
        {
            printf("%serror:%s file '%s' not found\n", RED, RESET, path_to_labels);
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
                printf("%serror:%s failed to read label from file\n", RED, RESET);
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
            printf("%serror:%s file '%s' not found\n", RED, RESET, path_to_images);
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
            if (fread(buffer, 1, pixel, file) != (unsigned)pixel)
            {
                printf("%serror:%s failed to read images from file\n", RED, RESET);
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

void destroy_dataset(Dataset dataset)
{
    for (int i = 0; i < dataset.size; i++)
    {
        free(dataset.images[i].label);
        free(dataset.images[i].data);
    }
    free(dataset.images);
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
        failures += fread(network.weights[l], sizeof(double), dims[l] * dims[l - 1], file) != (unsigned)dims[l] * dims[l - 1];
        failures += fread(network.biases[l], sizeof(double), dims[l], file) != (unsigned)dims[l];
    }

    if (failures)
    {
        printf("%serror:%s failed to load read network from file %d\n", RED, RESET, failures);
        exit(1);
    }

    free(dims);

    return network;
}

Network load_network(char *path)
{

    FILE *file = fopen(path, "rb");
    if (file == NULL)
    {
        printf("%serror:%s '%s' does not exist\n", RED, RESET, path);
        exit(1);
    }

    Network network = deserialize_network(file);
    fclose(file);

    printf("info: loaded model '%s' with size %d", path, network.dims[0]);
    for (int i = 1; i < network.ndim; i++)
    {
        printf("x%d", network.dims[i]);
    }
    printf("\n");

    return network;
}
