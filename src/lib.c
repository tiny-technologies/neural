#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

// TYPES

typedef struct Image
{
    double *label;
    double *data;
} Image;

typedef struct Dataset
{
    Image *images;
    int size;
    int rows;
    int cols;
} Dataset;

typedef struct Array
{
    double *data;
    int *size;
} Array;

// MATH

void print_matrix(int rows, int cols, double *matrix)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%.1f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

void print_vector(int size, double *vector)
{
    for (int i = 0; i < size; i++)
    {
        printf("%.2f ", vector[i]);
    }
    printf("\n");
}

void dot(int rows, int cols, double *matrix, double *vector, double *result)
{
    for (int i = 0; i < rows; i++)
    {
        result[i] = 0;
        for (int j = 0; j < cols; j++)
        {
            result[i] += matrix[i * cols + j] * vector[j];
        }
    }
}

void add(int size, double *a, double *b, double *result)
{
    for (int i = 0; i < size; i++)
    {
        result[i] = a[i] + b[i];
    }
}

void sub(int size, double *a, double *b, double *result)
{
    for (int i = 0; i < size; i++)
    {
        result[i] = a[i] - b[i];
    }
}

void scale(int size, double factor, double *a, double *result)
{
    for (int i = 0; i < size; i++)
    {
        result[i] = factor * a[i];
    }
}

void sigmoid(int size, double *a, double *result)
{
    for (int i = 0; i < size; i++)
    {
        result[i] = 1.0 / (1.0 + exp(-a[i]));
    }
}

// UTILS
double *random_array(int size)
{
    double *array = malloc(size * sizeof(double));
    for (int i = 0; i < size; i++)
    {
        array[i] = (double)rand() / ((double)RAND_MAX) - 0.5;
    }
    return array;
}

// NETWORK

typedef struct Network
{
    double *w1;
    double *b1;
    double *w2;
    double *b2;
    int s0;
    int s1;
    int s2;
} Network;

Network make_network(int s0, int s1, int s2)
{
    Network network = {
        .w1 = random_array(s0 * s1),
        .b1 = random_array(s1),
        .w2 = random_array(s1 * s2),
        .b2 = random_array(s2),
        .s0 = s0,
        .s1 = s1,
        .s2 = s2,
    };
    return network;
}

void destory_network(Network network)
{
    free(network.w1);
    free(network.b2);
    free(network.w2);
    free(network.b1);
}

void feed_forward(Network network, double *input, double *output)
{
    // first layer
    double a1[network.s1];
    dot(network.s1, network.s0, network.w1, input, a1);
    add(network.s1, network.b1, a1, a1);
    sigmoid(network.s1, a1, a1);
    // second layer
    dot(network.s2, network.s1, network.w2, a1, output);
    add(network.s2, network.b2, output, output);
    sigmoid(network.s2, output, output);
}

double back_propagation(Network network, double *input, double *label, double *nabla_ws[], double *nabla_bs[])
{
    double *a1 = malloc(network.s1 * sizeof(double));
    double *a2 = malloc(network.s2 * sizeof(double));

    // forward pass
    dot(network.s1, network.s0, network.w1, input, a1);
    add(network.s1, network.b1, a1, a1);
    sigmoid(network.s1, a1, a1);
    // second layer
    dot(network.s2, network.s1, network.w2, a1, a2);
    add(network.s2, network.b2, a2, a2);
    sigmoid(network.s2, a2, a2);

    // backward pass
    for (int i = 0; i < network.s2; i++)
    {
        nabla_bs[1][i] = 2 * (a2[i] - label[i]) * (a2[i] * (1 - a2[i]));
        for (int j = 0; j < network.s1; j++)
        {
            nabla_ws[1][i * network.s1 + j] = a1[j] * nabla_bs[1][i];
        }
    }

    for (int i = 0; i < network.s1; i++)
    {
        nabla_bs[0][i] = 0;
        // wT * delta
        for (int j = 0; j < network.s2; j++)
        {
            nabla_bs[0][i] += network.w2[j * network.s2 + i] * nabla_bs[1][j];
        }
        nabla_bs[0][i] *= 2 * a1[i] * (1 - a1[i]);

        for (int j = 0; j < network.s0; j++)
        {
            nabla_ws[0][i * network.s0 + j] = input[j] * nabla_bs[1][i];
        }
    }

    double loss = 0;
    for (int i = 0; i < network.s2; i++)
    {
        loss += a2[i] - label[i];
    }
    loss *= loss;

    free(a1);
    free(a2);

    return loss;
}

double update_mini_batch(Network network, Image *images, int batch_size)
{
    double *nabla_w[] = {
        calloc(sizeof(double), network.s0 * network.s1),
        calloc(sizeof(double), network.s1 * network.s2),
    };
    double *nabla_b[] = {
        calloc(sizeof(double), network.s1),
        calloc(sizeof(double), network.s2),
    };

    double *nabla_ws[] = {
        malloc(network.s0 * network.s1 * sizeof(double)),
        malloc(network.s1 * network.s2 * sizeof(double)),
    };
    double *nabla_bs[] = {
        malloc(network.s1 * sizeof(double)),
        malloc(network.s2 * sizeof(double)),
    };

    double loss = 0;
    for (int i = 0; i < batch_size; i++)
    {
        loss += back_propagation(network, images[i].data, images[i].label, nabla_ws, nabla_bs);
        add(network.s0 * network.s1, nabla_w[0], nabla_ws[0], nabla_w[0]);
        add(network.s1 * network.s2, nabla_w[1], nabla_ws[1], nabla_w[1]);
        add(network.s1, nabla_b[0], nabla_bs[0], nabla_b[0]);
        add(network.s2, nabla_b[1], nabla_bs[1], nabla_b[1]);
    }

    double learning_rate = 0.01;
    double factor = learning_rate / batch_size;
    scale(network.s0 * network.s1, factor, nabla_w[0], nabla_w[0]);
    scale(network.s1 * network.s2, factor, nabla_w[1], nabla_w[1]);
    scale(network.s1, factor, nabla_b[0], nabla_b[0]);
    scale(network.s2, factor, nabla_b[1], nabla_b[1]);

    sub(network.s0 * network.s1, network.w1, nabla_w[0], network.w1);
    sub(network.s1 * network.s2, network.w2, nabla_w[1], network.w2);
    sub(network.s1, network.b1, nabla_b[0], network.b1);
    sub(network.s2, network.b2, nabla_b[1], network.b2);

    for (int i = 0; i < 2; i++)
    {
        free(nabla_w[i]);
        free(nabla_b[i]);
        free(nabla_ws[i]);
        free(nabla_bs[i]);
    }

    return loss;
}

void epoch(Network network, Dataset dataset, int batch_size)
{
    // todo implement loading thing
    // Working... ━━━━╺━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  10% 0:00:46
    int batches = dataset.size / batch_size;
    printf("Start epoch with %d batches (batch_size: %d)\n", batches, batch_size);
    for (int i = 0; i < batches; i++)
    {
        double loss = update_mini_batch(network, dataset.images, batch_size) / batch_size;
        printf("loss: %.2f\r", loss);
    }
    printf("\n");
}

// IO

void load_images(char *path)
{
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

void store_network()
{
}

void load_network()
{
}

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

// CLI

int train()
{
    Dataset dataset = load_mnist_dataset("mnist/train-labels-idx1-ubyte", "mnist/train-images-idx3-ubyte");
    printf("loaded dataset with %d images\n", dataset.size);

    Network network = make_network(dataset.rows * dataset.cols, 64, 10);
    for (int i = 0; i < 10; i++)
    {
        printf("Epoch: %d\n", i);
        epoch(network, dataset, 100);
    }

    int predicted_correctly = 0;
    for (int i = 0; i < dataset.size; i++)
    {

        // print_image(dataset.images[i]);
        double output[10];
        feed_forward(network, dataset.images[i].data, output);
        predicted_correctly += arg_max(output) == arg_max(dataset.images[i].label);
    }
    printf("predicted: %d, accurarcy: %f", predicted_correctly, ((double)predicted_correctly) / dataset.size);

    return 0;
}

int run()
{
    printf("not implemented yet\n");
    return 0;
}
