typedef struct Array
{
    double *data;
    int *shape;
    int ndim;
    int size;
} Array;

Array array_new(int ndim, int *shape)
{
    int size = 1;
    for (int i = 0; i < ndim; i++)
    {
        size *= shape[i];
    }
    Array array = {
        .data = malloc(size * sizeof(double)),
        .ndim = ndim,
        .shape = shape,
    };
    return array;
}

Array array_new_1d(int size)
{
    int *shape = malloc(sizeof(int));
    shape[0] = size;
    return array_new(1, shape);
}

Array array_new_2d(int size_0, int size_1)
{
    int *shape = malloc(sizeof(int));
    shape[0] = size_0;
    shape[1] = size_1;
    return array_new(2, shape);
}

Array array_free(Array array)
{
    free(array.data);
    free(array.shape);
}

Array array_add(Array a, Array b, Array out)
{
    if (a.ndim != b.ndim)
        printf("error: could not add arrays with different dimensions %d and %d",
               a.ndim, b.ndim);
    // work in progress
    // for (int i = 0; i < 0)
    exit(1);
}