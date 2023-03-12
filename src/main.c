#include "test.c"

void print_usage_main()
{
    printf("Usage:\n\n");
    printf("    %sneural <command> [<args>]%s\n\n", BOLD, RESET);
    printf("These are common commands:\n\n");
    printf("    %srun%s    Run inference using a trained network\n", BOLD, RESET);
    printf("    %strain%s  Train a new network and store it to disk\n", BOLD, RESET);
    printf("    %stest%s   Run unit tests\n", BOLD, RESET);
    printf("\n");
}

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
    else if (strcmp(argv[1], "test") == 0)
    {
        return test(argv + 2, argc - 2);
    }
    else
    {
        printf("neural: '%s' is not a neural command. See 'neural --help'.\n", argv[1]);
        return 1;
    }

    return 0;
}
