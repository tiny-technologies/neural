#include <locale.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <wchar.h>

#include "lib.c"

int n_passed = 0;
int n_failed = 0;

char **test_names;
int n_tests;

// UTILS

int assert_fails = 0;

void assert_array(char *name, int size, double *expected, double *actual)
{
    int equals = 1;
    for (int i = 0; i < size; i++)
    {
        if (fabs(expected[i] - actual[i]) > 0.000001)
        {
            equals = 0;
            break;
        }
    }

    if (!equals)
    {
        assert_fails += 1;
        printf("array comparison %s\"%s\"%s failed\n", BOLD, name, RESET);
        printf("expected: ");
        print_array(size, expected);
        printf("  actual: ");
        print_array(size, actual);
        printf("\n");
    }
}

void assert_scalar(char *name, double expected, double actual)
{
    if (fabs(expected - actual) > 0.001)
    {
        assert_fails += 1;
        printf("scalar comparison %s\"%s\"%s failed\n", BOLD, name, RESET);
        printf("expected: %f\n", expected);
        printf("  actual: %f\n\n", actual);
    }
}

// TESTS

/* automatically generated by 'generate_test.py' */
#include "test_backprop.c"

// CLI

void run_test(char *name, void test())
{
    int is_selected = n_tests == 0;
    for (int i = 0; i < n_tests; i++)
    {
        is_selected = is_selected || !strcmp(name, test_names[i]);
    }

    if (!is_selected)
        return;

    printf("%sRunning \"%s\"%s\n", BOLD, name, RESET);
    assert_fails = 0;
    test();
    if (assert_fails == 0)
    {
        n_passed += 1;
        printf("\U00002705 %spassed%s\n", GREEN, RESET);
    }
    else
    {
        n_failed += 1;
        printf("\U0000274c %sfailed%s\n", RED, RESET);
    }
    printf("\n");
}

double timestamp()
{
    struct timespec start;
    clock_gettime(CLOCK_REALTIME, &start);
    return start.tv_sec + start.tv_nsec / 1000000000;
}

// USAGE

void print_usage()
{
    printf("Usage:\n\n");
    printf("    %sneural test [<names>]%s\n\n", BOLD, RESET);
    printf("Run specified tests, otherwise run all.\n\n");
}

int main(int argc, char *argv[])
{
    if (argc > 1 && strcmp(argv[1], "--help") == 0)
    {
        print_usage();
        return 0;
    }

    srand(0);

    test_names = argv + 1;
    n_tests = argc - 1;

    double start = timestamp();

    run_test("test_back_propagation", test_back_propagation);

    double end = timestamp();

    printf("Ran %s%d%s tests (%s%d%s passed / %s%d%s failed) in %.2f s\n",
           BOLD, n_passed + n_failed, RESET,
           GREEN, n_passed, RESET,
           RED, n_failed, RESET,
           end - start);

    return n_failed != 0;
}
