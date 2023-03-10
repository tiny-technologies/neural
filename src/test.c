#include <locale.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <wchar.h>

// CONSTANTS

const char *RESET = "\x1B[0m";
const char *BOLD = "\x1B[1m";
const char *RED = "\x1B[1;31m";
const char *GREEN = "\x1B[1;32m";

int n_passed = 0;
int n_failed = 0;

char **test_names;
int n_tests;

// UTILS

int assert_eq(int size, double *expected, double *actual)
{
    int equals = 1;
    for (int i = 0; i < size; i++)
    {
        equals &= expected[i] == actual[i];
    }

    if (!equals)
    {
        printf("comparison failed\n");
        printf("expected: ");
        print_vector(size, expected);
        printf("  actual: ");
        print_vector(size, actual);
    }
    return equals;
}

// TESTS

int test_add(void)
{
    double a[] = {1., 2., 3.};
    add(3, a, a, a);
    double expected[] = {2., 4., 6.};
    if (!assert_eq(3, expected, a))
        return 0;

    return 1;
}

int test_sub(void)
{
    double a[] = {1., 2., 3.};
    double b[] = {1., 1., 1.};
    sub(3, a, b, a);
    double expected[] = {0., 1., 2.};
    if (!assert_eq(3, expected, a))
        return 0;

    return 1;
}

int test_dot(void)
{
    double a[] = {1., 2., 3., 4.};
    double b[] = {1., 2.};
    double result[2];
    dot(2, 2, a, b, result);
    double expected[] = {5., 11.};
    if (!assert_eq(2, expected, result))
        return 0;

    return 1;
}

int test_scale(void)
{
    double a[] = {2., 4., 6.};
    scale(3, 0.5, a, a);
    double expected[] = {1., 2., 3.};
    if (!assert_eq(3, expected, a))
        return 0;

    return 1;
}

// CLI

void run_test(char *name, int test())
{
    int is_selected = n_tests == 0;
    for (int i = 0; i < n_tests; i++)
    {
        is_selected = is_selected || !strcmp(name, test_names[i]);
    }

    if (!is_selected)
        return;

    printf("%sRunning \"%s\"%s\n", BOLD, name, RESET);
    if (test())
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

void print_usage_test()
{
    printf("usage: neural test [<names>]\n\n");
    printf("Run specified test otherwise run all.\n\n");
}

double timestamp()
{
    struct timespec start;
    clock_gettime(CLOCK_REALTIME, &start);
    return start.tv_sec + start.tv_nsec / 1000000000;
}

int test(char *names[], int n)
{
    test_names = names;
    n_tests = n;

    double start = timestamp();

    run_test("test_add", test_add);
    run_test("test_sub", test_sub);
    run_test("test_scale", test_scale);
    run_test("test_dot", test_dot);

    double end = timestamp();

    printf("Ran %s%d%s tests (%s%d%s passed / %s%d%s failed) in %.2f s\n",
           BOLD, n_passed + n_failed, RESET,
           GREEN, n_passed, RESET,
           RED, n_failed, RESET,
           end - start);

    return n_failed != 0;
}
