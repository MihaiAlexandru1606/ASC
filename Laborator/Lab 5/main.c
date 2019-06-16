#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <sys/time.h>

#define N   16000

double *createMatrix(int size) {
    double *matrix = malloc(size * size * sizeof(double));

    assert(matrix != NULL);

    srand((unsigned int) time(NULL));
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[i * size + j] = rand() % 1000;
        }
    }

    return matrix;
}

double *mult_norm(double *a, double *b, int size, double *time) {
    double *c = calloc(size * size, sizeof(double));
    struct timeval begin, end;

    gettimeofday(&begin, NULL);

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            for (int k = 0; k < size; k++) {
                c[i * size + j] += a[i * size + k] * b[k * size + j];
            }
        }
    }

    gettimeofday(&end, NULL);

    *time = (end.tv_sec - begin.tv_sec) +
            ((end.tv_usec - begin.tv_usec) / 1000000.0);

    return c;
}

double *task1(double *a, double *b, int size, double *time) {
    double *c = malloc(size * size * sizeof(double));
    struct timeval begin, end;

    gettimeofday(&begin, NULL);

    for (int i = 0; i < size; i++) {
        double *orig_pa = a + (i * size);

        for (int j = 0; j < size; j++) {

            double *pa = orig_pa;
            double *pb = b + j;
            register double suma = 0;

            for (int k = 0; k < size; k++) {
                suma += *pa * *pb;
                pa++;
                pb += size;
            }
            c[i * size + j] = suma;
        }
    }

    gettimeofday(&end, NULL);

    *time = (end.tv_sec - begin.tv_sec) +
            ((end.tv_usec - begin.tv_usec) / 1000000.0);

    return c;
}

double *task2_i_k_j(double *a, double *b, int size, double *time) {
    double *c = calloc(size * size, sizeof(double));
    struct timeval begin, end;

    gettimeofday(&begin, NULL);

    for (int i = 0; i < size; i++) {
        for (int k = 0; k < size; k++) {
            for (int j = 0; j < size; j++) {
                c[i * size + j] += a[i * size + k] * b[k * size + j];
            }
        }
    }

    gettimeofday(&end, NULL);

    *time = (end.tv_sec - begin.tv_sec) +
            ((end.tv_usec - begin.tv_usec) / 1000000.0);

    return c;
}

double *task2_j_i_k(double *a, double *b, int size, double *time) {
    double *c = calloc(size * size, sizeof(double));
    struct timeval begin, end;

    gettimeofday(&begin, NULL);

    for (int j = 0; j < size; j++) {
        for (int i = 0; i < size; i++) {
            for (int k = 0; k < size; k++) {
                c[i * size + j] += a[i * size + k] * b[k * size + j];
            }
        }
    }

    gettimeofday(&end, NULL);

    *time = (end.tv_sec - begin.tv_sec) +
            ((end.tv_usec - begin.tv_usec) / 1000000.0);

    return c;
}

double *task2_j_k_i(double *a, double *b, int size, double *time) {
    double *c = calloc(size * size, sizeof(double));
    struct timeval begin, end;

    gettimeofday(&begin, NULL);

    for (int j = 0; j < size; j++) {
        for (int k = 0; k < size; k++) {
            for (int i = 0; i < size; i++) {
                c[i * size + j] += a[i * size + k] * b[k * size + j];
            }
        }
    }

    gettimeofday(&end, NULL);

    *time = (end.tv_sec - begin.tv_sec) +
            ((end.tv_usec - begin.tv_usec) / 1000000.0);

    return c;
}


double *task2_k_i_j(double *a, double *b, int size, double *time) {
    double *c = calloc(size * size, sizeof(double));
    struct timeval begin, end;

    gettimeofday(&begin, NULL);

    for (int k = 0; k < size; k++) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                c[i * size + j] += a[i * size + k] * b[k * size + j];
            }
        }
    }

    gettimeofday(&end, NULL);

    *time = (end.tv_sec - begin.tv_sec) +
            ((end.tv_usec - begin.tv_usec) / 1000000.0);

    return c;
}

double *task2_k_j_i(double *a, double *b, int size, double *time) {
    double *c = calloc(size * size, sizeof(double));
    struct timeval begin, end;

    gettimeofday(&begin, NULL);

    for (int k = 0; k < size; k++) {
        for (int j = 0; j < size; j++) {
            for (int i = 0; i < size; i++) {
                c[i * size + j] += a[i * size + k] * b[k * size + j];
            }
        }
    }

    gettimeofday(&end, NULL);

    *time = (end.tv_sec - begin.tv_sec) +
            ((end.tv_usec - begin.tv_usec) / 1000000.0);

    return c;
}

double *task3(double *a, double *b, int size, double *time) {
    double *c = calloc(size * size, sizeof(double));
    struct timeval begin, end;

    gettimeofday(&begin, NULL);
    int chache_line = 8;

    for (int i = 0; i < size; i += chache_line) {
        for (int j = 0; j < size; j += chache_line) {
            for (int k = 0; k < size; k += chache_line) {

                for (int m = i; m < chache_line + i && m < size; m++) {
                    for (int n = j; n < chache_line + j && n < size; n++) {
                        for (int p = k; p < chache_line + k && k < size; p++) {
                            c[m * size + n] +=
                                    a[m * size + p] *
                                    b[p * size + n];
                        }
                    }
                }


            }
        }
    }

    gettimeofday(&end, NULL);

    *time = (end.tv_sec - begin.tv_sec) +
            ((end.tv_usec - begin.tv_usec) / 1000000.0);

    return c;
}

double *bonus(int size, double *time) {
    double **a = malloc(size * sizeof(double *));
    assert(a != NULL);

    double **b = malloc(size * sizeof(double *));
    assert(b != NULL);

    double **c = malloc(size * sizeof(double *));
    assert(c != NULL);

    for (int i = 0; i < size; i++) {
        a[i] = malloc(size * sizeof(double));
        assert(a[i] != NULL);
        b[i] = malloc(size * sizeof(double));
        c[i] = calloc(size, sizeof(double));
    }

    struct timeval begin, end;

    gettimeofday(&begin, NULL);

    for (int k = 0; k < size; k++) {
        for (int j = 0; j < size; j++) {
            for (int i = 0; i < size; i++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    gettimeofday(&end, NULL);

    *time = (end.tv_sec - begin.tv_sec) +
            ((end.tv_usec - begin.tv_usec) / 1000000.0);

    return c;
}

int main() {
    double time;
    int size = 900;
    printf("Hello, World!\n");
    double *a = createMatrix(size);
    double *b = createMatrix(size);
    double *c = mult_norm(a, b, size, &time);
    printf("Time normal : %lf\n", time);
    double *d = task3(a, b, size, &time);
    printf("Time block : %lf\n", time);

    bonus(900, &time);
    printf("Time matrix : %lf\n", time);

    return 0;
}