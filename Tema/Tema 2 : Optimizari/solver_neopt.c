#include "utils.h"

/**
 * functia calculeaza C = (zerotr(TRANSPOSE(A) * B + TRANSPOSE(B) * A )) ^ 2
 * @param N numarul de linii/coloane
 * @param A matricea A
 * @param B matricea B
 * @return matricea C
 */
double *my_solver(int N, double *A, double *B) {
    double *E;
    double *C;
    double swap;
    int i, j, k;

    E = calloc(N * N, sizeof(double));
    if (E == NULL) {
        return NULL;
    }

    C = calloc(N * N, sizeof(double));
    if (C == NULL) {
        free(E);
        return NULL;
    }

    /**
     * se calculeza A = TRANSPOSE(A)
     * transpunearea matricei A
     */
    for (i = 0; i < N; i++) {
        for (j = i + 1; j < N; j++) {
            swap = A[i * N + j];
            A[i * N + j] = A[j * N + i];
            A[j * N + i] = swap;
        }
    }

    /**
     * se calculeaza E = zerotr(TRANSPOSE(A) * B + TRANSPOSE(B) * A)
     */
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            for (k = 0; k < N; k++) {
                if (i == j) {
                    E[i * N + j] += 2 * A[i * N + k] * B[k * N + j];
                    continue;
                }

                if (i > j) {
                    E[j * N + i] += A[i * N + k] * B[k * N + j];
                    continue;
                }

                E[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }

    /**
     * se calculeaza C = E * E
     */
    for (i = 0; i < N; i++) {
        for (j = i; j < N; j++) {
            for (k = i; k <= j; k++) {
                C[i * N + j] += E[i * N + k] * E[k * N + j];
            }
        }
    }

    free(E);

    return C;
}

