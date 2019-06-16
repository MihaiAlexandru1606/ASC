#include "utils.h"
#include "cblas.h"

#define ALFA        1.0l
#define BETA        0.0l


/**
 * functia calculeaza C = (zerotr(TRANSPOSE(A) * B + TRANSPOSE(B) * A )) ^ 2
 * @param N numarul de linii/coloane
 * @param A matricea A
 * @param B matricea B
 * @return matricea C
 */
double *my_solver(int N, double *A, double *B) {
    double *C;
    double *E;

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
     * se calculeaza E = zerotr(TRANSPOSE(A) * B + TRANSPOSE(B) * A)
     */
    cblas_dsyr2k(
            CblasRowMajor,
            CblasUpper,
            CblasTrans,
            N,
            N,
            ALFA,
            A,
            N,
            B,
            N,
            BETA,
            E,
            N
    );

    /**
     * se calculeaza C = E * E
     */
    cblas_dgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            N,
            N,
            N,
            ALFA,
            E,
            N,
            E,
            N,
            BETA,
            C,
            N
    );

    free(E);

    return C;
}
