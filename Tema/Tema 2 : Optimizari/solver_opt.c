#include "utils.h"

double *my_solver(int N, double *A, double *B) {
    double *E = calloc(N * N, sizeof(double));
    double *C = calloc(N * N, sizeof(double));
    register double sum;
    double swap;

    double *origin_a;
    register double *pa;
    register double *pb;
    int in = -N; /** se retine i * N */

    int i;
    register int j;
    register int k;

    /**
     * se translateaza matricea A si B, A din fomula si B pentru a optimiza,
     */
    for (i = 0; i < N; i++) {
        for (j = i + 1; j < N; j++) {
            swap = A[i * N + j];
            A[i * N + j] = A[j * N + i];
            A[j * N + i] = swap;

            swap = B[i * N + j];
            B[i * N + j] = B[j * N + i];
            B[j * N + i] = swap;

        }
    }

    /**
     * se calculeaza E = zerotr(TRANSPOSE(A) * B + TRANSPOSE(B) * A)
     */
    origin_a = A - N; /** se retine adresa liniei curente pentru A */
    /** pentru fiecare  */
    for (i = 0; i < N; i++) {
        origin_a += N;
        in += N;
        pb = B; /** adresa de inceput a matricei */
        for (j = 0; j < N; j++) {
            pa = origin_a;

            sum = 0;
            for (k = 0; k < N; k++) {
                sum += *pa * *pb;
                pa++;
                pb++;
            }

            E[in + j] += sum;
            E[j * N + i] += sum;

        }
    }

    /**
     * se calculeaza C = E * E
     */
    origin_a = E - N - 1; /** se retine adresa primului termen din inmultire */
    in = -N;
    for (i = 0; i < N; i++) {
        origin_a += N + 1;
        in += N;

        for (j = i; j < N; j++) {
            pa = origin_a;
            pb = E + j * N + i;
            sum = 0;
            for (k = i; k <= j; k++) {
                sum += *pa * *pb;
                pa++;
                pb++;
            }

            C[in + j] = sum;
        }
    }

    free(E);

    return C;
}