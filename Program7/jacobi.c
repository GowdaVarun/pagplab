
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define N 1000
#define ITER 100

int main() {
    int i, j, iter;
    double *A, *B;

    A = (double *)malloc(N * N * sizeof(double));
    B = (double *)malloc(N * N * sizeof(double));

    // Initialize matrices
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            A[i * N + j] = 0.0;
            B[i * N + j] = 0.0;
        }
    }

    // Boundary conditions
    for (i = 0; i < N; i++) {
        A[i * N] = 100.0;
        A[i * N + N - 1] = 100.0;
    }

    for (j = 0; j < N; j++) {
        A[j] = 100.0;
        A[(N - 1) * N + j] = 100.0;
    }

    // Jacobi iteration using OpenACC
    #pragma acc data copy(A[0:N*N], B[0:N*N])
    {
        for (iter = 0; iter < ITER; iter++) {

            #pragma acc parallel loop collapse(2)
            for (i = 1; i < N - 1; i++) {
                for (j = 1; j < N - 1; j++) {

                    B[i * N + j] =
                        0.25 * (
                            A[(i - 1) * N + j] +
                            A[(i + 1) * N + j] +
                            A[i * N + (j - 1)] +
                            A[i * N + (j + 1)]
                        );
                }
            }

            // Swap pointers
            double *temp = A;
            A = B;
            B = temp;
        }
    }

    printf("Jacobi iteration completed successfully!\n");
    printf("Iterations: %d\n", ITER);
    printf("Center value: %f\n", A[(N / 2) * N + (N / 2)]);

    free(A);
    free(B);

    return 0;
}