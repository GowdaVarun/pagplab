/* ==============================================================
   PROGRAM 1A - 1a.c
   ============================================================== */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include <omp.h>

#define SEED 35791246

int main(int argc, char *argv[])
{
    int niter = 0;

    int thread_count[] = {1, 2, 4, 8};
    // int num_thread_configs =
    //     sizeof(thread_count) / sizeof(thread_count[0]);

    printf("Enter the no of iterations used to estimate pi: ");
    scanf("%d", &niter);

    printf("Input size execution Time - Number of Threads\n");
    printf("Threads\tExecution Time(s)\tPI Estimation\n");

    for (int t = 0; t < 4; t++) {
        int num_threads = thread_count[t];
        int count = 0;
        double pi;
        double st, end;

        omp_set_num_threads(num_threads);
        st = omp_get_wtime();

        #pragma omp parallel
        {
            int local_count = 0;
            // unsigned int seed = SEED + omp_get_thread_num();

        #pragma omp for
            for (int i = 0; i < niter; i++) {
                double x = (double)rand() / RAND_MAX;
                double y = (double)rand() / RAND_MAX;
                double z = x * x + y * y;

                if (z <= 1) {
                    local_count++;
                }
            }

    #pragma omp atomic
            count += local_count;
        }

        pi = (double)count / niter * 4;
        end = omp_get_wtime();

        printf("%d\t%f\t%g\n", num_threads, end - st, pi);
    }

    return 0;
}


/* ==============================================================
   PROGRAM 1B - 1b.c
   ============================================================== */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define SEED 3655942
#define ROOT 0

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double t;
    if (rank == ROOT) {
        t = MPI_Wtime();
    }

    int n = 10000;
    int count = 0, red_count;

    srand(SEED + rank);
    for (int i = 0; i < n / size; i++) {
        double x = (double)rand() / RAND_MAX;
        double y = (double)rand() / RAND_MAX;
        double z = x * x + y * y;
        if (z <= 1.0) count++;
    }

    MPI_Reduce(&count, &red_count, 1, MPI_INT, MPI_SUM, ROOT, MPI_COMM_WORLD);

    if (rank == ROOT) {
        t = MPI_Wtime() - t;
        printf("Time: %f seconds\n", t);

        double pi = (double)red_count / n * 4;
        printf("No. of threads: %d\n", size);
        printf("No. of trials: %d\n", n);
        printf("Estimated value of pi: %f\n", pi);
    }

    MPI_Finalize();
    return 0;
}


/* ==============================================================
   PROGRAM 2 - 2.c
   ============================================================== */
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <matrix_dimension> <num_threads>\n", argv[0]);
        return -1;
    }

    int DIM = atoi(argv[1]);
    int thread_count = atoi(argv[2]);

    if (DIM <= 0 || thread_count <= 0) {
        printf("Matrix dimension and number of threads must be positive integers.\n");
        return -1;
    }

    int *A = (int *)malloc(DIM * DIM * sizeof(int));
    int *B = (int *)malloc(DIM * DIM * sizeof(int));
    int *C = (int *)malloc(DIM * DIM * sizeof(int));

    if (A == NULL || B == NULL || C == NULL) {
        printf("Memory allocation failed!\n");
        return -1;
    }

    for (int i = 0; i < DIM * DIM; i++) {
        A[i] = rand() % 10;
        B[i] = rand() % 10;
        C[i] = 0;
    }

    double t = omp_get_wtime();

    #pragma omp parallel for num_threads(thread_count)
    for (int i = 0; i < DIM; i++) {
        for (int j = 0; j < DIM; j++) {
            for (int k = 0; k < DIM; k++) {
                C[DIM * i + k] += A[DIM * i + j] * B[DIM * j + k];
            }
        }
    }

    t = omp_get_wtime() - t;

    printf("Matrix Dimension: %d, Threads: %d, Time: %f seconds\n", DIM, thread_count, t);

    free(A);
    free(B);
    free(C);

    return 0;
}


/* ==============================================================
   PROGRAM 3 - 3.c
   ============================================================== */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

int strike(int* comp, int start, int step, int stop) {
    int i;
    for (i = start; i <= stop; i += step)
        comp[i] = 1;
    return i;
}

int unfriendlySieve(int n) {
    int m = (int)sqrt(n);
    int* comp = (int*)calloc(n + 1, sizeof(int));

    double t = omp_get_wtime();

    int count = 0;
    for (int i = 2; i <= m; i++) {
        if (!comp[i]) {
            count++;
            strike(comp, 2 * i, i, n);
        }
    }
    for (int i = m + 1; i <= n; i++) {
        if (!comp[i]) {
            count++;
        }
    }

    t = omp_get_wtime() - t;
    printf("Unfriendly Time: %f seconds\n", t);

    free(comp);
    return count;
}

int friendlySieve(int n) {
    int m = (int)sqrt(n);
    int* comp = (int*)calloc(n + 1, sizeof(int));

    int* striker = (int*)malloc(m * sizeof(int));
    int* factor = (int*)malloc(m * sizeof(int));
    int fac_cnt = 0;

    double t = omp_get_wtime();

    int count = 0;
    for (int i = 2; i <= m; i++) {
        if (!comp[i]) {
            count++;
            factor[fac_cnt] = i;
            striker[fac_cnt] = strike(comp, 2 * i, i, m);
            fac_cnt++;
        }
    }
    for (int left = m + 1; left <= n; left += m) {
        int right = (left + m - 1 < n) ? left + m - 1 : n;
        for (int k = 0; k < fac_cnt; k++) {
            striker[k] = strike(comp, striker[k], factor[k], right);
        }
        for (int i = left; i <= right; i++) {
            if (!comp[i]) {
                count++;
            }
        }
    }

    t = omp_get_wtime() - t;
    printf("Friendly Time: %f seconds\n", t);

    free(striker);
    free(factor);
    free(comp);
    return count;
}

int parallelySieve(int n) {
    int m = (int)sqrt(n);
    int* comp = (int*)calloc(n + 1, sizeof(int));

    int* striker = (int*)malloc(m * sizeof(int));
    int* factor = (int*)malloc(m * sizeof(int));
    int fac_cnt = 0;

    double t = omp_get_wtime();

    int count = 0;
    for (int i = 2; i <= m; i++) {
        if (!comp[i]) {
            count++;
            factor[fac_cnt] = i;
            striker[fac_cnt] = strike(comp, 2 * i, i, m);
            fac_cnt++;
        }
    }
    for (int left = m + 1; left <= n; left += m) {
        int right = (left + m - 1 < n) ? left + m - 1 : n;

        #pragma omp parallel for
        for (int k = 0; k < fac_cnt; k++) {
            striker[k] = strike(comp, striker[k], factor[k], right);
        }

        #pragma omp parallel for reduction(+:count)
        for (int i = left; i <= right; i++) {
            if (!comp[i]) {
                count++;
            }
        }
    }

    t = omp_get_wtime() - t;
    printf("Parallely Time: %f seconds\n", t);

    free(striker);
    free(factor);
    free(comp);
    return count;
}

int main() {
    int n = 100000000;

    printf("Unfriendly Count: %d\n", unfriendlySieve(n));
    printf("Friendly Count: %d\n", friendlySieve(n));
    printf("Parallely Count: %d\n", parallelySieve(n));

    return 0;
}


/* ==============================================================
   PROGRAM 4 - 4.c
   ============================================================== */
#include <stdio.h>
#include <stdlib.h>
#include <gd.h>
#include <omp.h>

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <in.png> <out.png>\n", argv[0]);
        exit(1);
    }

    FILE *ifp = fopen(argv[1], "rb");
    if (!ifp) {
        fprintf(stderr, "Error: Cannot open input file.\n");
        exit(1);
    }

    FILE *ofp = fopen(argv[2], "wb");
    if (!ofp) {
        fprintf(stderr, "Error: Cannot open output file.\n");
        fclose(ifp);
        exit(1);
    }

    double t = omp_get_wtime();

    gdImagePtr img = gdImageCreateFromPng(ifp);
    if (!img) {
        fprintf(stderr, "Error: Cannot create image from input.\n");
        fclose(ifp);
        fclose(ofp);
        exit(1);
    }

    int w = gdImageSX(img);
    int h = gdImageSY(img);

    printf("Image size: %d x %d pixels\n", w, h);

    gdImagePtr outImg = gdImageCreateTrueColor(w, h);

    #pragma omp parallel for schedule(static, 100) collapse(2)
    for (int x = 0; x < w; x++) {
        for (int y = 0; y < h; y++) {
            int color = gdImageGetPixel(img, x, y);
            int avgColor = (gdImageRed(img, color) + gdImageGreen(img, color) + gdImageBlue(img, color)) / 3;

            int threadId = omp_get_thread_num();
            int threadColor = (threadId * 10) % 256;

            gdImageSetPixel(outImg, x, y, gdImageColorAllocate(outImg, avgColor + threadColor, avgColor + threadColor, avgColor + threadColor));
        }
    }

    gdImagePng(outImg, ofp);
    gdImageDestroy(img);
    gdImageDestroy(outImg);

    fclose(ifp);
    fclose(ofp);

    t = omp_get_wtime() - t;
    printf("Time: %f seconds\n", t);

    return 0;
}


/* ==============================================================
   PROGRAM 5 - 5.c
   ============================================================== */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define SIZE 4
#define BUFSIZE 32

#define ROOT 0
#define TAG 0

char* m[SIZE] = { "", "Hello", "RVCE", "CSE" };
char msg[BUFSIZE];

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != SIZE) {
        printf("Provide argument: -np %d\n", SIZE);
        exit(1);
    }

    if (rank != ROOT) {
        strcpy(msg, m[rank]);
        MPI_Send(&msg, BUFSIZE, MPI_CHAR, ROOT, TAG, MPI_COMM_WORLD);
    } else {
        for(int i = 0; i < size; i++) {
            if (i == ROOT) continue;
            MPI_Recv(&msg, BUFSIZE, MPI_CHAR, i, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("Received %s in process %d from process %d\n", msg, rank, i);
        }
    }

    MPI_Finalize();
    return 0;
}


/* ==============================================================
   PROGRAM 6 - 6.c
   ============================================================== */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <omp.h>

#define SIZE 5
#define FILE_NAME "Program6/words.txt"

char *words[SIZE] = {"lorem", "ipsum", "dolo", "sit", "amet"};
int counts[SIZE] = {0};

void to_lower(char *str) {
    for (int i = 0; str[i]; i++) {
        str[i] = tolower(str[i]);
    }
}

int get_count(const char *file_name, const char *key) {
    char lower_key[100];
    strcpy(lower_key, key);
    to_lower(lower_key);

    FILE *file = fopen(file_name, "r");
    if (!file) {
        fprintf(stderr, "Error opening file: %s\n", file_name);
        return 0;
    }

    char word[100];
    int count = 0;
    while (fscanf(file, "%99s", word) == 1) {
        to_lower(word);
        if (strcmp(lower_key, word) == 0) {
            count++;
        }
    }
    fclose(file);
    return count;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <num_threads>\n", argv[0]);
        return 1;
    }

    int num_threads = atoi(argv[1]);
    if (num_threads < 1) {
        fprintf(stderr, "Error: Number of threads must be at least 1.\n");
        return 1;
    }

    omp_set_num_threads(num_threads);

    double t = omp_get_wtime();

    #pragma omp parallel for
    for (int i = 0; i < SIZE; i++) {
        counts[i] = get_count(FILE_NAME, words[i]);
    }

    t = omp_get_wtime() - t;
    printf("Time: %f seconds\n", t);

    for (int i = 0; i < SIZE; i++) {
        printf("%s %d\n", words[i], counts[i]);
    }

    return 0;
}


/* ==============================================================
   PROGRAM 7A - 7.c
   ============================================================== */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 512

void initialize_matrix(float *matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = (float)(rand() % 100) / 10.0;
    }
}

void print_matrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%6.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

int main() {
    srand(time(NULL));

    float *A = (float *)malloc(N * N * sizeof(float));
    float *B = (float *)malloc(N * N * sizeof(float));
    float *C = (float *)malloc(N * N * sizeof(float));

    if (A == NULL || B == NULL || C == NULL) {
        printf("Memory allocation failed!\n");
        return 1;
    }

    initialize_matrix(A, N * N);
    initialize_matrix(B, N * N);

    #pragma acc data copyin(A[0:N*N], B[0:N*N]) copyout(C[0:N*N])
    {
        #pragma acc parallel loop collapse(2)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < N; k++) {
                    sum += A[i * N + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    }

    if (N <= 8) {
        printf("Matrix A:\n");
        print_matrix(A, N, N);
        printf("Matrix B:\n");
        print_matrix(B, N, N);
        printf("Resultant Matrix C:\n");
        print_matrix(C, N, N);
    } else {
        printf("Matrix multiplication completed successfully!\n");
    }

    free(A);
    free(B);
    free(C);

    return 0;
}


/* ==============================================================
   PROGRAM 7B - jacobi.c
   ============================================================== */
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

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            A[i * N + j] = 0.0;
            B[i * N + j] = 0.0;
        }
    }

    for (i = 0; i < N; i++) {
        A[i * N] = 100.0;
        A[i * N + N - 1] = 100.0;
    }

    for (j = 0; j < N; j++) {
        A[j] = 100.0;
        A[(N - 1) * N + j] = 100.0;
    }

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
