#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N (1<<16)

int main() {
    double *X, *Y;
    double a = 2.5;

    X = (double*) malloc(N * sizeof(double));
    Y = (double*) malloc(N * sizeof(double));

    for (int i = 0; i < N; i++) {
        X[i] = 1.0;
        Y[i] = 2.0;
    }

    double base_time = 0.0;

    for (int threads = 2; threads <= 32; threads *= 2) {

        omp_set_num_threads(threads);

        double start = omp_get_wtime();

        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            X[i] = a * X[i] + Y[i];
        }

        double end = omp_get_wtime();
        double time = end - start;

        if (threads == 2)
            base_time = time;

        printf("Threads: %2d | Time: %f | Speedup: %f\n",
               threads, time, base_time / time);
    }

    free(X);
    free(Y);
    return 0;
}
