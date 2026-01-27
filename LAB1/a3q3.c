#include <stdio.h>
#include <omp.h>

static long num_steps = 100000;
double step;

int main() {
    int i;
    double x, pi, sum;
    step = 1.0 / (double)num_steps;
    printf("Number of steps: %ld\n\n", num_steps);
    printf("Parallel Computation with varying thread counts:\n");
    double base_time = 0;
    for (int threads = 2; threads <= 32; threads *= 2) {
        omp_set_num_threads(threads);
        sum = 0.0;
        double start = omp_get_wtime();
        #pragma omp parallel for private(x) reduction(+:sum)
        for (i = 0; i < num_steps; i++) {
            x = (i + 0.5) * step;
            sum += 4.0 / (1.0 + x * x);
        }
        pi = step * sum;
        double end = omp_get_wtime();
        double time = end - start;
        if (threads == 2)
            base_time = time;
        printf("Threads:%d \t", threads);
        printf("Time:%f \t", time);
        printf("π = %.15f \t", pi);
        printf("speedup:%f\n", base_time / time);
    }
    
    return 0;
}