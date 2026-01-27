#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 1000  

int main() {
    int i, j, k;
    int **A = (int **)malloc(N * sizeof(int *));
    int **B = (int **)malloc(N * sizeof(int *));
    int **C = (int **)malloc(N * sizeof(int *));
    
    for(i = 0; i < N; i++) {
        A[i] = (int *)malloc(N * sizeof(int));
        B[i] = (int *)malloc(N * sizeof(int));
        C[i] = (int *)malloc(N * sizeof(int));
    }

    
    for(i = 0; i < N; i++)
        for(j = 0; j < N; j++) {
            A[i][j] = 1;   
            B[i][j] = 2;   
            C[i][j] = 0;
        }

    printf("1D Threading (Parallelizing outer loop only)\n");
    double base_time_1d = 0;
    for (int threads = 2; threads <= 32; threads *= 2) {
        
        for(i = 0; i < N; i++)
            for(j = 0; j < N; j++)
                C[i][j] = 0;
        
        omp_set_num_threads(threads);
        double start = omp_get_wtime();
        
        #pragma omp parallel for private(j,k)
        for(i = 0; i < N; i++) {
            for(j = 0; j < N; j++) {
                int sum = 0;
                for(k = 0; k < N; k++)
                    sum += A[i][k] * B[k][j];
                C[i][j] = sum;
            }
        }
        
        double end = omp_get_wtime();
        double time = end - start;
        
        if (threads == 2)
            base_time_1d = time;
        
        printf("Threads:%d \t", threads);
        printf("Time:%f \t", time);
        printf("base_time:%f \t", base_time_1d);
        printf("speedup:%f\n", base_time_1d / time);
    }

    printf("\n2D Threading (Parallelizing both i and j loops with collapse)\n");
    double base_time_2d = 0;
    for (int threads = 2; threads <= 32; threads *= 2) {
        
        for(i = 0; i < N; i++)
            for(j = 0; j < N; j++)
                C[i][j] = 0;
        
        omp_set_num_threads(threads);
        double start = omp_get_wtime();
        
        #pragma omp parallel for collapse(2) private(k)
        for(i = 0; i < N; i++) {
            for(j = 0; j < N; j++) {
                int sum = 0;
                for(k = 0; k < N; k++)
                    sum += A[i][k] * B[k][j];
                C[i][j] = sum;
            }
        }
        
        double end = omp_get_wtime();
        double time = end - start;
        
        if (threads == 2)
            base_time_2d = time;
        
        printf("Threads:%d \t", threads);
        printf("Time:%f \t", time);
        printf("base_time:%f \t", base_time_2d);
        printf("speedup:%f\n", base_time_2d / time);
    }

    printf("\nDirect Comparison: 1D vs 2D Threading \n");
    
    
    for(i = 0; i < N; i++)
        for(j = 0; j < N; j++)
            C[i][j] = 0;
    
    
    double start_1d = omp_get_wtime();
    #pragma omp parallel for private(j,k)
    for(i = 0; i < N; i++) {
        for(j = 0; j < N; j++) {
            int sum = 0;
            for(k = 0; k < N; k++)
                sum += A[i][k] * B[k][j];
            C[i][j] = sum;
        }
    }
    double end_1d = omp_get_wtime();
    double time_1d = end_1d - start_1d;
    printf("1D Threading Time: %f seconds\n", time_1d);
    
    
    for(i = 0; i < N; i++)
        for(j = 0; j < N; j++)
            C[i][j] = 0;
    
    
    double start_2d = omp_get_wtime();
    #pragma omp parallel for collapse(2) private(k)
    for(i = 0; i < N; i++) {
        for(j = 0; j < N; j++) {
            int sum = 0;
            for(k = 0; k < N; k++)
                sum += A[i][k] * B[k][j];
            C[i][j] = sum;
        }
    }
    double end_2d = omp_get_wtime();
    double time_2d = end_2d - start_2d;
    printf("2D Threading Time: %f seconds\n", time_2d);
    printf("Speedup (1D vs 2D): %f\n", time_1d / time_2d);

    
    for(i = 0; i < N; i++) {
        free(A[i]); 
        free(B[i]); 
        free(C[i]);
    }
    free(A); 
    free(B); 
    free(C);

    return 0;
}