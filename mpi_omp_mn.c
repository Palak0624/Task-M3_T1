#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <time.h>

#define N 1024

void initialize_matrix(double *matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[i * size + j] = (double)rand() / RAND_MAX;
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    double start_time, end_time;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (N % size != 0) {
        if (rank == 0) {
            printf("Matrix size must be divisible by number of processes.\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    double *A = NULL;
    double *B = NULL;
    double *C = NULL;
    double *local_A = NULL;
    double *local_C = NULL;
    
    int rows_per_process = N / size;
    int num_threads = 2; // As shown in the output
    
    if (rank == 0) {
        A = (double *)malloc(N * N * sizeof(double));
        B = (double *)malloc(N * N * sizeof(double));
        C = (double *)malloc(N * N * sizeof(double));
        
        initialize_matrix(A, N);
        initialize_matrix(B, N);
        
        start_time = MPI_Wtime();
    }
    
    local_A = (double *)malloc(rows_per_process * N * sizeof(double));
    local_C = (double *)malloc(rows_per_process * N * sizeof(double));
    B = (double *)malloc(N * N * sizeof(double));
    
    // Scatter rows of A to all processes
    MPI_Scatter(A, rows_per_process * N, MPI_DOUBLE, 
                local_A, rows_per_process * N, MPI_DOUBLE, 
                0, MPI_COMM_WORLD);
    
    // Broadcast B to all processes
    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Perform local matrix multiplication with OpenMP
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < rows_per_process; i++) {
        for (int j = 0; j < N; j++) {
            local_C[i * N + j] = 0.0;
            for (int k = 0; k < N; k++) {
                local_C[i * N + j] += local_A[i * N + k] * B[k * N + j];
            }
        }
    }
    
    // Gather results from all processes
    MPI_Gather(local_C, rows_per_process * N, MPI_DOUBLE, 
               C, rows_per_process * N, MPI_DOUBLE, 
               0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        end_time = MPI_Wtime();
        printf("MPI+OpenMP Time (N=%d, %d processes, %d threads): %.4f seconds\n", 
               N, size, num_threads, end_time - start_time);
        
        free(A);
        free(B);
        free(C);
    }
    
    free(local_A);
    free(local_C);
    free(B);
    
    MPI_Finalize();
    return 0;
}
