#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

using namespace std;

// VectorAdd parallel function
__global__ void vectorAdd(int *a, int *b, int *result, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n)
    {
        result[tid] = a[tid] + b[tid];
    }
}

int main()
{
    int *a, *b, *c;
    int *a_dev, *b_dev, *c_dev;
    int n = 1 << 24; // Total number of elements

    a = new int[n];
    b = new int[n];
    c = new int[n];

    int *d = new int[n]; // For serial addition
    int size = n * sizeof(int);

    // Allocate memory on device
    cudaMalloc(&a_dev, size);
    cudaMalloc(&b_dev, size);
    cudaMalloc(&c_dev, size);

    // Array initialization
    for (int i = 0; i < n; i++)
    {
        a[i] = 1;
        b[i] = 2;
        d[i] = a[i] + b[i]; // calculating serial addition
    }

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // Copy data from host to device
    cudaMemcpy(a_dev, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_dev, b, size, cudaMemcpyHostToDevice);

    int threads = 1024;
    int blocks = (n + threads - 1) / threads;

    cudaEventRecord(start);

    // Parallel addition kernel invocation
    vectorAdd<<<blocks, threads>>>(a_dev, b_dev, c_dev, n);

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float time = 0.0;
    cudaEventElapsedTime(&time, start, end);

    // Copy result back to host
    cudaMemcpy(c, c_dev, size, cudaMemcpyDeviceToHost);

    // Calculate the error term.
    int error = 0;
    for (int i = 0; i < n; i++)
    {
        error += d[i] - c[i];
    }

    cout << "Error : " << error << endl;
    cout << "Time Elapsed:  " << time << " milliseconds" << endl;

    // Free memory
    delete[] a;
    delete[] b;
    delete[] c;
    delete[] d;
    cudaFree(a_dev);
    cudaFree(b_dev);
    cudaFree(c_dev);

    return 0;
}


//output

// Error : 0
// Time Elapsed:  2.34865 milliseconds
