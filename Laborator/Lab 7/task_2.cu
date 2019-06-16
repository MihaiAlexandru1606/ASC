#include <stdio.h>
#include <math.h>
#include "utils/utils.h"

// TODO 6: Write the code to add the two arrays element by element and 
// store the result in another array
__global__ void add_arrays(const float *a, const float *b, float *c, int N) {
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

  	// Avoid accessing out of bounds elements
  	if (i < N) {
    		c[i] = a[i] + b[i];
  	}    
}

int main(void) {
    	cudaSetDevice(0);
    	int N = 1 << 20;
	const int num_elements = N;
  	const int num_bytes = num_elements * sizeof(float);
    	float *host_array_a = 0;
    	float *host_array_b = 0;
    	float *host_array_c = 0;

    float *device_array_a = 0;
    float *device_array_b = 0;
    float *device_array_c = 0;

    // TODO 1: Allocate the host's arrays
		

    // TODO 2: Allocate the device's arrays

    // TODO 3: Check for allocation errors

	// Allocating the host array
  	host_array_a = (float *) malloc(num_bytes);
	host_array_b = (float *) malloc(num_bytes);
	host_array_c = (float *) malloc(num_bytes);

	
  	// Allocating the device's array; notice that we use a special
  	// function named cudaMalloc that takes the reference of the
  	// pointer declared above and the number of bytes.
  	cudaMalloc((void **) &device_array_a, num_bytes);
	cudaMalloc((void **) &device_array_b, num_bytes);
	cudaMalloc((void **) &device_array_c, num_bytes);

  	// If any memory allocation failed, report an error message
  	if (host_array_a == 0 || host_array_b == 0 || host_array_c == 0 || device_array_a == 0 || device_array_b == 0 || device_array_c == 0) {
    		printf("[HOST] Couldn't allocate memory\n");
    		return 1;
  	}

    // TODO 4: Fill array with values; use fill_array_float to fill
    // host_array_a and fill_array_random to fill host_array_b. Each
    // function has the signature (float *a, int n), where n = number of elements.
	fill_array_float(host_array_a, N);
	fill_array_random(host_array_b, N);


    // TODO 5: Copy the host's arrays to device
	cudaMemcpy(device_array_a, host_array_a, num_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(device_array_b, host_array_b, num_bytes, cudaMemcpyHostToDevice);

    // TODO 6: Execute the kernel, calculating first the grid size
    // and the amount of threads in each block from the grid
    // Hint: For this execise the block_size can have any value lower than the
    //      API's maximum value (it's recommended to be close to the maximum
    //      value).
	const size_t block_size = 256;
  	size_t blocks_no = num_elements / block_size;

	if (num_elements % block_size) 
		++blocks_no;

	add_arrays<<<blocks_no, block_size>>>(device_array_a, device_array_b, device_array_c, N);
	
    // TODO 7: Copy back the results and then uncomment the checking function
	cudaDeviceSynchronize();	
cudaMemcpy(host_array_c, device_array_c, num_bytes, cudaMemcpyDeviceToHost);

    check_task_2(host_array_a, host_array_b, host_array_c, N);

    // TODO 8: Free the memory
   	free(host_array_a);
	free(host_array_b);
	free(host_array_c);
	cudaFree(device_array_a);
	cudaFree(device_array_b);
	cudaFree(device_array_c);

    return 0;
}

