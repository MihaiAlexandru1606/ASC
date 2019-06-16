#include <iostream>

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#define NUM_ELEM    	128
#define BLOCK_SIZE	32

using namespace std;

/**
	fiecare thread aduaga data[idx] la result
*/
__global__ void addSum(int *data, int *result, int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (idx < N) {
		atomicAdd(result, data[idx]);	
	}
}

/**
	fiecare worker calculeaza result[idx] = sum(data[0] to data[idx]),
*/
__global__ void worker(int *data, int *result, int N)
{
	// TODO, compute sum and store in result
	int idx = threadIdx.x + blockIdx.x * blockDim.x;	
	
	if (idx >= N)
		return;	
	
	int numBlock = data[idx] / BLOCK_SIZE;
	if (data[idx] % BLOCK_SIZE)
		numBlock++;

	addSum <<< numBlock, BLOCK_SIZE >>>(data, result + idx, data[idx]); 
	cudaDeviceSynchronize();
}


// TODO
// master will launch threads to compute sum on first N elements
__global__ void master(int *data, int *rezult, int N)
{
	// TODO, schedule worker threads
	int numBlock = N / BLOCK_SIZE;
	if (N % BLOCK_SIZE)
		numBlock++;

	worker <<< numBlock, BLOCK_SIZE >>> (data, rezult, N);
	cudaDeviceSynchronize();
}

void generateData(int *data, int num)
{
	srand(time(0));

	for (int i = 0; i < num; i++) {
		data[i] = rand() % 8 + 2;
	}
}

void print(int *data, int num)
{
	for (int i = 0; i < num; i++) {
		cout << data[i] << " ";
	}
	cout << endl;
}

// TASK check
// each element result[i] should be sum of first data[i] elements of data[i]
bool checkResult(int *data, int num, int *result)
{

	for (int i = 0; i < num; i++) {

		int sum = 0;
		for (int j = 0; j < data[i]; j++) {
			sum += data[j];
		}

		if (result[i] != sum) {
			cout << "Error at " << i << ", requested sum of first "
				 << data[i] << " elem, got " << result[i] << endl;
			return false;
		}
	}

	return true;
}

int main(int argc, char *argv[])
{
	int *data = NULL;
	cudaMallocManaged(&data, NUM_ELEM * sizeof(int));

	int *result = NULL;
	cudaMallocManaged(&result, NUM_ELEM * sizeof(int));

	generateData(data, NUM_ELEM);

	// TODO schedule master threads and pass data/result/num
	master <<< 1, 1 >>> (data, result, NUM_ELEM);
	cudaDeviceSynchronize();

	print(data, NUM_ELEM);
	print(result, NUM_ELEM);

	if (checkResult(data, NUM_ELEM, result)) {
		cout << "Result OK" << endl;
	} else {
		cout << "Result ERR" << endl;
	}

	cudaFree(data);
	cudaFree(result);

	return 0;
}
