#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <ctime>
#include <sstream>
#include <string>

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>


#include "gpu_hashtable.hpp"

/**
 *  functia numara cheile care sunt deja in hashtable
 * @param hashTable_keys cheile hashTable-uluo
 * @param hashTable_size
 * @param keys
 * @param numKeys
 * @param count rezultatul este intors prin efect lateral
 */
__global__ void update_count(int *hashTable_keys, int hashTable_size, int *keys,
							 int numKeys, int *count)
{
	TypeHashFunc func[NUM_HASH] = {hash_generic, hash_fnv_1, hash_cpp,
								   hash1, hash2, hash3, hash4, hash5, hash6,
								   hash7, hash8, hash1_fnv_1, hash9, hash10,
								   hash11, hash6_fnv_1, hash5_fnv_1,
								   hash4_fnv_1, hash3_fnv_1, hash2_fnv_1,
								   hash7_fnv_1};

	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx < numKeys) {
		int hash[NUM_HASH];

		for (int i = 0; i < NUM_HASH; i++) {
			hash[i] = func[i](keys[idx], hashTable_size);
		}

		for (int i = 0; i < NUM_HASH; i++) {
			if (hashTable_keys[hash[i]] == keys[idx]) {
				atomicAdd(count, 1);
				return;
			}
		}
	}
}

/**
 * functia insereaza/ face update pentru perechile : cheie-> valuare
 * algoritmul este destul de simplu : se genereaza toate hash-urile pentru o
 * chei, se vede daca se afla acolo respectiva cheie sau este un slot liber,
 * altef se incepe de 0 pana size se cauta un slot sau se faca actualizare
 * @param hashTable_keys
 * @param hashTable_values
 * @param hashTable_size
 * @param keys
 * @param values
 * @param numKeys
 * @param count este incrementat de fiecare data cand este aduagata o noua
 * 				perechie
 * @param reshape daca este 1 se face doar inserctie de noi perechi fara update
 */
__global__ void insert(int *hashTable_keys, int *hashTable_values,
					   int hashTable_size, int *keys, int *values, int numKeys,
					   int *count, int reshape)
{
	TypeHashFunc func[NUM_HASH] = {hash_generic, hash_fnv_1, hash_cpp,
								   hash1, hash2, hash3, hash4, hash5, hash6,
								   hash7, hash8, hash1_fnv_1, hash9, hash10,
								   hash11, hash6_fnv_1, hash5_fnv_1,
								   hash4_fnv_1, hash3_fnv_1, hash2_fnv_1,
								   hash7_fnv_1};

	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx < numKeys) {
		// se varifca ca perechia sa fie valida
		if (keys[idx] < 1 || values[idx] < 1)
			return;

		int hash[NUM_HASH];

		for (int i = 0; i < NUM_HASH; i++) {
			hash[i] = func[i](keys[idx], hashTable_size);
		}

		/**
		 * cand old este egal cu KEY_INVALID se face o noua insserare
		 * atfel se verfica daca chesia exista si innoieste valuarea
		 */

		for (int i = 0; i < NUM_HASH; i++) {
			int old = atomicCAS(hashTable_keys + hash[i],
								KEY_INVALID, keys[idx]);

			if (old == KEY_INVALID) {
				hashTable_values[hash[i]] = values[idx];
				atomicAdd(count, 1);
				return;
			}

			if (reshape)
				continue;

			old = atomicCAS(hashTable_keys + hash[i],
							keys[idx], keys[idx]);
			if (old == keys[idx]) {
				hashTable_values[hash[i]] = values[idx];
				return;
			}
		}


		for (int i = 0; i < hashTable_size; i++) {
			int old = atomicCAS(hashTable_keys + i,
								KEY_INVALID, keys[idx]);

			if (old == KEY_INVALID) {
				hashTable_values[i] = values[idx];
				atomicAdd(count, 1);
				return;
			}

			if (reshape)
				continue;

			old = atomicCAS(hashTable_keys + i,
							keys[idx], keys[idx]);
			if (old == keys[idx]) {
				hashTable_values[i] = values[idx];
				return;
			}
		}
	}
}

/**
 * se cauta mai intai unde ideca hash-urile, daca nu gasete se cauta de la
 * 0 pana la size -1, acet lucru se intampla din cauza algoritmului inserare
 * se compara cheia, daca sunt egale, este scrisa in values[idx]
 * @param hashTable_keys
 * @param hashTable_values
 * @param hashTable_size
 * @param keys
 * @param values
 * @param numKeys
 */

__global__ void get(int *hashTable_keys, int *hashTable_values,
					int hashTable_size, int *keys, int *values, int numKeys)
{
	TypeHashFunc func[NUM_HASH] = {hash_generic, hash_fnv_1, hash_cpp,
								   hash1, hash2, hash3, hash4, hash5, hash6,
								   hash7, hash8, hash1_fnv_1, hash9, hash10,
								   hash11, hash6_fnv_1, hash5_fnv_1,
								   hash4_fnv_1, hash3_fnv_1, hash2_fnv_1,
								   hash7_fnv_1};

	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx < numKeys) {
		int hash[NUM_HASH];

		for (int i = 0; i < NUM_HASH; i++) {
			hash[i] = func[i](keys[idx], hashTable_size);
		}

		for (int i = 0; i < NUM_HASH; i++) {
			if (hashTable_keys[hash[i]] == keys[idx]) {
				values[idx] = hashTable_values[hash[i]];
				return;
			}
		}

		for (int i = 0; i < hashTable_size; i++) {
			if (hashTable_keys[i] == keys[idx]) {
				values[idx] = hashTable_values[i];
				return;
			}
		}
	}
}

/* INIT HASH
 */
GpuHashTable::GpuHashTable(int size)
{
	//cudaSetDevice(1);

	int new_size = get_next_size(size);
	int *keys = nullptr;
	int *values = nullptr;
	int *numberBucket = nullptr;

	cudaMallocManaged(reinterpret_cast<void **>(&keys), new_size);
	cudaMallocManaged(reinterpret_cast<void **>(&values), new_size);
	cudaMallocManaged(reinterpret_cast<void **>(&numberBucket),
					  sizeof(int));
	if (keys == nullptr || values == nullptr || numberBucket == nullptr) {
		DIE(1, "malloc Managed");
	}

	cudaMemset(keys, 0, sizeof(int) * new_size);
	cudaMemset(values, 0, sizeof(int) * new_size);
	*numberBucket = 0;

	this->_values = values;
	this->_keys = keys;
	this->size = new_size;
	this->number_bucket = numberBucket;
}

/* DESTROY HASH
 */
GpuHashTable::~GpuHashTable()
{
	if (_keys)
		cudaFree(_keys);

	if (_values)
		cudaFree(_values);

	if (number_bucket)
		cudaFree(number_bucket);
}

/* RESHAPE HASH
 */
void GpuHashTable::reshape(int numBucketsReshape)
{
	int new_size = numBucketsReshape; //get_next_size(numBucketsReshape);
	int *keys = nullptr;
	int *values = nullptr;
	int *old_keys, *old_values;
	int numBlock, blockSize = 256;
	int *count = nullptr;

	cudaMallocManaged(reinterpret_cast<void **>(&count), sizeof(int));
	cudaMallocManaged(reinterpret_cast<void **>(&keys),
					  new_size * sizeof(int));
	cudaMallocManaged(reinterpret_cast<void **>(&values),
					  new_size * sizeof(int));

	if (keys == nullptr || values == nullptr || count == nullptr) {
		DIE(1, "malloc Managed");
	}

	cudaMemset(keys, 0, sizeof(int) * new_size);
	cudaMemset(values, 0, sizeof(int) * new_size);
	*count = 0;

	/**
	 * daca sunt elemente de muata
	 */
	if (*(this->number_bucket) != 0) {
		numBlock = this->size / blockSize;
		if (this->size % blockSize != 0)
			numBlock++;

		insert << < numBlock, blockSize >> > (keys, values, new_size,
			this->_keys, this->_values, size, count, 1);

		cudaDeviceSynchronize();
	}


	old_keys = this->_keys;
	old_values = this->_values;

	this->_values = values;
	this->_keys = keys;
	this->size = new_size;

	cudaFree(old_keys);
	cudaFree(old_values);
	cudaFree(count);
}

/* INSERT BATCH
 */
bool GpuHashTable::insertBatch(int *keys, int *values, int numKeys)
{
	int *device_keys = NULL;
	int *device_values = NULL;
	int *count = NULL;
	int numBlock, blockSize = 256;

	cudaMallocManaged(reinterpret_cast<void **>(&device_keys),
					  sizeof(int) * numKeys);
	cudaMallocManaged(reinterpret_cast<void **>(&device_values),
					  sizeof(int) * numKeys);
	cudaMallocManaged(reinterpret_cast<void **>(&count), sizeof(int));

	if (device_keys == NULL || device_values == NULL || count == NULL)
		DIE(1, "cudaMallocManaged");

	memcpy(device_keys, keys, sizeof(int) * numKeys);
	memcpy(device_values, values, sizeof(int) * numKeys);
	*count = 0;

	numBlock = numKeys / blockSize;
	if (numKeys % blockSize != 0)
		numBlock++;
	/** se calculeaza nr de update-uri, pentr a vedea daca mai este spatiu */
	update_count << < numBlock, blockSize >> > (_keys, size, device_keys,
		numKeys, count);
	cudaDeviceSynchronize();

	/** daca despaste 0.5 din size atunci se mareste capacitatea cu 1.5 */
	if (*this->number_bucket + numKeys - *count > 9 * this->size / 10) {
		reshape(195 * (*this->number_bucket + numKeys) / 100);
	}

	/** inserarea proizisa */
	insert << < numBlock, blockSize >> > (_keys, _values, size, device_keys,
		device_values, numKeys, number_bucket, 0);
	cudaDeviceSynchronize();

	cudaFree(device_keys);
	cudaFree(device_values);
	cudaFree(count);

	return false;
}

/* GET BATCH
 */
int *GpuHashTable::getBatch(int *keys, int numKeys)
{
	int *device_keys = NULL;
	int *device_values = NULL;
	int *host_value;
	int numBlock, blockSize = 256;

	host_value = static_cast<int *>(malloc(sizeof(int) * numKeys));

	cudaMallocManaged(reinterpret_cast<void **>(&device_keys),
					  sizeof(int) * numKeys);
	cudaMallocManaged(reinterpret_cast<void **>(&device_values),
					  sizeof(int) * numKeys);


	if (device_keys == NULL || device_values == NULL) {
		DIE(1, "cudaMalloc");
	}

	memcpy(device_keys, keys, sizeof(int) * numKeys);

	numBlock = numKeys / blockSize;
	if (numKeys % blockSize != 0)
		numBlock++;

	get << < numBlock, blockSize >> > (_keys, _values, size,
		device_keys, device_values, numKeys);
	cudaDeviceSynchronize();

	memcpy(host_value, device_values, sizeof(int) * numKeys);

	cudaFree(device_keys);
	cudaFree(device_values);

	return host_value;
}

/* GET LOAD FACTOR
 * num elements / hash total slots elements
 */
float GpuHashTable::loadFactor()
{
	return 1.0f * *this->number_bucket / this->size;
}

/*********************************************************/

#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
