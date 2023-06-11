#include <cstdlib>
#include <cmath>
#include <ctime>
#include <iostream>
#include <random>
#include <cassert>

#define DO_PRINT false
#define DO_CHECK false
#define GRID_SIZE_SORT 4096
#define BLOCK_SIZE_SORT 1024
#define BLOCK_SIZE_DOTPROD 1024

size_t arrayLength = 8;

/** Fills array with random values with uniform distribution between 0 and 100
 * @param a Array of float-typed numbers
 * @param n Size of the array a
 */
void fillArray(float* a, size_t n) {
	std::random_device rd;  // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<float> dis(0, 1);
  for (size_t i = 0; i < n; i++) a[i] = dis(gen);
}

/** Prints array
 * @param a Array of numbers
 * @param n Size of the array a
 */
void printArray(float* a, size_t n) {
	std::cout << "[";
	for (size_t i = 0; i < n-1; i++) std::cout << a[i] << ", ";
	std::cout << a[n-1] << "]" << std::endl;
}

/** Check if array is sorted or not
 * @param a Array of float-typed numbers
 * @param n Size of the array a
 * returns true if array is sorted and false if not
 */
bool checkArraySorted(float* a, size_t n) {
	for (size_t i = 0; i < n - 1; i++) {
		if (a[i] > a[i+1]) {
			return false;
		}
	}
	return true;
}

/** Kernel for the bitonic sort step
 * @param a Sorted array
 * @param j Parameter of the minor loop of the bitonic sorting
 * @param k Parameter of the main loop of the bitonic sorting
 */
__global__ void bitonicSortStep(float* a, size_t j, size_t k) {
	// Thread linear id
	size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	// Current thread works only with idx-th and (idx xor j)-th elements
	size_t idxXj = idx ^ j;
	// Threads that have idx < idxXj do the step of the bitonic sorting
	if ((idxXj) > idx) {
		bool order = (idx & k);
		// sort in ascending order on threads with order == 0
		if (order == 0) {
			if (a[idx] > a[idxXj]) {
				// Swap two elements
				float temp = a[idx];
				a[idx] = a[idxXj];
				a[idxXj] = temp;
			}	
		}
		// sort in descending order on threads with order != 0
		if (order != 0) {
			if (a[idx] < a[idxXj]) {
				// Swap two elements
				float temp = a[idx];
				a[idx] = a[idxXj];
				a[idxXj] = temp;
			}
		} 
	}
}

/** Kernel for the bitonic sorting algorithm 
 * @param hostArray Sorted array
 */
void bitonicSort(float* hostArray) {
	// Allocate memory for the deviceArray
	float* deviceArray;
	cudaMalloc(&deviceArray, arrayLength * sizeof(float));
	// Copy host vector to a device memory
	cudaMemcpy(deviceArray, hostArray, arrayLength * sizeof(float), cudaMemcpyHostToDevice);
	// Main loop of the sorting algorithm
	size_t j, k;
	for (k = 2; k <= arrayLength; k <<= 1) {
		// Minor loop of the sorting algorithm
		for (j = k >> 1; j > 0; j >>= 1) {
			bitonicSortStep<<<GRID_SIZE_SORT, BLOCK_SIZE_SORT>>>(deviceArray, j, k);
		}
	}
	cudaMemcpy(hostArray, deviceArray, arrayLength * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(deviceArray);
}

/** Computes intermediate result of dot product
 * @param blockSize Size of a CUDA-block
 * @param deviceA First vector
 * @param deviceB Second vector
 * @param deviceC Intermediate result
 * @param n Size of the first and the second vectors
 */
template<size_t blockSize>
__global__ void dotProdKernel(float const *deviceA, float const *deviceB, float *deviceC, size_t n){
	// shared array for cache (this is a shared array for the whole threads in the current block)
	__shared__ float cache[blockSize];
	// Get our thread linear ID
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0;
  // Computation of sum of all elements that lie on this thread
  while (idx < n) {
  	sum += deviceA[idx] * deviceB[idx];
  	idx += blockDim.x * gridDim.x;
  }
  cache[threadIdx.x] = sum; // Filling cache
  __syncthreads(); // Barier synchronisation
  // Summation the cache (on the current block)
  size_t i = blockDim.x / 2;
  while (i != 0){
 		if (threadIdx.x < i) cache[threadIdx.x] += cache[threadIdx.x + i];
 		__syncthreads(); // Barier synchronisation
 		i /= 2;
 	}
 	// Sum of the cache on the current block is at cache[0] 
 	// Saving this result for each blocks
 	if (threadIdx.x == 0) deviceC[blockIdx.x] = cache[0];
}

/** Computes dot product
 * @param hostA First array
 * @param hostB Second array
 * @param n Size of arrays
 */
float dotProd(float const *hostA, float const *hostB, size_t n){
	float *hostC; // declaration of vectors
	float *deviceA, *deviceB, *deviceC;
	// Number of thread blocks per grid
  size_t gridDimDotProd = (n + BLOCK_SIZE_DOTPROD - 1) / BLOCK_SIZE_DOTPROD;
  // Size in bytes of the vector C
  size_t bytesC = gridDimDotProd * sizeof(float);
	// Vectors allocation
	hostC = (float*)malloc(bytesC);
	cudaMalloc(&deviceA, n * sizeof(float));
  cudaMalloc(&deviceB, n * sizeof(float));
	cudaMalloc(&deviceC, bytesC);	
	// Copy host vectors to device
  cudaMemcpy(deviceA, hostA, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, n * sizeof(float), cudaMemcpyHostToDevice);
  // Execute the kernel
  dotProdKernel<BLOCK_SIZE_DOTPROD><<<gridDimDotProd, BLOCK_SIZE_DOTPROD>>>(deviceA, deviceB, deviceC, n);
  // Copy array back to host
  cudaMemcpy(hostC, deviceC, bytesC, cudaMemcpyDeviceToHost);
  // Release device memory
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);
  // Finish up on the host
  float res = 0;
  for (size_t i = 0; i < gridDimDotProd; i++){
  	res += hostC[i];
  }
	// Release C
	free(hostC);
	// Result
  return res;
}

int main(int argc, char *argv[]) {
	if (argc > 1) {
		arrayLength = atoi(argv[1]); // Arrays length
	}
	// Time measurement starts here
	clock_t t1, t2;
	t1 = clock();
	// Allocate host arrays A and B
	float* hostA;
	float* hostB;
	hostA = (float*)malloc(arrayLength * sizeof(float));
	hostB = (float*)malloc(arrayLength * sizeof(float));
	// Fill them with values
	fillArray(hostA, arrayLength);
	fillArray(hostB, arrayLength);
	if (DO_PRINT) {
		std::cout << "A before sorting:" << std::endl;
		printArray(hostA, arrayLength);
		std::cout << "B before sorting:" << std::endl;
		printArray(hostB, arrayLength);	
	}
	// Sort arrays
	bitonicSort(hostA);
	bitonicSort(hostB);
	if (DO_CHECK) assert((void("A is not sorted"), checkArraySorted(hostA, arrayLength)));
	if (DO_CHECK) assert((void("B is not sorted"), checkArraySorted(hostB, arrayLength)));
	if (DO_PRINT) {
		std::cout << "A after sorting:" << std::endl;
		printArray(hostA, arrayLength);
		std::cout << "B after sorting:" << std::endl;
		printArray(hostB, arrayLength);	
	}
	// Compute dot product
	float dot = dotProd(hostA, hostB, arrayLength);
	if (DO_PRINT) std::cout << "Dot product: " << dot << std::endl;
	// Release memory
	free(hostA);
	free(hostB);
	// Time measurements stops here
	t2 = clock();
	std::cout << arrayLength << "," << (float)(t2-t1)/CLOCKS_PER_SEC << std::endl;
	return 0;
}