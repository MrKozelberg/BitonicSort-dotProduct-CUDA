#define blockSizeBubbleSort 1024
#define blockSizeDotProd 1024
#define isPrint false 

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/** Swaps elements idx and idx+1 if the element idx+1 is greater than the
 * 	element idx, if idx is lower than the current step by 2 and if idx is
 *  not n-1
 *  @param deviceArray Sorted array
 *	@param n Size of the sorted array
 * 	@param step Current step
 */ 
__global__ void bubbleSortKernel(int *deviceArray, int n, int step){
	// Thread linear id
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int temp; // Temporary variable
	if (idx<(n-1)) {
    if ((step-2)>=idx){
      if (deviceArray[idx]<deviceArray[idx+1]){
        temp = deviceArray[idx];
        deviceArray[idx]=deviceArray[idx+1];
        deviceArray[idx+1] = temp;
      }
    }
  }
}

/** Sorts hostArray with the help of the Bubble sorting (CUDA is used)
 *	@param hostArray Sorted array
 *	@param n Size of the sorted array
 */
void bubbleSortCUDA(int *hostArray, int n){
	int *deviceArray; // declaration of a device copy of the sorted array
	cudaMalloc(&deviceArray, n * sizeof(int)); // Memory allocation
	// Copy host vector to device memory
	cudaMemcpy(deviceArray, hostArray, n*sizeof(int), cudaMemcpyHostToDevice);
	int gridSize = n / blockSizeBubbleSort + 1; // Size of a CUDA-grid
	// Bubble sort loop
	for (int step = 0; step <= n+n; step++){
		bubbleSortKernel<<<gridSize, blockSizeBubbleSort>>>(deviceArray, n, step);
		cudaThreadSynchronize();
	}
	// Copy back to host
	cudaMemcpy(hostArray, deviceArray, n*sizeof(int), cudaMemcpyDeviceToHost);
	// Release memory
	cudaFree(deviceArray);
}

/** Sorts hostArray with the help of the Bubble sorting (CUDA is not used)
 *	@param hostArray Sorted array
 *	@param n Size of the sorted array
 */
void bubbleSortCPU(int *hostArray, int n){
	int temp; // Temporary variable
	for (int i = 0; i < n; i++){
    for (int j = 0; j < n-i-1; j++) {
      if (hostArray[j]<hostArray[j+1]){
        temp = hostArray[j];
        hostArray[j] = hostArray[j+1];
        hostArray[j+1] = temp;
      }
    }
  }
}

/** Computes intermediate result of dot product
 *	@param blockSize Size of a CUDA-block
 *	@param deviceA First vector
 *	@param deviceB Second vector
 *	@param deviceC Intermediate result
 *	@param n Size of the first and the second vectors
 */
template<int blockSize>
__global__ void dotProdKernel(int const *deviceA, int const *deviceB, int *deviceC, int n){
	// shared array for cache (this is a shared array for the whole threads in the current block)
	__shared__ int cache[blockSize];
	// Get our thread linear ID
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int sum = 0;
  // Computation of sum of all elements that lie on this thread
  while (idx < n) {
  	sum += deviceA[idx] * deviceB[idx];
  	idx += blockDim.x * gridDim.x;
  }
  cache[threadIdx.x] = sum; // Filling cache
  __syncthreads(); // Barier synchronisation
  // Summation the cache (on the current block)
  int i = blockDim.x / 2;
  while (i != 0){
 		if (threadIdx.x < i) cache[threadIdx.x] += cache[threadIdx.x + i];
 		__syncthreads(); // Barier synchronisation
 		i /= 2;
 	}
 	// Sum of the cache on the current block is at cache[0] 
 	// Saving this result for each blocks
 	if (threadIdx.x == 0) deviceC[blockIdx.x] = cache[0];
}

/** Computes dot product (CUDA is used)
 *	@param hostA First array
 *	@param hostB Second array
 *	@param n Size of arrays
 */
int dotProdCUDA(int const *hostA, int const *hostB, int n){
	int *hostC; // declaration of vectors
	int *deviceA, *deviceB, *deviceC;
	// Number of thread blocks per grid
  int gridDimDotProd = (n + blockSizeDotProd - 1) / blockSizeDotProd;
  // Size in bytes of the vector C
  size_t bytesC = gridDimDotProd * sizeof(int);
	// Vectors allocation
	hostC = (int*)malloc(bytesC);
	cudaMalloc(&deviceA, n * sizeof(int));
  cudaMalloc(&deviceB, n * sizeof(int));
	cudaMalloc(&deviceC, bytesC);	
	// Copy host vectors to device
  cudaMemcpy(deviceA, hostA, n * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, n * sizeof(int), cudaMemcpyHostToDevice);
  // Execute the kernel
  dotProdKernel<blockSizeDotProd><<<gridDimDotProd, blockSizeDotProd>>>(deviceA, deviceB, deviceC, n);
  // Copy array back to host
  cudaMemcpy(hostC, deviceC, bytesC, cudaMemcpyDeviceToHost);
  // Release device memory
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);
  // Finish up on the host
  int res = 0;
  for (int i = 0; i < gridDimDotProd; i++){
  	res += hostC[i];
  }
  return res;
}

/** Computes dot product (CUDA is not used)
 *	@param hostA First array
 *	@param hostB Second array
 *	@param n Size of arrays
 */
int dotProdCPU(int const *hostA, int const *hostB, int n){
	int res = 0;
	for (int i = 0; i < n; i++) {
		res += hostA[i] * hostB[i];
	}
	return res;
}

/** Prints array
 *	@param a Array
 *	@param n Array length
 */
void printArray(int *a, int n){
	printf("[%d", a[0]);
	for (int i = 1; i < n; i++) {
		printf(",%d", a[i]);
	}
	printf("]\n");
}

void taskCPU(int n){
	// Arrays declaration
	int *hostA, *hostB;
	clock_t t1, t2;
	// CPU variant
	if (isPrint) printf("CUDA is not used\n");
	t1 = clock();
	// Arrays allocation
	hostA = (int*)malloc(n*sizeof(int));
	hostB = (int*)malloc(n*sizeof(int));
	// Arrays initialization
	for (int i=0; i<n; i++){
		hostA[i] = rand() % 10;
		hostB[i] = rand() % 10;
	}
	// Print array
	if (isPrint) printf("A\t");
	if (isPrint) printArray(hostA, n);
	if (isPrint) printf("B\t");
	if (isPrint) printArray(hostB, n);
	// Sorting
	bubbleSortCPU(hostA, n);
	bubbleSortCPU(hostB, n);
	// Print arrays
	if (isPrint) printf("sorted A\t");
	if (isPrint) printArray(hostA, n);
	if (isPrint) printf("sorted B\t");
	if (isPrint) printArray(hostB, n);
	// Dot product
	int dot_prod = dotProdCPU(hostA, hostB, n);
	if (isPrint) printf("dot_prods\t%d\n", dot_prod);
	t2 = clock();
	if (isPrint) {
		printf("time\t%f\n", (float)(t2-t1)/CLOCKS_PER_SEC);
	} else {
		printf("%i %f\n", n, (float)(t2-t1)/CLOCKS_PER_SEC);
	}
	// Release memory
	free(hostA);
	free(hostB);
}

void taskCUDA(int n){
	// Arrays declaration
	int *hostA, *hostB;
	clock_t t1, t2;
	// CPU variant
	if (isPrint) printf("CUDA is not used\n");
	t1 = clock();
	// Arrays allocation
	hostA = (int*)malloc(n*sizeof(int));
	hostB = (int*)malloc(n*sizeof(int));
	// Arrays initialization
	for (int i=0; i<n; i++){
		hostA[i] = rand() % 10;
		hostB[i] = rand() % 10;
	}
	// Print array
	if (isPrint) printf("A\t");
	if (isPrint) printArray(hostA, n);
	if (isPrint) printf("B\t");
	if (isPrint) printArray(hostB, n);
	// Sorting
	bubbleSortCUDA(hostA, n);
	bubbleSortCUDA(hostB, n);
	// Print arrays
	if (isPrint) printf("sorted A\t");
	if (isPrint) printArray(hostA, n);
	if (isPrint) printf("sorted B\t");
	if (isPrint) printArray(hostB, n);
	// Dot product
	int dot_prod = dotProdCUDA(hostA, hostB, n);
	if (isPrint) printf("dot_prods\t%d\n", dot_prod);
	t2 = clock();
	if (isPrint) {
		printf("time\t%f\n", (float)(t2-t1)/CLOCKS_PER_SEC);
	} else {
		printf("%i %f\n", n, (float)(t2-t1)/CLOCKS_PER_SEC);
	}
	// Release memory
	free(hostA);
	free(hostB);
}

int main(){
	int n; // Arrays size
	scanf("%d", &n);
	taskCPU(n);
	taskCUDA(n);
	return 0;	
}



























