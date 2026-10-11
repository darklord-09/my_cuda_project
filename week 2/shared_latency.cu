#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define S_MEM 1024

__global__ void shared_threads_opearations(unsigned int * seed, long long * cycles, int iterations){
   __shared__ unsigned int s_data[S_MEM];
   for(int i=threadIdx.x;i<S_MEM;i+=blockDim.x){
      s_data[i]=seed[i];
   }
   __syncthreads();

   if(threadIdx.x==0){
      unsigned int idx=0;
      long long start=clock64();
      for(int i=0;i<iterations;i++){
         idx=s_data[idx%S_MEM];
      }
      long long end=clock64();
      *cycles=end-start;
      seed[0]=idx;
   }
}

int main(){

    unsigned int *h_seed = (unsigned int *) malloc(S_MEM*sizeof(unsigned int));
  unsigned int *d_seed;
  long long *h_cycles = (long long *) malloc(sizeof(long long));
  long long *d_cycles;

  cudaMalloc(&d_seed,sizeof(unsigned int));
  cudaMalloc(&d_cycles,sizeof(long long));

  srand(7);

  for(int i=S_MEM-1;i>0;i--){
    int j=rand()%(i+1);
    int temp=h_seed[i];
    h_seed[i]=h_seed[j];
    h_seed[j]=temp;}

    cudaMemcpy(d_seed,h_seed,sizeof(unsigned int),cudaMemcpyHostToDevice);

  shared_threads_opearations<<<1,256>>>(d_seed,d_cycles,S_MEM);

  cudaDeviceSynchronize();

  cudaMemcpy(h_cycles,d_cycles,sizeof(long long),cudaMemcpyDeviceToHost);



  printf("Shared memory: %.2f cycles/access\n", (double)*h_cycles / S_MEM);

  cudaFree(d_seed);
  cudaFree(d_cycles);
  free(h_seed);
  free(h_cycles);


return 0;
}
