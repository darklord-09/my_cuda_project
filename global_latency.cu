#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define ARRAY_SIZE (1<<24)

__global__ void global_cycle_counter(unsigned int * data, unsigned int * sink, long long * cycles, long long int iterations){
   unsigned int idx=0;
   long long start=clock64();
   for(int i=0;i<iterations;i++){
    idx=data[idx];
   }
   long long end=clock64();
   *cycles=end-start;
   *sink=idx;
}

int main(){
  unsigned int *h_data = (unsigned int *) malloc(ARRAY_SIZE*sizeof(unsigned int));
  unsigned int *h_sink= (unsigned int *) malloc(sizeof(unsigned int));
  unsigned int *d_data;
  unsigned int *d_sink;
  long long *h_cycles = (long long *) malloc(sizeof(long long));
  long long *d_cycles;

  for(int i=0;i<ARRAY_SIZE;i++){
    h_data[i]=i;
  }

  srand(42);

  for(int i=ARRAY_SIZE-1;i>0;i--){
    int j=rand()%(i+1);
    int temp=h_data[i];
    h_data[i]=h_data[j];
    h_data[j]=temp;
  }

  cudaMalloc(&d_data,ARRAY_SIZE*sizeof(unsigned int));
  cudaMalloc(&d_sink,sizeof(unsigned int));
  cudaMalloc(&d_cycles,sizeof(long long));

  cudaMemcpy(d_data,h_data,ARRAY_SIZE*sizeof(unsigned int),cudaMemcpyHostToDevice);

  global_cycle_counter<<<1,1>>>(d_data,d_sink,d_cycles,ARRAY_SIZE);
  cudaDeviceSynchronize();

  cudaMemcpy(h_cycles,d_cycles,sizeof(long long),cudaMemcpyDeviceToHost);



  printf("Global memory: %.2f cycles/access\n", (double)*h_cycles / ARRAY_SIZE);

  cudaFree(d_data);
  cudaFree(d_sink);
  cudaFree(d_cycles);
  free(h_data);
  free(h_sink);
  free(h_cycles);


return 0;

}
