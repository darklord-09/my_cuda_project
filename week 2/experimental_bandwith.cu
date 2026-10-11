#include <cstdio>
#include <cuda_runtime.h>

__global__ void simple_copy(const float * in, float * out, int n){
   int i=blockIdx.x*blockDim.x+threadIdx.x;
   if(i<n){
      out[i]=in[i];
   }
}

int main(){
  int n=1<<26;
  size_t bytes=n*sizeof(float);
  float * d_in;
  float * d_out;
  float * h_in = (float*)malloc(bytes);

  for(int i=0;i<n;i++){
    h_in[i]=1.0f;
  }


  cudaMalloc(&d_in,bytes);
  cudaMalloc(&d_out,bytes);

  cudaMemcpy(d_in,h_in,bytes,cudaMemcpyHostToDevice);

  int threads=1024;
  int blocks=(n+threads-1)/threads;

  //warm-up run
  simple_copy<<<blocks,threads>>>(d_in,d_out,n);
  cudaDeviceSynchronize();

  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  simple_copy<<<blocks,threads>>>(d_in,d_out,n);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float time;
  cudaEventElapsedTime(&time,start,stop);

  double GB_processed = (2 * n)/(1e9); //2 as one for read and one for write
  printf("measured bandwidth in GB/s %.4f",GB_processed/(time/1000));

  return 0;
}
