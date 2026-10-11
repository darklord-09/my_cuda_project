#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__device__ float warpShuffle(float val){

  for(int offset=16;offset>0;offset/=2){
    val+=__shfl_down_sync(0xffffffff,val,offset);
  }
  return val;


}

__global__ void naive_parallel_reduction( float * out, const float * in ,int n){
extern __shared__ float sdata[];

int tid=threadIdx.x;
int i=blockIdx.x*blockDim.x+threadIdx.x;

if(i<n){
  sdata[tid]=in[i];
}

__syncthreads();


for(int s=1;s<blockDim.x;s*=2){
  if(tid% (2*s)==0){
    sdata[tid]+=sdata[tid+s];
  }
  __syncthreads();
}

if(tid==0){
  out[blockIdx.x]=sdata[0];
}

}


__global__ void sequential_addressing_optimisation(float * out, const float * in, int n){
  extern __shared__ float sdata[];

  int tid=threadIdx.x;
  int i=blockIdx.x*blockDim.x+threadIdx.x;

  if(i<n){
    sdata[tid]=in[i];
  }
  __syncthreads();

  for(int s=blockDim.x/2;s>0;s/=2){
     if(tid<s){
      sdata[tid]+=sdata[tid+s];
     }
    __syncthreads();
  }

  if(tid==0){
    out[blockIdx.x]=sdata[0];
  }
}

__global__ void warp_shuffle_optimisation(float * out, const float * in, int n){
  extern __shared__ float sdata[];
  int tid=threadIdx.x;
  int i=blockIdx.x*blockDim.x+threadIdx.x;

  if(i<n){
    sdata[tid]=in[i];
  }
  __syncthreads();

  for(int s=blockDim.x/2;s>32;s/=2){
      if(tid<s){
      sdata[tid]+=sdata[tid+s];
      }
    __syncthreads();
  }

  float val=(tid<32)?sdata[tid]+sdata[tid+32]:0.0f;
  if(tid<32){
    val=warpShuffle(val);
  }

  if(tid==0){
    out[blockIdx.x]=val;
  }
}


int main(){
  int N=1<<22;
  float * h_in;
  float * h_partial;
  float * d_in;
  float * d_partial;

  h_in=(float*)malloc(N*sizeof(float));
  h_partial=(float*)malloc((N+BLOCK_SIZE-1)/BLOCK_SIZE*sizeof(float));

  cudaMalloc(&d_in,N*sizeof(float));
  cudaMalloc(&d_partial,(N+BLOCK_SIZE-1)/BLOCK_SIZE*sizeof(float));

  float ref=0.0f;
  for(int j=0;j<N;j++){
    h_in[j]=1.0f;
    ref+=h_in[j];
  }

  cudaMemcpy(d_in,h_in,N*sizeof(float),cudaMemcpyHostToDevice);

  auto result= [&](auto kernel, const char * name){
    int blocks=(N+BLOCK_SIZE-1)/BLOCK_SIZE;
    int smem=BLOCK_SIZE*sizeof(float);

    //warmup run
    kernel<<<blocks,BLOCK_SIZE,smem>>>(d_partial,d_in,N);
    cudaDeviceSynchronize();

    //harness
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    kernel<<<blocks,BLOCK_SIZE,smem>>>(d_partial,d_in,N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time,start,stop);

    cudaMemcpy(h_partial,d_partial,blocks*sizeof(float),cudaMemcpyDeviceToHost);

    //check
    float res=0.0f;
    for(int i=0;i<blocks;i++){
      res+=h_partial[i];
    }

    if(res!=ref){
      printf("the results = %f, expected = %f \n",res,ref);
    }
    else{
      printf("the number of cycles taken for this operation was %.4f ms \n",elapsed_time);
    }
    };


  result(naive_parallel_reduction,"naive_parallel_reduction");
  result(sequential_addressing_optimisation,"sequential_addressing_optimisation");
  result(warp_shuffle_optimisation,"warp_shuffle_optimisation");

  cudaFree(d_in);
  cudaFree(d_partial);

  free(h_in);
  free(h_partial);

  return 0;

}
