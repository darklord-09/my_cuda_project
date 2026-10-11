#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>

#define TILE 32

__global__ void transposer_tiled(float * out, const float * in, float * global_tile, int N){

  int blockId=blockIdx.y*gridDim.x+blockIdx.x;
  float * tile=global_tile+blockId*TILE*TILE;

  int x=blockIdx.x*blockDim.x+threadIdx.x;
  int y=blockIdx.y*blockDim.y+threadIdx.y;

  if(x<N&&y<N){
    tile[threadIdx.y*TILE+threadIdx.x]=in[y*N+x];
  }

  __syncthreads();

  {
  int x=blockIdx.y*TILE+threadIdx.x;
  int y=blockIdx.x*TILE+threadIdx.y;

  if(x<N&&y<N){
    out[y*N+x]=tile[threadIdx.x*TILE+threadIdx.y];
  }
  }

}

int main(){
  int N=4096;

  float * h_A;
  float * h_B;
  float * d_A;
  float * d_B;
  float * d_tile;

  h_A=(float*)malloc(N*N*sizeof(float));
  h_B=(float*)malloc(N*N*sizeof(float));

  cudaMalloc(&d_A,N*N*sizeof(float));
  cudaMalloc(&d_B,N*N*sizeof(float));

  for(int i=0;i<N*N;i++){
    h_A[i]=(float)i;
  }

  cudaMemcpy(d_A,h_A,N*N*sizeof(float),cudaMemcpyHostToDevice);

  dim3 block(TILE,TILE);
  dim3 grid((N+block.x-1)/block.x,(N+block.y-1)/block.y);

  cudaMalloc(&d_tile,grid.x*grid.y*TILE*TILE*sizeof(float));

  transposer_tiled<<<grid,block>>>(d_B,d_A,d_tile,N);
  cudaDeviceSynchronize();

  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  transposer_tiled<<<grid,block>>>(d_B,d_A,d_tile,N);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);

  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time,start,stop);

  cudaMemcpy(h_B,d_B,N*N*sizeof(float),cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();
  bool flag=true;
  for(int j=0;j<N;j++){
    for(int k=0;k<N;k++){
      int A=h_A[j*N+k];
      int B=h_B[k*N+j];
      if(A!=B){
        printf("the data is mismatching at %d and %d", j,k);
        flag=false;
        break;
      }
    }
  }

  if(flag){
    printf("the number of cycles taken for this operation was %.4f ms",elapsed_time);
  }

  cudaFree(d_tile);
  cudaFree(d_A);
  cudaFree(d_B);

  free(h_A);
  free(h_B);


}
