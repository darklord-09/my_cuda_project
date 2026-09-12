#include<cstdio>
#include<cmath>
#include<cuda_runtime.h>

__global__ void transposer(float *A,float *B,int N){
  int col=blockIdx.x*blockDim.x+threadIdx.x;
  int row=blockIdx.y*blockDim.y+threadIdx.y;

  if(row<N&&col<N){
    B[col*N+row]=A[row*N+col];
  }
}

int main(){
    float * h_A;
    float * h_B;
    float * d_A;
    float * d_B;
    int N=4096;
    h_A=(float*)malloc(N*N*sizeof(float));
    h_B=(float*)malloc(N*N*sizeof(float));
    cudaMalloc(&d_A,N*N*sizeof(float));
    cudaMalloc(&d_B,N*N*sizeof(float));
    for(int i=0;i<N*N;i++){
        h_A[i]=(float)i;
    }
    cudaMemcpy(d_A,h_A,N*N*sizeof(float),cudaMemcpyHostToDevice);

    dim3 block(32,32);
    dim3 grid((N+block.x-1)/block.x,(N+block.y-1)/block.y);

    //warm up run
    transposer<<<grid,block>>>(d_A,d_B,N);
    cudaDeviceSynchronize();

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    transposer<<<grid,block>>>(d_A,d_B,N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaMemcpy(h_B,d_B,N*N*sizeof(float),cudaMemcpyDeviceToHost);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time,start,stop);
    cudaDeviceSynchronize();
     

    bool correct=true; 
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            int A_val=h_A[i*N+j];
            int expected_B=h_B[j*N+i];
          
            if(A_val!=expected_B){
              printf("mistake in row = %d col = %d",i,j);
              correct=false;
              break;
            }
        }
    }

    if(correct){
      printf("the transponse worked correctly with time taken = %f cycles",elapsed_time);
    }

    cudaFree(d_A);
    cudaFree(d_B);

    free(h_A);
    free(h_B);

    return 0;      

}
