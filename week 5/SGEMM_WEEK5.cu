#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>


__global__ void sgemm_naive(int M, int N, int K, float alpha, const float * A, const float * B, float beta,float *C){

    // here deliberately row and column definitions will be changed so we get the non coalescing effects heere row varies and col stays same

    int row=blockIdx.x*blockDim.x+threadIdx.x;
    int col=blockIdx.y*blockDim.y+threadIdx.y;

    float sum=0.0f;
    if(row<M&&col<N){
    for(int k=0;k<K;k++){
         sum+=A[row*K+k]*B[N*k+col];
    }
    C[row*N+col]=alpha*sum+beta*C[row*N+col];
    }

}

__global__ void sgemm_coalesced(int M, int N, int K, float alpha, const float * A, const float * B, float beta,float *C){
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int col=blockIdx.x*blockDim.x+threadIdx.x;

    float sum=0.0f;


    if(row<M&&col<N){
    for(int k=0;k<K;k++){
         sum+=A[row*K+k]*B[N*k+col];
    }
    C[row*N+col]=alpha*sum+beta*C[row*N+col];
    }


}

bool verify(const float * custom, const float * ref,int M, int N){
  int mistakes=0;
  for(int i=0;i<M;i++){
    for(int j=0;j<N;j++){
      float diff=fabsf(custom[i*N+j]-ref[i*N+j]);
      float rel=diff/fabsf(ref[i*N+j]+1e-6f);
      if(diff>1e-2f&&rel>1e-3f){

      mistakes++;
      if(mistakes<=5){ // we want to print for limited number of mistakes
      printf("There was a mismatch in row number %d and column number %d",i,j);}
      }

    }
  }


  if(mistakes>0){
    printf("There were %d mistakes",mistakes);

  }
  return mistakes==0;

}



int main(){
  int M=1024;
  int N=1024;
  int K=1024;
  float alpha=1.0f;
  float beta=0.0f;
  int bytes_A=M*K*sizeof(float);
  int bytes_B=K*N*sizeof(float);
  int bytes_C=M*N*sizeof(float);

  float *h_A,*h_B,*h_C,*d_A,*d_B,*d_C,*h_C_ref,*d_C_ref;

  h_A=(float*)malloc(bytes_A);
  h_B=(float*)malloc(bytes_B);
  h_C=(float*)malloc(bytes_C);
  h_C_ref=(float*)malloc(bytes_C);

  cudaMalloc(&d_A,bytes_A);
  cudaMalloc(&d_B,bytes_B);
  cudaMalloc(&d_C,bytes_C);
  cudaMalloc(&d_C_ref, bytes_C);



  srand(42);

  for(int i=0;i<M;i++){
    for(int j=0;j<K;j++){
      h_A[i*K+j]=(float)(rand()%100)/100.0f;
    }
  }

  for(int i=0;i<K;i++){
    for(int j=0;j<N;j++){
      h_B[i*N+j]=(float)(rand()%100)/100.0f;
    }
  }

  //__cuBLAS implementation of reference

  cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice);

  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);

  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C_ref, N);

  cudaMemcpy(h_C_ref, d_C_ref, bytes_C, cudaMemcpyDeviceToHost);

  cublasDestroy(handle);

  double totalFlops=2.0*M*N*K;
  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  auto deploy=[&](auto kernel, dim3 grid, dim3 block,const char* name){
    cudaMemset(d_C, 0, bytes_C);
    kernel<<<grid,block>>>(M,N,K,alpha,d_A,d_B,beta,d_C);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    kernel<<<grid,block>>>(M,N,K,alpha,d_A,d_B,beta,d_C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_ms;
    cudaEventElapsedTime(&time_ms,start,stop);


    cudaMemcpy(h_C,d_C,bytes_C,cudaMemcpyDeviceToHost);
    bool ok=verify(h_C,h_C_ref,M,N);
    double gflops= (totalFlops)/(time_ms/1000.0)/1e9;

    printf("%.4f ms %.4f GFLOPS \n",time_ms,gflops);
    if(ok){
      printf("Correct\n");
    }
    else{
      printf("Incorrect\n");
    }
  };


  dim3 block(32, 32);
  dim3 grid((M + 31) / 32, (N + 31) / 32);

  deploy(sgemm_naive, grid, block, "naive");
  deploy(sgemm_coalesced, grid, block, "coalesced");

  cublasCreate(&handle);
  cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);   // benchmark cuBLAS itself the same way, for the % comparison
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C_ref, N);  // warm-up
cudaDeviceSynchronize();
  cudaEventRecord(start);
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C_ref, N);
  cudaEventRecord(stop); cudaEventSynchronize(stop);
  float msCublas; cudaEventElapsedTime(&msCublas, start, stop);
  printf("cuBLAS (pedantic):   %8.4f ms   %8.2f GFLOPS\n", msCublas, totalFlops / (msCublas / 1000.0) / 1e9);

  cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_C_ref);

  free(h_A);
  free(h_B);
  free(h_C);
  free(h_C_ref);

  return 0;


}
