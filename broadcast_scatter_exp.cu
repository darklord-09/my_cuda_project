#include <cstdio>
#include <cuda_runtime.h>

__constant__ float c_arr[256];

__global__ void broadcast_run(float * out,int n, int iterations){
   int i=blockIdx.x*blockDim.x+threadIdx.x;

   if(i<n){
      float num=0.0f;
      #pragma unroll 1
      for(int j=0;j<iterations;j++){
         num+=c_arr[0];
      }
      out[i]=num;
   }
}

__global__ void scatter_run(float * out,int n, int iterations){
   int i=blockIdx.x*blockDim.x+threadIdx.x;

   if(i<n){
      float num=0.0f;
      int lane=threadIdx.x;
      #pragma unroll 1
      for(int j=0;j<iterations;j++){
         num+=c_arr[lane%32];
      }
      out[i]=num;
   }
}

int main(){
   float h_arr[256];
   for(int j=0;j<256;j++){
    h_arr[j]=(float)j+0.001f;
   }
   cudaMemcpyToSymbol(c_arr,h_arr,sizeof(float)*256);

   float * d_out;
   int n=1<<20;
   size_t bytes=n*sizeof(float);

   cudaMalloc(&d_out,bytes);

   int threads=32;
   int blocks=(n+threads-1)/threads;
   int iterations=2000;

//warm up run to avoid noise
   broadcast_run<<<blocks, threads>>>(d_out, n, iterations);
    scatter_run<<<blocks,threads>>>(d_out,n,iterations);
cudaDeviceSynchronize();


   cudaEvent_t start,stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);

   cudaEventRecord(start);
   broadcast_run<<<blocks,threads>>>(d_out,n,iterations);
   cudaEventRecord(stop);

   cudaEventSynchronize(stop);
   float broadcast_time;
   cudaEventElapsedTime(&broadcast_time,start,stop);

   cudaEventRecord(start);
   scatter_run<<<blocks,threads>>>(d_out,n,iterations);
   cudaEventRecord(stop);

   cudaEventSynchronize(stop);
   float scatter_time;
   cudaEventElapsedTime(&scatter_time,start,stop);

   printf("Broadcast run time %.4f\n", broadcast_time);
   printf("Scatter run time %.4f\n", scatter_time);
   printf("Slowdown time %.4f", scatter_time/broadcast_time);

   cudaFree(d_out);
   return 0;
}


