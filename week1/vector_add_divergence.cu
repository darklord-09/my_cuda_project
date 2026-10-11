#include <cstdio>
#include <cstdlib>

__global__ void vecAdd(const float *a,const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && i%2==0) {
    c[i] = a[i] + 2*b[i];
  }
  else if(i<n&& i%2!=0){
    c[i]=a[i]+b[i];
  }
}

int main(){
     int n = 1<<20;
     size_t bytes = n * sizeof(float);
     float *h_a, *h_b, *h_c;
     float *d_a, *d_b, *d_c;

     h_a = (float *)malloc(bytes);
     h_b = (float *)malloc(bytes);
     h_c = (float *)malloc(bytes);


     for(int i=0;i<n;i++){
       h_a[i]=1;
       h_b[i]=2;
     }

     cudaMalloc(&d_a, bytes);
     cudaMalloc(&d_b, bytes);
     cudaMalloc(&d_c, bytes);


     cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
     cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);


     int number_of_threads=1024; //per block threads
     int number_of_blocks=(n+number_of_threads-1)/number_of_threads; //no of blocks assuming roundoff that c causes


//for time measurements to avoid getting wrong results due to noise as there are other factors we do the kernel execution 50 times

for(int j=0;j<50;j++){
     vecAdd<<<number_of_blocks,number_of_threads>>>(d_a,d_b,d_c,n);
}
     cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

     //optional verification starts
     bool ok=true;
     for(int j=0;j<n;j++){
      if(j%2!=0&&h_c[j]-3.0f>1e-5){
        ok=false;
        break;
      }
      else if(j%2==0&&h_c[j]-5.0f>1e-5){
        ok=false;
        break;
      }
     }

     printf(ok ? "Verified successfully" : "failed");

     //optional verification ends
     free(h_a);
     free(h_b);
     free(h_c);

     cudaFree(d_a);
     cudaFree(d_b);
     cudaFree(d_c);

     return 0;


}
