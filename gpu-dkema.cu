/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


const int szblock=4;

typedef unsigned char   uchar;

/*
    Parallel reduction

    This sample shows how to perform a reduction operation on an array of values
    to produce a single value in a single kernel (as opposed to two or more
    kernel calls as shown in the "reduction" CUDA Sample).  Single-pass
    reduction requires Cooperative Groups.

    Reductions are a very common computation in parallel algorithms.  Any time
    an array of values needs to be reduced to a single value using a binary
    associative operator, a reduction can be used.  Example applications include
    statistics computations such as mean and standard deviation, and image
    processing applications such as finding the total luminance of an
    image.

    This code performs sum reductions, but any associative operator such as
    min() or max() could also be used.

    It assumes the input size is a power of 2.

    COMMAND LINE ARGUMENTS

    "--n=<N>":         Specify the number of elements to reduce (default 33554432)
    "--threads=<N>":   Specify the number of threads per block (default 128)
    "--maxblocks=<N>": Specify the maximum number of thread blocks to launch (kernel 6 only, default 64)
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h> 
// includes, project
#include <helper_functions.h>
#include <helper_cuda.h>

#include <cuda_runtime.h>

const char *sSDKsample = "reductionMultiBlockCG";

#include <cuda_runtime_api.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

/*
    Parallel sum reduction using shared memory
    - takes log(n) steps for n input elements
    - uses n/2 threads
    - only works for power-of-2 arrays

    This version adds multiple elements per thread sequentially.  This reduces the overall
    cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
    (Brent's Theorem optimization)

    See the CUDA SDK "reduction" sample for more information.
*/


__device__ __host__ static  uint64_t splitmix64(uint64_t index) {
  uint64_t z = (index + UINT64_C(0x9E3779B97F4A7C15));
  z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
  z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
  return z ^ (z >> 31);
}



__device__ __host__ inline
ulong xorshift64(ulong t)
{
  /* Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs" */
  ulong x = t;
  x ^= x << 13; // a				\
                                                                                              
  x ^= x >> 7; // b				\
                                                                                              
  x ^= x << 17; // c				\
                                                                                              
  return x;
}





void rc4key16(uchar *key, uint16_t *sc, int size_DK) {

  for(int i=0;i<65536;i++) {
    sc[i]=i;
  }


  uint16_t j0 = 0;
  for(int i0=0; i0<65536; i0++) {
    j0 = (j0 + sc[i0] + key[i0%size_DK] )&0xFFFF;
    uint16_t tmp = sc[i0];
    sc[i0] = sc[j0 ];
    sc[j0] = tmp;
  }
}
void rc4key(uchar *key, uchar *sc, int size_DK) {

  for(int i=0;i<256;i++) {
    sc[i]=i;
  }


  uchar j0 = 0;
  for(int i0=0; i0<256; i0++) {
    j0 = (j0 + sc[i0] + key[i0%size_DK] )&0xFF;
    uchar tmp = sc[i0];
    sc[i0] = sc[j0 ];
    sc[j0] = tmp;
  }
}


__global__ void call_mixblock(ulong *g_idata, unsigned int n, unsigned int nb,
		       const uint16_t *Sbox16, const uchar *Sbox8, const uchar *DK)
{
  int idx = blockIdx.x*blockDim.x+threadIdx.x;

  if(idx<n) {
    for(uint i=idx;i<n;i+=nb) {
      ulong r=(i)^DK[i&255]^g_idata[i];
      uchar *rr=(uchar*)&r;                                                                                  
      rr[0]=Sbox8[rr[0]];                                                                                            
      rr[1]=Sbox8[rr[1]];                                                                                            
      rr[2]=Sbox8[rr[2]];                                                                                            
      rr[3]=Sbox8[rr[3]];                                                                                            
      rr[4]=Sbox8[rr[4]];                                                                                            
      rr[5]=Sbox8[rr[5]];                                                                                            
      rr[6]=Sbox8[rr[6]];                                                                                            
      rr[7]=Sbox8[rr[7]];  
      r=xorshift64(r);
      //      r=splitmix64(r);
      //ulong r=xorshift64((i)^g_idata[i]);                                                      
      /*uint16_t *rr=(uint16_t*)&r;                                                                                    
      rr[0]=Sbox16[rr[0]];                                                                                                  rr[1]=Sbox16[rr[1]];                                                                                           
      rr[2]=Sbox16[rr[2]];                                                                                           
      rr[3]=Sbox16[rr[3]];                                                                                           
      */
      

      
      g_idata[i]=r;
      
    }
    
  }
  
}

__device__ __host__ inline uint64_t rotl(const uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
}

__global__ void call_mixblockdiff(ulong *g_idata, unsigned int n, unsigned int nb,
		       const uint16_t *Sbox16, const uchar * __restrict__ Sbox8, const uchar * __restrict__ DK)
{
  int idx = blockIdx.x*blockDim.x+threadIdx.x;



  
  if(idx<n) {
    for(uint i=idx*4;i<n;i+=nb) {


      
      ulong r1=(i)^(((ulong*)DK)[i&31])^g_idata[i];
      r1=rotl(r1,DK[i&255]&63);
      uchar *rr=(uchar*)&r1;                                                                                  
      rr[0]=Sbox8[rr[0]];                                                                                            
      rr[1]=Sbox8[rr[1]];                                                                                            
      rr[2]=Sbox8[rr[2]];                                                                                            
      rr[3]=Sbox8[rr[3]];                                                                                            
      rr[4]=Sbox8[rr[4]];                                                                                            
      rr[5]=Sbox8[rr[5]];                                                                                            
      rr[6]=Sbox8[rr[6]];                                                                                            
      rr[7]=Sbox8[rr[7]];  
      //r1=xorshift64(r1);
      r1=splitmix64(r1);
      
      ulong r2=(i+1)^(((ulong*)DK)[(i+1)&31])^g_idata[i+1];
      r2=rotl(r2,DK[(i+1)&255]&63);
      rr=(uchar*)&r2;                                                                                  
      rr[0]=Sbox8[rr[0]];                                                                                            
      rr[1]=Sbox8[rr[1]];                                                                                            
      rr[2]=Sbox8[rr[2]];                                                                                            
      rr[3]=Sbox8[rr[3]];                                                                                            
      rr[4]=Sbox8[rr[4]];                                                                                            
      rr[5]=Sbox8[rr[5]];                                                                                            
      rr[6]=Sbox8[rr[6]];                                                                                            
      rr[7]=Sbox8[rr[7]];  
      //r2=xorshift64(r2);
      r2=splitmix64(r2);

      ulong r3=(i+2)^(((ulong*)DK)[(i+2)&31])^g_idata[i+2];
      r3=rotl(r3,DK[(i+2)&255]&63);
      rr=(uchar*)&r3;                                                                                  
      rr[0]=Sbox8[rr[0]];                                                                                            
      rr[1]=Sbox8[rr[1]];                                                                                            
      rr[2]=Sbox8[rr[2]];                                                                                            
      rr[3]=Sbox8[rr[3]];                                                                                            
      rr[4]=Sbox8[rr[4]];                                                                                            
      rr[5]=Sbox8[rr[5]];                                                                                            
      rr[6]=Sbox8[rr[6]];                                                                                            
      rr[7]=Sbox8[rr[7]];  
      //r3=xorshift64(r3);
      r3=splitmix64(r3);

      ulong r4=(i+3)^(((ulong*)DK)[(i+3)&31])^g_idata[i+3];
      r4=rotl(r4,DK[(i+3)&255]&63);
      rr=(uchar*)&r4;                                                                                  
      rr[0]=Sbox8[rr[0]];                                                                                            
      rr[1]=Sbox8[rr[1]];                                                                                            
      rr[2]=Sbox8[rr[2]];                                                                                            
      rr[3]=Sbox8[rr[3]];                                                                                            
      rr[4]=Sbox8[rr[4]];                                                                                            
      rr[5]=Sbox8[rr[5]];                                                                                            
      rr[6]=Sbox8[rr[6]];                                                                                            
      rr[7]=Sbox8[rr[7]];  
      //r4=xorshift64(r4);
      r4=splitmix64(r4);

      ulong t=r1^r2^r3^r4;
      rr=(uchar*)&t;                                                                                  
      rr[0]=Sbox8[rr[0]];                                                                                            
      rr[1]=Sbox8[rr[1]];                                                                                            
      rr[2]=Sbox8[rr[2]];                                                                                            
      rr[3]=Sbox8[rr[3]];                                                                                            
      rr[4]=Sbox8[rr[4]];                                                                                            
      rr[5]=Sbox8[rr[5]];                                                                                            
      rr[6]=Sbox8[rr[6]];                                                                                            
      rr[7]=Sbox8[rr[7]];  
      //t=xorshift64(t);
      t=splitmix64(t);
      
      g_idata[i]=r1^t;
      g_idata[i+1]=r2^t;
      g_idata[i+2]=r3^t;
      g_idata[i+3]=r4^t;
      
      
    }
    
  }
  
}






__device__ void reduceBlock(ulong *sdata, const cg::thread_block &cta)
{
    const unsigned int tid = cta.thread_rank();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);
    ulong beta  = sdata[tid];
    ulong temp;

    for (int i = tile32.size()/ 2; i >= szblock; i >>= 1) {
        if (tile32.thread_rank() < i) {
            temp       = sdata[(tid+i)];
            beta       ^= temp;
            sdata[tid] = beta;
        }
	cg::sync(tile32);
    }
    cg::sync(cta);
    if (cta.thread_rank() <4) {
      beta  = 0;
      for (int i = tid; i < blockDim.x; i += tile32.size()) {
	beta  ^= sdata[i];
      }
      sdata[tid] = beta;
    }
    cg::sync(cta);
}

__device__ void  apply_sub_and_prng(ulong *r1, uchar *ssbox) {
  uchar *rr = (uchar*)r1;
  rr[0]=ssbox[rr[0]];
  rr[1]=ssbox[rr[1]];
  rr[2]=ssbox[rr[2]];
  rr[3]=ssbox[rr[3]];
  rr[4]=ssbox[rr[4]];
  rr[5]=ssbox[rr[5]];
  rr[6]=ssbox[rr[6]];
  rr[7]=ssbox[rr[7]];
  *r1=splitmix64(*r1);
}

extern "C" __global__ void reduceSinglePassMultiBlockCG(const ulong * __restrict__  g_idata, ulong * __restrict__ g_odata, unsigned int n,
				  const uchar *Sbox9, const uchar * __restrict__ Sbox8, const uchar * __restrict__ DK)
{
    cg::thread_block block = cg::this_thread_block();
    cg::grid_group grid = cg::this_grid();
    extern ulong __shared__ sdata3[];
    ulong *sdata = (ulong*)sdata3;
    ulong *sdata2 = (ulong*)&sdata3[block.size()];
    uchar *ssbox =(uchar*)&sdata3[block.size()*2];
    sdata[block.thread_rank()] = 0;
    sdata2[block.thread_rank()] = 0;
    if(block.thread_rank()<256) {
      ssbox[block.thread_rank()]=Sbox8[block.thread_rank()];
    }
    uint64_t r1;
    for (int i = grid.thread_rank(); i < n; i += grid.size()) {
      r1=(i)^(((uint64_t*)DK)[i&63])^g_idata[i];
      r1=rotl(r1,ssbox[i&255]&63);
      apply_sub_and_prng(&r1,ssbox);
      sdata2[block.thread_rank()]= r1;
      if ((block.thread_rank()&3) == 0) {
	r1=sdata2[block.thread_rank()]^sdata2[block.thread_rank()+1]^sdata2[block.thread_rank()+2]^sdata2[block.thread_rank()+3];
	apply_sub_and_prng(&r1,ssbox);
        sdata[block.thread_rank()]^=sdata2[block.thread_rank()]^r1;
        sdata[block.thread_rank()+1]^=sdata2[block.thread_rank()+1]^r1;
	sdata[block.thread_rank()+2]^=sdata2[block.thread_rank()+2]^r1;
	sdata[block.thread_rank()+3]^=sdata2[block.thread_rank()+3]^r1;
      }
    }
    sdata2[block.thread_rank()] = 0;
    cg::sync(block);
    reduceBlock(sdata, block);
    if (block.thread_rank() == 0) {
      g_odata[blockIdx.x*szblock] = sdata[0];
      g_odata[blockIdx.x*szblock+1] = sdata[1];
      g_odata[blockIdx.x*szblock+2] = sdata[2];
      g_odata[blockIdx.x*szblock+3] = sdata[3];
    }
    cg::sync(grid);
    if (grid.thread_rank() == 0) {
	for (int i = 1; i < gridDim.x; i++) {
            g_odata[0] ^= g_odata[i*4];
	    g_odata[1] ^= g_odata[i*4+1];
	    g_odata[2] ^= g_odata[i*4+2];
	    g_odata[3] ^= g_odata[i*4+3];
        }
	sdata[0]=g_odata[0]^g_odata[1]^g_odata[2]^g_odata[3];
	apply_sub_and_prng(&sdata[0],ssbox);	
	g_odata[0]=g_odata[0]^sdata[0];
	g_odata[1]=g_odata[1]^sdata[0];
	g_odata[2]=g_odata[2]^sdata[0];
	g_odata[3]=g_odata[3]^sdata[0];
    }
}

////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
void call_reduceSinglePassMultiBlockCG(int size, int threads, int numBlocks, ulong *d_idata, ulong *d_odata,
				       uchar *d_Sbox9,uchar *d_Sbox8, uchar *d_DK)
{
    int smemSize = 2*threads * sizeof(ulong)+256*2;
    void *kernelArgs[] = {
        (void*)&d_idata,
        (void*)&d_odata,
        (void*)&size,
	(void*)&d_Sbox9,
	(void*)&d_Sbox8,
	(void*)&d_DK
    };

    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(numBlocks, 1, 1);

    cudaLaunchCooperativeKernel((void*)reduceSinglePassMultiBlockCG, dimGrid, dimBlock, kernelArgs, smemSize, NULL);
    // check if kernel execution generated an error
    getLastCudaError("Kernel execution failed");
}


////////////////////////////////////////////////////////////////////////////////
// declaration, forward
bool runTest(int argc, char **argv, int device);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv)
{
    cudaDeviceProp deviceProp = { 0 };
    int dev;


    
    printf("%s Starting...\n\n", sSDKsample);

    dev = findCudaDevice(argc, (const char **)argv);
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
    if (!deviceProp.cooperativeLaunch)
    {
        printf("\nSelected GPU (%d) does not support Cooperative Kernel Launch, Waiving the run\n", dev);
        exit(EXIT_WAIVED);
    }

    bool bTestPassed = false;
    bTestPassed = runTest(argc, argv, dev);

    exit(bTestPassed ? EXIT_SUCCESS : EXIT_FAILURE);
}

////////////////////////////////////////////////////////////////////////////////
//! Compute sum reduction on CPU
//! We use Kahan summation for an accurate sum of large arrays.
//! http://en.wikipedia.org/wiki/Kahan_summation_algorithm
//!
//! @param data       pointer to input data
//! @param size       number of input data elements
////////////////////////////////////////////////////////////////////////////////


//RAPH: calcul CPU
template<class T>
T* reduceCPUdiff(T *data, int size, uchar *Sbox9, uchar* Sbox8,uchar* DK)
{
  T *sum=(T*)malloc(sizeof(T)*szblock);
  
  T sum1 = 0;///data[idx];
  T sum2 = 0;///data[idx];
  T sum3 = 0;///data[idx];
  T sum4 = 0;///data[idx];
  for (int i = 0; i < size; i+=szblock)
    {
      
      ulong r1=(i)^(((ulong*)DK)[i&63])^data[i];
      r1=rotl(r1,Sbox8[(i)&255]&63);
      uchar *rr=(uchar*)&r1;                                                                                  
      rr[0]=Sbox8[rr[0]];                                                                                            
      rr[1]=Sbox8[rr[1]];                                                                                            
      rr[2]=Sbox8[rr[2]];                                                                                            
      rr[3]=Sbox8[rr[3]];                                                                                            
      rr[4]=Sbox8[rr[4]];                                                                                            
      rr[5]=Sbox8[rr[5]];                                                                                            
      rr[6]=Sbox8[rr[6]];                                                                                            
      rr[7]=Sbox8[rr[7]];  
      //      r1=xorshift64(r1);
      r1=splitmix64(r1);
      
      ulong r2=(i+1)^(((ulong*)DK)[(i+1)&63])^data[i+1];
      r2=rotl(r2,Sbox8[(i+1)&255]&63);
      rr=(uchar*)&r2;                                                                                  
      rr[0]=Sbox8[rr[0]];                                                                                            
      rr[1]=Sbox8[rr[1]];                                                                                            
      rr[2]=Sbox8[rr[2]];                                                                                            
      rr[3]=Sbox8[rr[3]];                                                                                            
      rr[4]=Sbox8[rr[4]];                                                                                            
      rr[5]=Sbox8[rr[5]];                                                                                            
      rr[6]=Sbox8[rr[6]];                                                                                            
      rr[7]=Sbox8[rr[7]];  
      //r2=xorshift64(r2);
      r2=splitmix64(r2);

      
      ulong r3=(i+2)^(((ulong*)DK)[(i+2)&63])^data[i+2];
      r3=rotl(r3,Sbox8[(i+2)&255]&63);
      rr=(uchar*)&r3;                                                                                  
      rr[0]=Sbox8[rr[0]];                                                                                            
      rr[1]=Sbox8[rr[1]];                                                                                            
      rr[2]=Sbox8[rr[2]];                                                                                            
      rr[3]=Sbox8[rr[3]];                                                                                            
      rr[4]=Sbox8[rr[4]];                                                                                            
      rr[5]=Sbox8[rr[5]];                                                                                            
      rr[6]=Sbox8[rr[6]];                                                                                            
      rr[7]=Sbox8[rr[7]];  
      //r3=xorshift64(r3);
      r3=splitmix64(r3);
      
      ulong r4=(i+3)^(((ulong*)DK)[(i+3)&63])^data[i+3];
      r4=rotl(r4,Sbox8[(i+3)&255]&63);
      rr=(uchar*)&r4;                                                                                  
      rr[0]=Sbox8[rr[0]];                                                                                            
      rr[1]=Sbox8[rr[1]];                                                                                            
      rr[2]=Sbox8[rr[2]];                                                                                            
      rr[3]=Sbox8[rr[3]];                                                                                            
      rr[4]=Sbox8[rr[4]];                                                                                            
      rr[5]=Sbox8[rr[5]];                                                                                            
      rr[6]=Sbox8[rr[6]];                                                                                            
      rr[7]=Sbox8[rr[7]];  
      //r4=xorshift64(r4);
      r4=splitmix64(r4);

      ulong t=r1^r2^r3^r4;
      rr=(uchar*)&t;                                                                                  
      rr[0]=Sbox8[rr[0]];                                                                                            
      rr[1]=Sbox8[rr[1]];                                                                                            
      rr[2]=Sbox8[rr[2]];                                                                                            
      rr[3]=Sbox8[rr[3]];                                                                                            
      rr[4]=Sbox8[rr[4]];                                                                                            
      rr[5]=Sbox8[rr[5]];                                                                                            
      rr[6]=Sbox8[rr[6]];                                                                                            
      rr[7]=Sbox8[rr[7]];  
      //t=xorshift64(t);
      t=splitmix64(t);

      
      sum1^=r1^t;
      sum2^=r2^t;
      sum3^=r3^t;
      sum4^=r4^t;

      /*      printf("lala %lu\n",r1^t);
      printf("lala %lu\n",r2^t);
      printf("lala %lu\n",r3^t);
      printf("lala %lu\n",r4^t);
      */

      
    }


  ulong t=sum1^sum2^sum3^sum4;
  uchar *rr=(uchar*)&t;
  rr[0]=Sbox8[rr[0]];
  rr[1]=Sbox8[rr[1]];
  rr[2]=Sbox8[rr[2]];
  rr[3]=Sbox8[rr[3]];
  rr[4]=Sbox8[rr[4]];
  rr[5]=Sbox8[rr[5]];
  rr[6]=Sbox8[rr[6]];
  rr[7]=Sbox8[rr[7]];
  //t=xorshift64(t);                                                                                                                                                                      
  t=splitmix64(t);

  
  sum[0]=sum1^t;
  sum[1]=sum2^t;
  sum[2]=sum3^t;
  sum[3]=sum4^t;
 

   
    return sum;

}




unsigned int nextPow2(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}


////////////////////////////////////////////////////////////////////////////////
// Compute the number of threads and blocks to use for the reduction
// We set threads / block to the minimum of maxThreads and n/2.
////////////////////////////////////////////////////////////////////////////////
void getNumBlocksAndThreads(int n, int maxBlocks, int maxThreads, int &blocks, int &threads)
{
    if (n == 1)
    {
        threads = 1;
        blocks = 1;
    }
    else
    {
        threads = (n < maxThreads*2) ? nextPow2(n / (2)) : maxThreads;
        blocks = max(1, n / (threads * 2));
	
    }

    blocks = min(maxBlocks, blocks);
}

////////////////////////////////////////////////////////////////////////////////
// This function performs a reduction of the input data multiple times and
// measures the average reduction time.
////////////////////////////////////////////////////////////////////////////////
ulong* benchmarkReduce(int  n,
                      int  numThreads,
                      int  numBlocks,
                      int  maxThreads,
                      int  maxBlocks,
                      int  testIterations,
                      StopWatchInterface *timer,
                      ulong *h_odata,
                      ulong *d_idata,
		       ulong *d_odata, uchar *d_Sbox9,uchar* d_Sbox8, uchar *d_DK)
{
    ulong* gpu_result = (ulong*)malloc(sizeof(ulong)*szblock);
    cudaError_t error;

    printf("\nLaunching %s kernel\n", "SinglePass Multi Block Cooperative Groups");
    for (int i = 0; i < testIterations; ++i)
    {
	for(int j=0;j<szblock;j++)
	  gpu_result[j] = 0;
	  
        sdkStartTimer(&timer);

	dim3 dimg(numThreads/4, 1, 1);
	dim3 dimb(numBlocks, 1, 1);
	unsigned int nb=numThreads*numBlocks;
	//call_mixblock<<<dimg,dimb>>>(d_idata,n,nb,d_Sbox16,d_Sbox8, d_DK);
	//	call_mixblockdiff<<<dimg,dimb>>>(d_idata,n,nb,d_Sbox16,d_Sbox8, d_DK);
	//	cudaDeviceSynchronize();
	//RAPH: debut du calcul
	call_reduceSinglePassMultiBlockCG(n, numThreads, numBlocks, d_idata, d_odata, d_Sbox9,d_Sbox8, d_DK);
        cudaDeviceSynchronize();
        sdkStopTimer(&timer);
    }

    // copy final sum from device to host
    error = cudaMemcpy(gpu_result, d_odata, sizeof(ulong)*szblock, cudaMemcpyDeviceToHost);
    checkCudaErrors(error);

    return gpu_result;
}

////////////////////////////////////////////////////////////////////////////////
// The main function which runs the reduction test.
////////////////////////////////////////////////////////////////////////////////
bool
runTest(int argc, char **argv, int device)
{
    int size = 1 << 28;    // number of elements to reduce
    bool bTestPassed = false;

    if (checkCmdLineFlag(argc, (const char **) argv, "n"))
    {
        size = getCmdLineArgumentInt(argc, (const char **)argv, "n");
    }

    printf("%d elements\n", size);

    // Set the device to be used
    cudaDeviceProp prop = { 0 };
    checkCudaErrors(cudaSetDevice(device));
    checkCudaErrors(cudaGetDeviceProperties(&prop, device));

    // create random input data on CPU
    unsigned int bytes = size * sizeof(ulong);

    ulong *h_idata = (ulong *) malloc(bytes);
    srand(219);

    uchar DK[512];
    for(int i=0;i<512;i++) {
      DK[i]=xorshift64(i+121);
    }
    uchar* Sbox9=(uchar*)malloc(256*sizeof(uchar)); 
    rc4key(&DK[0], Sbox9, 256);  

    uchar* Sbox8=(uchar*)malloc(256*sizeof(uchar)); 
    rc4key(&DK[256], Sbox8, 256);  
    
    printf("ALLOC size of ulong %u\n",sizeof(ulong));

    for (int i = 0; i < size; i++)
    {
        // Keep the numbers small so we don't get truncation error in the sum
      h_idata[i] = rand() ;
      //      printf("%ld ",h_idata[i]);
    }

    
    
    printf("\n");

    // Determine the launch configuration (threads, blocks)
    int maxThreads = 0;
    int maxBlocks = 0;
    int addbit=-1;

    if (checkCmdLineFlag(argc, (const char **) argv, "bit"))
    {
        addbit = getCmdLineArgumentInt(argc, (const char **)argv, "bit");
	h_idata[addbit]++;
	printf("addbit %d\n",addbit);
    }
    if (checkCmdLineFlag(argc, (const char **) argv, "threads"))
    {
        maxThreads = getCmdLineArgumentInt(argc, (const char **)argv, "threads");
    }
    else
    {
        maxThreads = prop.maxThreadsPerBlock;
    }

    if (checkCmdLineFlag(argc, (const char **) argv, "maxblocks"))
    {
        maxBlocks  = getCmdLineArgumentInt(argc, (const char **)argv, "maxblocks");
    }
    else
    {

	//To correct bug on the A100
    	prop.maxThreadsPerMultiProcessor=1024;
        maxBlocks = prop.multiProcessorCount * (prop.maxThreadsPerMultiProcessor / prop.maxThreadsPerBlock);
    }


//    printf("prop.multiProcessorCount %d prop.maxThreadsPerMultiProcessor %d prop.maxThreadsPerBlock %d \n", prop.multiProcessorCount ,prop.maxThreadsPerMultiProcessor , prop.maxThreadsPerBlock);

    int numBlocks = 0;
    int numThreads = 0;
    getNumBlocksAndThreads(size, maxBlocks, maxThreads, numBlocks, numThreads);


    // We calculate the occupancy to know how many block can actually fit on the GPU
    int numBlocksPerSm = 0;
    checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, reduceSinglePassMultiBlockCG, numThreads, numThreads*sizeof(ulong)));

    int numSms = prop.multiProcessorCount;
    if (numBlocks > numBlocksPerSm * numSms)
    {
        numBlocks = numBlocksPerSm * numSms;
    }
    printf("numThreads: %d\n", numThreads);
    printf("numBlocks: %d\n", numBlocks);

    // allocate mem for the result on host side
    ulong *h_odata = (ulong *) malloc(numBlocks*sizeof(ulong)*szblock);

    // allocate device memory and data
    ulong *d_idata = NULL;
    ulong *d_odata = NULL;
    uchar *d_Sbox9 = NULL;
    uchar *d_Sbox8 = NULL;
    uchar *d_DK = NULL;
    
    checkCudaErrors(cudaMalloc((void **) &d_idata, bytes));
    checkCudaErrors(cudaMalloc((void **) &d_odata, numBlocks*sizeof(long)*szblock));
    checkCudaErrors(cudaMalloc((void **) &d_Sbox9, 256*sizeof(uchar)));
    checkCudaErrors(cudaMalloc((void **) &d_Sbox8, 256*sizeof(uchar)));
    checkCudaErrors(cudaMalloc((void **) &d_DK, 512*sizeof(uchar)));
    
   
    StopWatchInterface *transfertimer = 0; 
    sdkCreateTimer(&transfertimer); 
    sdkStartTimer(&transfertimer);
    //sleep(2);
    // copy data directly to device memory
    //RAPH: là on copie du CPU vers le GPU
    checkCudaErrors(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
  

    sdkStopTimer(&transfertimer);
    
    //RAPH: on copie pour la sortie CA SERT A RIEN, j'ai du laisser ca car ca buggait à un moment
    checkCudaErrors(cudaMemcpy(d_odata, h_idata, numBlocks*sizeof(ulong)*szblock, cudaMemcpyHostToDevice));
   
   
    checkCudaErrors(cudaMemcpy(d_Sbox9, Sbox9, 256*sizeof(uchar), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_Sbox8, Sbox8, 256*sizeof(uchar), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_DK, DK, 512*sizeof(uchar), cudaMemcpyHostToDevice));

    //sdkStopTimer(&transfertimer);

   float transferTime = sdkGetAverageTimerValue(&transfertimer);
    printf("Average transfer (HOST-> DEVICE) time: %f ms\n", transferTime);

    int testIterations = 100;

    StopWatchInterface *timer = 0;
    sdkCreateTimer(&timer);

    ulong *gpu_result;


    //RAPH: on lance la routine qui lance 100 itérations
    gpu_result = benchmarkReduce(size, numThreads, numBlocks, maxThreads, maxBlocks,
                                 testIterations, timer, h_odata, d_idata, d_odata, d_Sbox9, d_Sbox8, d_DK);

    float reduceTime = sdkGetAverageTimerValue(&timer);
    printf("Average time: %f ms\n", reduceTime);
    printf("Bandwidth:    %f GB/s\n\n", (size * sizeof(ulong)) / (reduceTime * 1.0e6));
printf("aaaa\n");
    // compute reference solution

  StopWatchInterface *cputimer = 0;
  sdkCreateTimer(&cputimer);
  sdkStartTimer(&cputimer);


ulong *cpu_result = reduceCPUdiff<ulong>(h_idata, size,Sbox9,Sbox8,DK);

  sdkStopTimer(&cputimer);

float cpuTime = sdkGetAverageTimerValue(&cputimer);

 printf("Average CPU time: %f ms\n", cpuTime);


printf("aaaa\n");
    printf("GPU result = ");
    for(int i=0;i<szblock;i++) {
      printf("%lu ", gpu_result[i]);
    }
    printf("\n");
    printf("CPU result = ");
    for(int i=0;i<szblock;i++) {
      printf("%lu ", cpu_result[i]);
    }
    printf("\n");

/*    long threshold = 1e-8 * size;
    long diff = abs((double)gpu_result - (double)cpu_result);
    bTestPassed = (diff < threshold);
*/
    // cleanup
    sdkDeleteTimer(&timer);

    sdkDeleteTimer(&transfertimer);
    free(h_idata);
    free(h_odata);
    cudaFree(d_idata);
    cudaFree(d_odata);

    return bTestPassed;
}

