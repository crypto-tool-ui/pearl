// INT8 -> INT32 full-GEMM reference ceiling. No Pearl noise, transcript or BLAKE3.
// Usage: cublaslt_int8_baseline [M N K repeats]
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

#define CUDA_OK(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { \
  std::fprintf(stderr,"%s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); return 2; } } while(0)
#define LT_OK(call) do { cublasStatus_t s=(call); if(s!=CUBLAS_STATUS_SUCCESS) { \
  std::fprintf(stderr,"%s:%d: cuBLASLt status %d\n",__FILE__,__LINE__,int(s)); return 2; } } while(0)

__global__ void fill_int8(int8_t* p, uint64_t n, uint32_t seed) {
  uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    uint32_t x = uint32_t(i) ^ seed;
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15;
    p[i] = int8_t(int(x % 127) - 63);
  }
}

int main(int argc, char** argv) {
  int M = 56832, N = 56832, K = 4096, repeats = 5;
  if (argc == 5) { M=std::atoi(argv[1]); N=std::atoi(argv[2]); K=std::atoi(argv[3]); repeats=std::atoi(argv[4]); }
  else if (argc != 1) { std::fprintf(stderr,"usage: %s [M N K repeats]\n",argv[0]); return 2; }
  if (M <= 0 || N <= 0 || K < 128 || repeats < 1 || M%128 || N%128 || K%128) {
    std::fprintf(stderr,"M,N,K must be positive multiples of 128; repeats > 0\n"); return 2;
  }
  const uint64_t a_bytes=uint64_t(M)*K, b_bytes=uint64_t(K)*N, c_bytes=uint64_t(M)*N*sizeof(int32_t);
  int device=0; cudaDeviceProp prop{}; CUDA_OK(cudaGetDevice(&device)); CUDA_OK(cudaGetDeviceProperties(&prop,device));
  std::printf("GPU=%s cc=%d.%d M=%d N=%d K=%d A+B=%.3f GiB C=%.3f GiB\n",prop.name,prop.major,prop.minor,M,N,K,
    double(a_bytes+b_bytes)/(1ull<<30),double(c_bytes)/(1ull<<30));
  int8_t *A=nullptr,*B=nullptr; int32_t *C=nullptr; void* workspace=nullptr;
  CUDA_OK(cudaMalloc(&A,a_bytes)); CUDA_OK(cudaMalloc(&B,b_bytes)); CUDA_OK(cudaMalloc(&C,c_bytes)); CUDA_OK(cudaMalloc(&workspace,64ull<<20));
  fill_int8<<<(a_bytes+255)/256,256>>>(A,a_bytes,0x13579bdu);
  fill_int8<<<(b_bytes+255)/256,256>>>(B,b_bytes,0x2468aceu);
  CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize());
  cublasLtHandle_t handle; LT_OK(cublasLtCreate(&handle));
  cublasLtMatmulDesc_t op; LT_OK(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32I,CUDA_R_32I));
  cublasOperation_t trans=CUBLAS_OP_T;
  LT_OK(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&trans,sizeof(trans)));
  LT_OK(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSB,&trans,sizeof(trans)));
  // Column-major views: A points at row-major MxK (thus KxM); B points at
  // row-major KxN (thus NxK). Transpose both to compute the desired MxN GEMM.
  cublasLtMatrixLayout_t a,b,c; LT_OK(cublasLtMatrixLayoutCreate(&a,CUDA_R_8I,K,M,K));
  LT_OK(cublasLtMatrixLayoutCreate(&b,CUDA_R_8I,N,K,N));
  LT_OK(cublasLtMatrixLayoutCreate(&c,CUDA_R_32I,M,N,M));
  cublasLtMatmulPreference_t pref; LT_OK(cublasLtMatmulPreferenceCreate(&pref));
  size_t workspace_limit=64ull<<20;
  LT_OK(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace_limit,sizeof(workspace_limit)));
  cublasLtMatmulHeuristicResult_t results[16]{}; int found=0;
  LT_OK(cublasLtMatmulAlgoGetHeuristic(handle,op,a,b,c,c,pref,16,results,&found));
  if (!found) { std::fprintf(stderr,"cuBLASLt found no INT8/INT32 algorithm for this shape\n"); return 3; }
  const int32_t alpha=1,beta=0;
  cudaEvent_t start,stop; CUDA_OK(cudaEventCreate(&start)); CUDA_OK(cudaEventCreate(&stop));
  int best=-1; float best_ms=1e30f; int32_t sample[8]{};
  for (int i=0;i<found;++i) {
    cublasStatus_t status=cublasLtMatmul(handle,op,&alpha,A,a,B,b,&beta,C,c,C,c,
      &results[i].algo,workspace,workspace_limit,nullptr);
    if(status!=CUBLAS_STATUS_SUCCESS) continue;
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy2D(sample,sizeof(int32_t),C,uint64_t(M)*sizeof(int32_t),
      sizeof(int32_t),8,cudaMemcpyDeviceToHost));
    // Column-major C stores the first output row at stride M.
    std::vector<int8_t> av(K), bv(K);
    CUDA_OK(cudaMemcpy(av.data(),A,K,cudaMemcpyDeviceToHost));
    for(int j=0;j<8;++j) {
      CUDA_OK(cudaMemcpy2D(bv.data(),1,B+j,N,K,1,cudaMemcpyDeviceToHost));
      int64_t sum=0; for(int q=0;q<K;++q) sum+=int32_t(av[q])*int32_t(bv[q]);
      if(sum<INT32_MIN||sum>INT32_MAX||sample[j]!=sum) {
        std::fprintf(stderr,"cuBLASLt output check failed: algo=%d col=%d got=%d expected=%lld\n",i,j,sample[j],(long long)sum); return 4;
      }
    }
    CUDA_OK(cudaEventRecord(start));
    for(int r=0;r<repeats;++r) {
      LT_OK(cublasLtMatmul(handle,op,&alpha,A,a,B,b,&beta,C,c,C,c,
        &results[i].algo,workspace,workspace_limit,nullptr));
    }
    CUDA_OK(cudaEventRecord(stop)); CUDA_OK(cudaEventSynchronize(stop));
    float ms=0; CUDA_OK(cudaEventElapsedTime(&ms,start,stop)); ms/=repeats;
    int algo_id=-1; size_t written=0;
    cublasLtMatmulAlgoConfigGetAttribute(&results[i].algo,CUBLASLT_ALGO_CONFIG_ID,&algo_id,sizeof(algo_id),&written);
    double macs=double(M)*N*K, rate=macs/(double(ms)*1e-3)/1e12;
    std::printf("algo=%d id=%d waves=%d workspace=%zu avg_ms=%.3f T_MAC/s=%.3f\n",
      i,algo_id,results[i].wavesCount,results[i].workspaceSize,ms,rate);
    if(ms<best_ms){best_ms=ms;best=i;}
  }
  if(best<0) { std::fprintf(stderr,"all cuBLASLt algorithms failed\n"); return 5; }
  std::printf("BEST cuBLASLt CUBLAS_COMPUTE_32I: %.3f T_MAC/s at M=%d N=%d K=%d (GEMM only; excludes Pearl transcript/hash)\n",
    double(M)*N*K/(double(best_ms)*1e-3)/1e12,M,N,K);
  return 0;
}
