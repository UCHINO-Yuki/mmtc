/**
 * @file mmtc.cu
 * @author Yuki UCHINO
 * @brief 
 * @version 1.0
 * @date 2024-07-26
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_bf16.h>

#define NTX 256	// 32 <= NTX <= 1024

//=====
// cast
//=====
template <typename in, typename out>
__global__ void Cast_Kernel (
	const size_t m,		    // #elements
	const size_t n,		    // #elements
	const in *devin,		// input
	const size_t ldin,	    // leading dimension
	out *devout,	        // output
	const size_t ldout	    // leading dimension
) {
	const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= m*n) {
		return;
	}
	const unsigned mi = tid % ldin;
	const unsigned ni = tid / ldin;
	const unsigned idx = ni * ldout + mi;
	devout[idx] = (out)devin[tid];
}

template <>
__global__ void Cast_Kernel (
	const size_t m,		    // #elements
	const size_t n,		    // #elements
	const double *devin,	// input
	const size_t ldin,	    // leading dimension
	__nv_bfloat16 *devout,	// output
	const size_t ldout	    // leading dimension
) {
	const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= m*n) {
		return;
	}
	const unsigned mi = tid % ldin;
	const unsigned ni = tid / ldin;
	const unsigned idx = ni * ldout + mi;
	devout[idx] = __double2bfloat16(devin[tid]);
}

template <typename in, typename out>
__host__ void Cast_Device (
	const size_t m,			// #rows
	const size_t n,			// #columns
	const in *devin,		// input
	const size_t ldin,	    // leading dimension
	out *devout,	        // output
	const size_t ldout		// leading dimension
) {
    size_t N = m * n;
	dim3 threads = NTX;	// <= 1024
	dim3 grid = (N + NTX - 1) / NTX;
	Cast_Kernel <<< grid, threads >>> (m, n, devin, ldin, devout, ldout);
}


template <typename in>
__global__ void Cast_Kernel2 (
	const size_t m,		    // #elements
	const size_t n,		    // #elements
	const in *devin,		// input
	const size_t ldin,	    // leading dimension
	double *devout,	        // output
	const size_t ldout	    // leading dimension
) {
	const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= m*n) {
		return;
	}
	const unsigned mi = tid % ldout;
	const unsigned ni = tid / ldout;
	const unsigned idx = ni * ldin + mi;
	devout[tid] = (double)devin[idx];
}

template <typename in>
__host__ void Cast_Device2 (
	const size_t m,			// #rows
	const size_t n,			// #columns
	const in *devin,		// input
	const size_t ldin,	    // leading dimension
	double *devout,	        // output
	const size_t ldout		// leading dimension
) {
    size_t N = m * n;
	dim3 threads = NTX;	// <= 1024
	dim3 grid = (N + NTX - 1) / NTX;
	Cast_Kernel2 <<< grid, threads >>> (m, n, devin, ldin, devout, ldout);
}


//=====
// main
//=====
void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	//=====
	// input
	//=====
	mxGPUArray const *A = mxGPUCreateFromMxArray(prhs[0]);				// create mxGPUArray object from A
	mxGPUArray const *B = mxGPUCreateFromMxArray(prhs[1]);				// create mxGPUArray object from B
	double const *devA = (double const *)mxGPUGetDataReadOnly(A);		// read-only pointer to the underlying data of A
	double const *devB = (double const *)mxGPUGetDataReadOnly(B);		// read-only pointer to the underlying data of B
	const mwSize *dimA = mxGPUGetDimensions(A);							// dimensions of A
	const mwSize *dimB = mxGPUGetDimensions(B);							// dimensions of B
	size_t m = dimA[0];													// #rows of A
	size_t k = dimA[1];													// #columns of A
	size_t n = dimB[1];													// #columns of B
    size_t mode = (size_t)mxGetScalar(prhs[2]);                         // Computation mode

	//=====
	// output
	//=====
	mwSize const ndims = mxGPUGetNumberOfDimensions(A);					// #dimensions of A
	size_t dimC[2] = {m,n};												// dimensions of C
	mxClassID const cid = mxGPUGetClassID(A);							// specifying the element class
	mxComplexity const cx = mxGPUGetComplexity(A);						// specifying the complexity
	mxGPUInitialize const init = MX_GPU_DO_NOT_INITIALIZE;				// specifying whether to initialize elements values to 0
	mxGPUArray *C = mxGPUCreateGPUArray(ndims, dimC, cid, cx, init);	// create mxGPUArray object 
	double *devC = (double *)mxGPUGetData(C);							// pointer to the underlying data of devC

	cublasHandle_t ch;
	cublasCreate(&ch);
	cublasSetPointerMode(ch, CUBLAS_POINTER_MODE_HOST);

    if (mode == 1) {
        // INT8
		using IN = int8_t;
		using OUT = int32_t;
        const auto typeIN = CUDA_R_8I;
        const auto typeOUT = CUDA_R_32I;
        const auto CompMode = CUBLAS_COMPUTE_32I;

	    const OUT alpha = 1;
	    const OUT beta  = 0;
	    IN *devAtmp, *devBtmp;
	    OUT *devCtmp;
	    size_t pitchA, pitchB, pitchC;
	    cudaMallocPitch((void **)&devAtmp, &pitchA, sizeof(IN) * m, k);
	    cudaMallocPitch((void **)&devBtmp, &pitchB, sizeof(IN) * k, n);
	    cudaMallocPitch((void **)&devCtmp, &pitchC, sizeof(OUT) * m, n);
	    size_t lda = pitchA/sizeof(IN);
	    size_t ldb = pitchB/sizeof(IN);
	    size_t ldc = pitchC/sizeof(OUT);
	    cudaDeviceSynchronize();
	    Cast_Device (m, k, devA, m, devAtmp, lda);
	    Cast_Device (k, n, devB, k, devBtmp, ldb);

	    cudaDeviceSynchronize();
	    cublasGemmEx (ch, CUBLAS_OP_N, CUBLAS_OP_N, 
		    m, n, k,
	        &alpha, 
	        devAtmp, typeIN, lda,
	        devBtmp, typeIN, ldb,
	        &beta, 
	        devCtmp, typeOUT, ldc,
	        CompMode, CUBLAS_GEMM_DEFAULT);
    
	    cudaDeviceSynchronize();
	    Cast_Device2 (m, n, devCtmp, ldc, devC, m);

	    plhs[0] = mxGPUCreateMxArrayOnGPU(C);
	    cublasDestroy(ch);
	    mxGPUDestroyGPUArray(A);
	    mxGPUDestroyGPUArray(B);
	    mxGPUDestroyGPUArray(C);
	    cudaFree(devAtmp);
	    cudaFree(devBtmp);
	    cudaFree(devCtmp);

    } else if (mode == 2) {
        // BF16
		using IN = __nv_bfloat16;
		using OUT = float;
        const auto typeIN = CUDA_R_16BF;
        const auto typeOUT = CUDA_R_32F;
        const auto CompMode = CUBLAS_COMPUTE_32F;
        
	    const OUT alpha = 1.0;
	    const OUT beta  = 0.0;
	    IN *devAtmp, *devBtmp;
	    OUT *devCtmp;
	    size_t pitchA, pitchB, pitchC;
	    cudaMallocPitch((void **)&devAtmp, &pitchA, sizeof(IN) * m, k);
	    cudaMallocPitch((void **)&devBtmp, &pitchB, sizeof(IN) * k, n);
	    cudaMallocPitch((void **)&devCtmp, &pitchC, sizeof(OUT) * m, n);
	    size_t lda = pitchA/sizeof(IN);
	    size_t ldb = pitchB/sizeof(IN);
	    size_t ldc = pitchC/sizeof(OUT);
	    cudaDeviceSynchronize();
	    Cast_Device (m, k, devA, m, devAtmp, lda);
	    Cast_Device (k, n, devB, k, devBtmp, ldb);

	    cudaDeviceSynchronize();
	    cublasGemmEx (ch, CUBLAS_OP_N, CUBLAS_OP_N, 
		    m, n, k,
	        &alpha, 
	        devAtmp, typeIN, lda,
	        devBtmp, typeIN, ldb,
	        &beta, 
	        devCtmp, typeOUT, ldc,
	        CompMode, CUBLAS_GEMM_DEFAULT);
    
	    cudaDeviceSynchronize();
	    Cast_Device2 (m, n, devCtmp, ldc, devC, m);

	    plhs[0] = mxGPUCreateMxArrayOnGPU(C);
	    cublasDestroy(ch);
	    mxGPUDestroyGPUArray(A);
	    mxGPUDestroyGPUArray(B);
	    mxGPUDestroyGPUArray(C);
	    cudaFree(devAtmp);
	    cudaFree(devBtmp);
	    cudaFree(devCtmp);

    } else {
        // TF32
		using IN = float;
		using OUT = float;
        const auto typeIN = CUDA_R_32F;
        const auto typeOUT = CUDA_R_32F;
        const auto CompMode = CUBLAS_COMPUTE_32F;
	    cublasSetMathMode(ch, CUBLAS_TF32_TENSOR_OP_MATH);
        
	    const OUT alpha = 1.0;
	    const OUT beta  = 0.0;
	    IN *devAtmp, *devBtmp;
	    OUT *devCtmp;
	    size_t pitchA, pitchB, pitchC;
	    cudaMallocPitch((void **)&devAtmp, &pitchA, sizeof(IN) * m, k);
	    cudaMallocPitch((void **)&devBtmp, &pitchB, sizeof(IN) * k, n);
	    cudaMallocPitch((void **)&devCtmp, &pitchC, sizeof(OUT) * m, n);
	    size_t lda = pitchA/sizeof(IN);
	    size_t ldb = pitchB/sizeof(IN);
	    size_t ldc = pitchC/sizeof(OUT);
	    cudaDeviceSynchronize();
	    Cast_Device (m, k, devA, m, devAtmp, lda);
	    Cast_Device (k, n, devB, k, devBtmp, ldb);

	    cudaDeviceSynchronize();
	    cublasGemmEx (ch, CUBLAS_OP_N, CUBLAS_OP_N, 
		    m, n, k,
	        &alpha, 
	        devAtmp, typeIN, lda,
	        devBtmp, typeIN, ldb,
	        &beta, 
	        devCtmp, typeOUT, ldc,
	        CompMode, CUBLAS_GEMM_DEFAULT);
    
	    cudaDeviceSynchronize();
	    Cast_Device2 (m, n, devCtmp, ldc, devC, m);

	    plhs[0] = mxGPUCreateMxArrayOnGPU(C);
	    cublasDestroy(ch);
	    mxGPUDestroyGPUArray(A);
	    mxGPUDestroyGPUArray(B);
	    mxGPUDestroyGPUArray(C);
	    cudaFree(devAtmp);
	    cudaFree(devBtmp);
	    cudaFree(devCtmp);
    }

}
