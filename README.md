# mmtc - Interface of Matrix Multiplication using Tensor Core for MATLAB

## Installation

1. Download the files.
2. Add the path of "mmtc-main" folder to MATLAB's search path.
3. Execute the following command to compile the mex files:

```
mmtc_compile;
```

If this failed, try the following command and then recompile.

```
setenv("NVCC_APPEND_FLAGS", '-allow-unsupported-compiler');
```

4. You can test `mmtc` by the following command:

```
mmtc_testrun;
```

`mmtc` worked well if you got the results like the followings:

```
ans =
     0
ans =
   1.0000e+00
ans =
   9.7728e-04
ans =
   1.0228e-07
ans =
   3.4093e-08
```

Results may vary slightly depending on the execution environment.

## Usage

```
C = mmtc(A, B, Mode);

%
% Input matrices A & B must be double.
% Output matrix C is double.
% Mode = 1      : mmtc uses INT8 Tensor Core (int8 input, int32 output)
% Mode = 2      : mmtc uses BF16 Tensor Core (BF16 input, FP32 output)
% Mode = others : mmtc uses TF32 Tensor Core (TF32 input, FP32 output)
%
```
