| Matrix | M (tokens) | Kernel | Sparsity | Median ms | Speed-up vs dense | Error / unsupported |
|---|---|---|---|---|---|---|
| q | 1 | dense_fp16 | - | 0.0462 | 1.00x |  |
| q | 1 | dense_fp16_with_zeros | 0.5 | 0.0450 | 1.03x |  |
| q | 1 | csr_spmm_float16 | 0.5 | 0.0942 | 0.49x |  |
| q | 1 | csr_spmm_float32 | 0.5 | 0.0980 | 0.47x |  |
| q | 1 | dense_fp16_with_zeros | 0.75 | 0.0446 | 1.04x |  |
| q | 1 | csr_spmm_float16 | 0.75 | 0.0710 | 0.65x |  |
| q | 1 | csr_spmm_float32 | 0.75 | 0.0712 | 0.65x |  |
| q | 1 | semi_structured_2of4 | - | - | - | RuntimeError: CUDA error: architecture mismatch when calling `cusparseLtInit(&handle)` |
| q | 1 | dense_fp16_2of4_zeros | - | 0.0435 | 1.06x |  |
| q | 1 | int4_unpack_dequant_then_gemm | - | 2.2016 | 0.02x |  |
| q | 1 | int4_unpack_dequant_only | - | 2.1709 | 0.02x |  |
| q | 512 | dense_fp16 | - | 0.1188 | 1.00x |  |
| q | 512 | dense_fp16_with_zeros | 0.5 | 0.1274 | 0.93x |  |
| q | 512 | csr_spmm_float16 | 0.5 | 3.3716 | 0.04x |  |
| q | 512 | csr_spmm_float32 | 0.5 | 3.2532 | 0.04x |  |
| q | 512 | dense_fp16_with_zeros | 0.75 | 0.1659 | 0.72x |  |
| q | 512 | csr_spmm_float16 | 0.75 | 1.6118 | 0.07x |  |
| q | 512 | csr_spmm_float32 | 0.75 | 1.6732 | 0.07x |  |
| q | 512 | semi_structured_2of4 | - | - | - | RuntimeError: CUDA error: architecture mismatch when calling `cusparseLtInit(&handle)` |
| q | 512 | dense_fp16_2of4_zeros | - | 0.1415 | 0.84x |  |
| q | 512 | int4_unpack_dequant_then_gemm | - | 2.3131 | 0.05x |  |
| q | 512 | int4_unpack_dequant_only | - | 2.1795 | 0.05x |  |
| q | 2048 | dense_fp16 | - | 0.3149 | 1.00x |  |
| q | 2048 | dense_fp16_with_zeros | 0.5 | 0.5463 | 0.58x |  |
| q | 2048 | csr_spmm_float16 | 0.5 | 13.2528 | 0.02x |  |
| q | 2048 | csr_spmm_float32 | 0.5 | 13.1279 | 0.02x |  |
| q | 2048 | dense_fp16_with_zeros | 0.75 | 0.4542 | 0.69x |  |
| q | 2048 | csr_spmm_float16 | 0.75 | 6.7428 | 0.05x |  |
| q | 2048 | csr_spmm_float32 | 0.75 | 6.7644 | 0.05x |  |
| q | 2048 | semi_structured_2of4 | - | - | - | RuntimeError: CUDA error: architecture mismatch when calling `cusparseLtInit(&handle)` |
| q | 2048 | dense_fp16_2of4_zeros | - | 0.3604 | 0.87x |  |
| q | 2048 | int4_unpack_dequant_then_gemm | - | 2.6242 | 0.12x |  |
| q | 2048 | int4_unpack_dequant_only | - | 2.1852 | 0.14x |  |
| gate | 1 | dense_fp16 | - | 0.1391 | 1.00x |  |
| gate | 1 | dense_fp16_with_zeros | 0.5 | 0.1378 | 1.01x |  |
| gate | 1 | csr_spmm_float16 | 0.5 | 0.3179 | 0.44x |  |
| gate | 1 | csr_spmm_float32 | 0.5 | 0.3660 | 0.38x |  |
| gate | 1 | dense_fp16_with_zeros | 0.75 | 0.1396 | 1.00x |  |
| gate | 1 | csr_spmm_float16 | 0.75 | 0.1876 | 0.74x |  |
| gate | 1 | csr_spmm_float32 | 0.75 | 0.2043 | 0.68x |  |
| gate | 1 | semi_structured_2of4 | - | - | - | RuntimeError: CUDA error: architecture mismatch when calling `cusparseLtInit(&handle)` |
| gate | 1 | dense_fp16_2of4_zeros | - | 0.1378 | 1.01x |  |
| gate | 1 | int4_unpack_dequant_then_gemm | - | 12.2422 | 0.01x |  |
| gate | 1 | int4_unpack_dequant_only | - | 12.0965 | 0.01x |  |
| gate | 512 | dense_fp16 | - | 0.5813 | 1.00x |  |
| gate | 512 | dense_fp16_with_zeros | 0.5 | 0.8231 | 0.71x |  |
| gate | 512 | csr_spmm_float16 | 0.5 | 19.3203 | 0.03x |  |
| gate | 512 | csr_spmm_float32 | 0.5 | 19.1447 | 0.03x |  |
| gate | 512 | dense_fp16_with_zeros | 0.75 | 0.5698 | 1.02x |  |
| gate | 512 | csr_spmm_float16 | 0.75 | 9.8390 | 0.06x |  |
| gate | 512 | csr_spmm_float32 | 0.75 | 9.7188 | 0.06x |  |
| gate | 512 | semi_structured_2of4 | - | - | - | RuntimeError: CUDA error: architecture mismatch when calling `cusparseLtInit(&handle)` |
| gate | 512 | dense_fp16_2of4_zeros | - | 0.3798 | 1.53x |  |
| gate | 512 | int4_unpack_dequant_then_gemm | - | 12.6271 | 0.05x |  |
| gate | 512 | int4_unpack_dequant_only | - | 12.1037 | 0.05x |  |
| gate | 2048 | dense_fp16 | - | 3.2419 | 1.00x |  |
| gate | 2048 | dense_fp16_with_zeros | 0.5 | 2.9844 | 1.09x |  |
| gate | 2048 | csr_spmm_float16 | 0.5 | 76.9716 | 0.04x |  |
| gate | 2048 | csr_spmm_float32 | 0.5 | 76.9626 | 0.04x |  |
| gate | 2048 | dense_fp16_with_zeros | 0.75 | 2.8397 | 1.14x |  |
| gate | 2048 | csr_spmm_float16 | 0.75 | 39.8008 | 0.08x |  |
| gate | 2048 | csr_spmm_float32 | 0.75 | 39.1926 | 0.08x |  |
| gate | 2048 | semi_structured_2of4 | - | - | - | RuntimeError: CUDA error: architecture mismatch when calling `cusparseLtInit(&handle)` |
| gate | 2048 | dense_fp16_2of4_zeros | - | 2.9947 | 1.08x |  |
| gate | 2048 | int4_unpack_dequant_then_gemm | - | 14.1955 | 0.23x |  |
| gate | 2048 | int4_unpack_dequant_only | - | 12.0897 | 0.27x |  |
| down | 1 | dense_fp16 | - | 0.1381 | 1.00x |  |
| down | 1 | dense_fp16_with_zeros | 0.5 | 0.1371 | 1.01x |  |
| down | 1 | csr_spmm_float16 | 0.5 | 0.3172 | 0.44x |  |
| down | 1 | csr_spmm_float32 | 0.5 | 0.3719 | 0.37x |  |
| down | 1 | dense_fp16_with_zeros | 0.75 | 0.1393 | 0.99x |  |
| down | 1 | csr_spmm_float16 | 0.75 | 0.1838 | 0.75x |  |
| down | 1 | csr_spmm_float32 | 0.75 | 0.2077 | 0.66x |  |
| down | 1 | semi_structured_2of4 | - | - | - | RuntimeError: CUDA error: architecture mismatch when calling `cusparseLtInit(&handle)` |
| down | 1 | dense_fp16_2of4_zeros | - | 0.1355 | 1.02x |  |
| down | 1 | int4_unpack_dequant_then_gemm | - | 12.2449 | 0.01x |  |
| down | 1 | int4_unpack_dequant_only | - | 12.1059 | 0.01x |  |
| down | 512 | dense_fp16 | - | 0.4581 | 1.00x |  |
| down | 512 | dense_fp16_with_zeros | 0.5 | 0.8099 | 0.57x |  |
| down | 512 | csr_spmm_float16 | 0.5 | 19.3597 | 0.02x |  |
| down | 512 | csr_spmm_float32 | 0.5 | 29.2178 | 0.02x |  |
| down | 512 | dense_fp16_with_zeros | 0.75 | 0.8030 | 0.57x |  |
| down | 512 | csr_spmm_float16 | 0.75 | 9.7832 | 0.05x |  |
| down | 512 | csr_spmm_float32 | 0.75 | 13.8816 | 0.03x |  |
| down | 512 | semi_structured_2of4 | - | - | - | RuntimeError: CUDA error: architecture mismatch when calling `cusparseLtInit(&handle)` |
| down | 512 | dense_fp16_2of4_zeros | - | 0.4254 | 1.08x |  |
| down | 512 | int4_unpack_dequant_then_gemm | - | 12.6444 | 0.04x |  |
| down | 512 | int4_unpack_dequant_only | - | 12.0959 | 0.04x |  |
| down | 2048 | dense_fp16 | - | 3.8077 | 1.00x |  |
| down | 2048 | dense_fp16_with_zeros | 0.5 | 3.5712 | 1.07x |  |
| down | 2048 | csr_spmm_float16 | 0.5 | 77.6241 | 0.05x |  |
| down | 2048 | csr_spmm_float32 | 0.5 | 116.7481 | 0.03x |  |
| down | 2048 | dense_fp16_with_zeros | 0.75 | 3.3548 | 1.13x |  |
| down | 2048 | csr_spmm_float16 | 0.75 | 38.6329 | 0.10x |  |
| down | 2048 | csr_spmm_float32 | 0.75 | 55.3349 | 0.07x |  |
| down | 2048 | semi_structured_2of4 | - | - | - | RuntimeError: CUDA error: architecture mismatch when calling `cusparseLtInit(&handle)` |
| down | 2048 | dense_fp16_2of4_zeros | - | 3.6761 | 1.04x |  |
| down | 2048 | int4_unpack_dequant_then_gemm | - | 14.4463 | 0.26x |  |
| down | 2048 | int4_unpack_dequant_only | - | 12.0713 | 0.32x |  |

Block 13 real FP16 weights, 50 reps after 10 warm-up, CUDA events. Storage of the CSR conversions is in analysis/microbench_storage.json.
