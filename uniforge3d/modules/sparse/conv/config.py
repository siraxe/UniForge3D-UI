import os

SPCONV_ALGO = 'auto'                                # 'auto', 'implicit_gemm', 'native'
# `auto` keeps the fast Triton masked implicit-gemm kernels for convolutions whose
# neighbor map and output fit the sparse package's OFFLOAD_BYTES budget, and switches
# the oversized ones to the chunked, host-offloading implementation in
# conv_flex_gemm.py. This environment has flex_gemm's Triton kernels, but their first
# call also runs an autotuner whose split-K intermediates can be many GB on the final
# 1024-resolution decoder. Force a specific algorithm with UNIFORGE_FLEX_GEMM_ALGO:
# 'explicit_gemm', 'implicit_gemm', 'implicit_gemm_splitk', 'masked_implicit_gemm',
# or 'masked_implicit_gemm_splitk'.
FLEX_GEMM_ALGO = os.environ.get('UNIFORGE_FLEX_GEMM_ALGO', 'auto')
FLEX_GEMM_HASHMAP_RATIO = 2.0                       # Ratio of hashmap size to input size

