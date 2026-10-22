import math
import torch
import torch.nn as nn
from .. import SparseTensor
from .. import config as sparse_config
from . import config as conv_config
import flex_gemm
from flex_gemm.ops.spconv import sparse_submanifold_conv3d
from flex_gemm.ops.spconv.submanifold_conv3d import SubMConv3dFunction

# The explicit-gemm path builds im2col in plain Torch, so this module builds it in chunks
# and keeps the two transient buffers under these caps: a 27-tap im2col of a million
# voxels at 512 channels is 27 GB in fp16 if it is built whole. Anything larger than the
# sparse package's `OFFLOAD_BYTES` budget is staged through host RAM instead.
#
# The default algorithm (`masked_implicit_gemm_splitk`) uses flex_gemm's Triton kernels
# and never enters this path, which is why it stays the default where those kernels exist.
# Set UNIFORGE_FLEX_GEMM_ALGO=explicit_gemm to trade speed for the chunked/offloading
# implementation when a sparse-convolution stage is the one running out of VRAM.
_IM2COL_CHUNK_BYTES = 64 * 1024 * 1024


def _submanifold_neighbor_map(
    x: SparseTensor,
    kernel_size: tuple[int, int, int],
    dilation: tuple[int, int, int],
) -> tuple[torch.Tensor, bool]:
    """Return the submanifold neighbor map of `x` and whether it has to be streamed.

    A row holds one input index per kernel tap, or -1 when that tap has no voxel. The map
    is cached on the sparse tensor, as flex_gemm's own entry point caches it. Building an
    oversized map costs more VRAM than the convolution it serves, so those are built with
    the features parked in RAM and the caller streams rows back in chunks.
    """
    key = (
        f'SubMConv3d_neighbor_cache_{kernel_size[2]}x{kernel_size[1]}x{kernel_size[0]}'
        f'_dilation{dilation}'
    )
    cache = x.get_spatial_cache(key)
    if cache is not None:
        return cache['neighbor_map'], False

    args = (x.coords, torch.Size([*x.shape, *x.spatial_shape]), kernel_size, dilation)
    # `needs_grad` only controls the extra backward data of the masked kernels; this
    # explicit path needs just the neighbor map, so it is never requested. (The fork's
    # older wheel had no such argument; this venv's flex_gemm requires it.)
    compute_cache = SubMConv3dFunction._compute_neighbor_cache

    if x.feats.shape[0] * math.prod(kernel_size) * 4 < sparse_config.OFFLOAD_BYTES:
        cache = compute_cache(*args, False)
        x.register_spatial_cache(key, cache)
        return cache['neighbor_map'], False

    features_ram = x.feats.cpu()
    x.feats = torch.empty(0, dtype=features_ram.dtype, device='cpu')
    torch.cuda.empty_cache()
    cache = compute_cache(*args, False)
    neighbor_map = cache['neighbor_map'].cpu()
    del cache
    torch.cuda.empty_cache()
    x.feats = features_ram.to(x.coords.device)
    return neighbor_map, True


def sparse_conv3d_init(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=None, bias=True, indice_key=None):
    assert stride == 1 and (padding is None), 'Currently flex_gemm implementation only support submanifold sparse convolution (stride=1, padding=None)'
    
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = tuple(kernel_size) if isinstance(kernel_size, (list, tuple)) else (kernel_size, ) * 3
    self.stride = tuple(stride) if isinstance(stride, (list, tuple)) else (stride, ) * 3
    self.dilation = tuple(dilation) if isinstance(dilation, (list, tuple)) else (dilation, ) * 3

    self.weight = nn.Parameter(torch.empty((out_channels, in_channels, *self.kernel_size)))
    if bias:
        self.bias = nn.Parameter(torch.empty(out_channels))
    else:
        self.register_parameter("bias", None)

    # initialize parameters
    torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    if self.bias is not None:
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    # Permute weight (Co, Ci, Kd, Kh, Kw) -> (Co, Kd, Kh, Kw, Ci)
    self.weight = nn.Parameter(self.weight.permute(0, 2, 3, 4, 1).contiguous())


def _explicit_gemm_forward(self, x: SparseTensor) -> SparseTensor:
    Co, Kd, Kh, Kw, Ci = self.weight.shape
    taps = Kd * Kh * Kw
    neighbor_map, stream_from_cpu = _submanifold_neighbor_map(x, self.weight.shape[1:4], self.dilation)

    feats = x.feats
    voxels = feats.shape[0]
    element_bytes = feats.element_size()
    weight_matrix = self.weight.reshape(Co, taps * Ci).t()  # [taps * Ci, Co]
    chunk = max(1024, _IM2COL_CHUNK_BYTES // (taps * Ci * element_bytes))

    # A large output would overlap the input features it is computed from, so it is
    # accumulated in RAM and only moved to the device once the features are released.
    # Releasing them empties `x.feats`, so a caller that needs these features afterwards
    # must park its own copy in host RAM before the call -- `SparseConvNeXtBlock3d` does
    # that for its residual, under the same `OFFLOAD_BYTES` budget.
    offload = voxels * Co * element_bytes >= sparse_config.OFFLOAD_BYTES
    output = torch.empty((voxels, Co), dtype=feats.dtype, device='cpu' if offload else feats.device)
    for start in range(0, voxels, chunk):
        end = min(start + chunk, voxels)
        rows = neighbor_map[start:end]
        if stream_from_cpu:
            rows = rows.to(feats.device, non_blocking=True)
        # int() turns the uint32 no-neighbour sentinel into -1. Clamping it to 0 lets the
        # gather run unconditionally and the mask zero the rows it must not contribute, so
        # the chunk needs no data-dependent boolean indexing: `flat[found]` and
        # `im2col[found] = ...` each run a `nonzero`, which reads a size back to the host
        # and stalls the device once per chunk.
        flat = rows.reshape(-1).int()
        valid = flat != -1
        im2col = feats.index_select(0, flat.clamp_min(0).long())
        im2col.mul_(valid.unsqueeze(1).to(feats.dtype))
        im2col = im2col.view(end - start, taps * Ci)
        gemm = torch.addmm(self.bias, im2col, weight_matrix) if self.bias is not None else torch.mm(im2col, weight_matrix)
        output[start:end] = gemm.cpu() if offload else gemm
        del im2col, gemm, rows

    if offload:
        device = feats.device
        x.feats = torch.empty(0, dtype=feats.dtype, device='cpu')
        del feats
        torch.cuda.empty_cache()
        output = output.to(device)

    return x.replace(output)


def _needs_host_offload(self, x: SparseTensor) -> bool:
    """Whether a Triton-path convolution would need oversized transient buffers.

    The masked implicit-gemm kernels keep an N x taps neighbor map on the device, and
    their split-K autotuner allocates a (SPLITK, N, Co) float32 intermediate while
    benchmarking. On the final 1024-resolution decoder those can be many GB. Above the
    sparse package's budget, the chunked explicit path is both lower-VRAM and immune
    to the first-use autotuning spike.
    """
    voxels = x.feats.shape[0]
    neighbor_bytes = voxels * math.prod(self.kernel_size) * 4
    output_bytes = voxels * self.out_channels * x.feats.element_size()
    return max(neighbor_bytes, output_bytes) >= sparse_config.OFFLOAD_BYTES


def sparse_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    algorithm = conv_config.FLEX_GEMM_ALGO
    if algorithm == 'explicit_gemm' or (algorithm == 'auto' and _needs_host_offload(self, x)):
        flex_gemm.ops.spconv.set_algorithm('explicit_gemm')
        return _explicit_gemm_forward(self, x)

    # `auto` falls through to the fastest Triton kernel only when the conv's transients
    # fit the same budget. A forced algorithm is passed through unchanged.
    flex_gemm.ops.spconv.set_algorithm(
        'masked_implicit_gemm_splitk' if algorithm == 'auto' else algorithm
    )
    flex_gemm.ops.spconv.set_hashmap_ratio(conv_config.FLEX_GEMM_HASHMAP_RATIO)

    # The installed flex_gemm has Triton kernels, so its own masked implicit-gemm path
    # needs no im2col and is the fast option. This is the upstream path unchanged.
    Co, Kd, Kh, Kw, Ci = self.weight.shape
    neighbor_cache_key = f'SubMConv3d_neighbor_cache_{Kw}x{Kh}x{Kd}_dilation{self.dilation}'
    neighbor_cache = x.get_spatial_cache(neighbor_cache_key)

    out, neighbor_cache_ = sparse_submanifold_conv3d(
        x.feats,
        x.coords,
        torch.Size([*x.shape, *x.spatial_shape]),
        self.weight,
        self.bias,
        neighbor_cache,
        self.dilation
    )

    if neighbor_cache is None:
        x.register_spatial_cache(neighbor_cache_key, neighbor_cache_)

    return x.replace(out)


def sparse_inverse_conv3d_init(self, *args, **kwargs):
    raise NotImplementedError('SparseInverseConv3d with flex_gemm is not implemented yet')


def sparse_inverse_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    raise NotImplementedError('SparseInverseConv3d with flex_gemm is not implemented yet')

