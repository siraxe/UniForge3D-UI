import torch
import torch.nn as nn
from .utils import manual_cast

# Row budget for normalizing a large fp16/bf16 activation. `LayerNorm32` promotes to
# fp32, which at the decoder's final level means a second and third full-size copy of a
# multi-GB feature tensor. LayerNorm is per-row, so rows can be processed in chunks with
# no change to the arithmetic.
_NORM_CHUNK_BYTES = 256 * 1024 * 1024


class LayerNorm32(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (x.dtype != torch.float32 and x.dim() >= 2
                and not torch.is_autocast_enabled()
                and x.numel() * 4 > _NORM_CHUNK_BYTES):
            channels = x.shape[-1]
            rows = max(1, _NORM_CHUNK_BYTES // max(1, channels * 4 * 3))
            out = torch.empty_like(x)
            for start in range(0, x.shape[0], rows):
                stop = min(start + rows, x.shape[0])
                out[start:stop] = super().forward(x[start:stop].float())
            return out
        x_dtype = x.dtype
        x = manual_cast(x, torch.float32)
        o = super().forward(x)
        return manual_cast(o, x_dtype)
    

class GroupNorm32(nn.GroupNorm):
    """
    A GroupNorm layer that converts to float32 before the forward pass.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x = manual_cast(x, torch.float32)
        o = super().forward(x)
        return manual_cast(o, x_dtype)
    
    
class ChannelLayerNorm32(LayerNorm32):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        DIM = x.dim()
        x = x.permute(0, *range(2, DIM), 1).contiguous()
        x = super().forward(x)
        x = x.permute(0, DIM-1, *range(1, DIM-1)).contiguous()
        return x
    
