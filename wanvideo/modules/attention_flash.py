import torch
from ...utils import log

def attention_func_error(*args, **kwargs):
    raise ImportError("Selected attention mode not available. Please ensure required packages are installed correctly.")

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except Exception as e:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except Exception as e:
    FLASH_ATTN_2_AVAILABLE = False

if not FLASH_ATTN_2_AVAILABLE and not FLASH_ATTN_3_AVAILABLE:
    flash_attention = attention_func_error
else:
    def flash_attention(q, k, v,
                        q_lens=None,
                        k_lens=None,
                        dropout_p=0.,
                        softmax_scale=None,
                        q_scale=None,
                        causal=False,
                        window_size=(-1, -1),
                        deterministic=False,
                        dtype=torch.bfloat16,
                        version=None,
                        return_attn=False,):
        half_dtypes = (torch.float16, torch.bfloat16)

        # params
        b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype
        nheads = q.size(2)
        head_dim = q.size(3)

        def half(x):
            return x if x.dtype in half_dtypes else x.to(dtype)

        # preprocess query
        if q_lens is None:
            q = half(q.flatten(0, 1))
            q_lens = torch.tensor(
                [lq] * b, dtype=torch.int32).to(
                    device=q.device, non_blocking=True)
        else:
            q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

        # preprocess key, value
        if k_lens is None:
            k = half(k.flatten(0, 1))
            v = half(v.flatten(0, 1))
            k_lens = torch.tensor(
                [lk] * b, dtype=torch.int32).to(
                    device=k.device, non_blocking=True)
        else:
            k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
            v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

        q = q.to(v.dtype)
        k = k.to(v.dtype)

        if return_attn:
            try:
                # Reshape q, k for manual computation
                q_reshaped = q.view(b, lq, nheads, head_dim)  # [B, Lq, Nq, C1]
                k_reshaped = k.view(b, lk, nheads, head_dim)  # [B, Lk, Nk, C1]

                # Compute QK^T / sqrt(d)
                scale = 1.0 / (head_dim ** 0.5) if softmax_scale is None else softmax_scale
                attn_scores = torch.einsum('bqhd,bkhd->bhqk', q_reshaped, k_reshaped) * scale  # * 2
                # Apply causal mask if needed
                if causal:
                    mask = torch.triu(torch.ones(lq, lk, device=q.device), diagonal=1).bool()
                    attn_scores = attn_scores.masked_fill(mask[None, None, :, :], float('-inf'))

                # Compute softmax
                attn = torch.nn.functional.softmax(attn_scores, dim=-1)  # .detach()  # [B, Nq, Lq, Lk]
            except Exception as e:
                raise Exception(f"Error computing manual attention map: {e}")

        if q_scale is not None:
            q = q * q_scale

        if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
            log.warning('Flash attention 3 is not available, use flash attention 2 instead.')

        if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
            # Note: dropout_p, window_size are not supported in FA3 now.
            x = flash_attn_interface.flash_attn_varlen_func(q=q, k=k, v=v,
                cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                    0, dtype=torch.int32).to(q.device, non_blocking=True),
                cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                    0, dtype=torch.int32).to(q.device, non_blocking=True),
                seqused_q=None, seqused_k=None, max_seqlen_q=lq, max_seqlen_k=lk,
                softmax_scale=softmax_scale, causal=causal,
                deterministic=deterministic)[0].unflatten(0, (b, lq))
        else:
            assert FLASH_ATTN_2_AVAILABLE
            x = flash_attn.flash_attn_varlen_func(q=q, k=k, v=v,
                cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                    0, dtype=torch.int32).to(q.device, non_blocking=True),
                cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                    0, dtype=torch.int32).to(q.device, non_blocking=True),
                max_seqlen_q=lq, max_seqlen_k=lk, dropout_p=dropout_p,
                softmax_scale=softmax_scale, causal=causal, window_size=window_size,
                deterministic=deterministic).unflatten(0, (b, lq))

        if return_attn:
            return x.type(out_dtype), attn
        else:
            return x.type(out_dtype)
