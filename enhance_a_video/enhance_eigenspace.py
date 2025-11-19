import torch
from einops import rearrange
from .globals import get_num_frames


@torch.compiler.disable()
def eigenspace_enhanced_attention(attention_scores, base_temperature, eigenspace_threshold,
                                  min_rank_preservation, adaptive_scaling):
    """
    Apply eigenspace-based enhancement to temporal attention matrices

    Args:
        attention_scores: Attention matrices, shape [(b*s), n, t, t] = [3200, 24, 5, 5]
        base_temperature: Base temperature parameter
        eigenspace_threshold: Minimum CFI for enhancement
        min_rank_preservation: Minimum rank preservation ratio
        adaptive_scaling: Enable adaptive temperature scaling
    """

    bs_spatial, n_heads, t_frames, _ = attention_scores.shape
    device = attention_scores.device

    # Process each (batch*spatial, head) combination
    enhanced_temp_list = []
    for bs in range(bs_spatial):
        for h in range(n_heads):
            # Extract temporal attention matrix [t, t] = [5, 5]
            attn_matrix = attention_scores[bs, h]  # Shape: [5, 5]

            # Compute eigenspace-based CFI and temperature
            enhanced_temp = compute_eigenspace_temperature(
                attn_matrix, base_temperature, eigenspace_threshold,
                min_rank_preservation, adaptive_scaling
            )

            # Apply enhanced temperature
            enhanced_temp_list.append(enhanced_temp)

    enhanced_temp_mean = 1 / torch.tensor(enhanced_temp_list, device=device).mean()
    print(f"Mean Enhanced Temperature: {enhanced_temp_mean:.4f}")
    return enhanced_temp_mean


@torch.compiler.disable()
def compute_eigenspace_temperature(attn_matrix, base_temp, threshold, min_rank_preservation, adaptive_scaling):
    """
    Compute adaptive temperature based on eigenspace analysis of temporal attention matrix

    Args:
        attn_matrix: Single temporal attention matrix, shape [t, t] = [5, 5]
        base_temp: Base temperature parameter
        threshold: CFI threshold for enhancement
        min_rank_preservation: Minimum rank preservation ratio
        adaptive_scaling: Enable adaptive temperature scaling
    """

    t_size = attn_matrix.shape[0]  # t = 5

    try:
        # Compute SVD decomposition of the temporal attention matrix
        U, S, Vh = torch.linalg.svd(attn_matrix)

        # Compute spectral Cross-Frame Intensity (CFI)
        # Weight off-diagonal contributions by singular values
        eye_mask = torch.eye(t_size, device=attn_matrix.device)
        off_diag_mask = 1.0 - eye_mask

        # Reconstruct off-diagonal components weighted by singular values
        reconstructed = U @ torch.diag(S) @ Vh
        off_diag_contribution = (reconstructed * off_diag_mask).abs().sum()
        total_magnitude = S.sum()

        # Spectral CFI: weighted off-diagonal energy
        spectral_cfi = off_diag_contribution / (total_magnitude + 1e-8)

        if not adaptive_scaling:
            # Simple enhancement based on CFI threshold
            if spectral_cfi > threshold:
                return base_temp * 0.7  # Increase cross-frame attention
            else:
                return base_temp

        # Advanced adaptive scaling based on spectral properties

        # 1. Compute effective rank using entropy of normalized singular values
        normalized_s = S / (S.sum() + 1e-8)
        entropy = -(normalized_s * torch.log(normalized_s + 1e-8)).sum()
        effective_rank = torch.exp(entropy) / t_size  # Normalize by max possible rank

        # 2. Compute spectral concentration (dominance of leading eigenvalue)
        spectral_concentration = S[0] / (S.sum() + 1e-8)

        # 3. Compute spectral gap between first and second eigenvalues
        if len(S) > 1:
            spectral_gap = (S[0] - S[1]) / (S[0] + 1e-8)
        else:
            spectral_gap = torch.tensor(1.0, device=attn_matrix.device)

        # 4. Adaptive temperature computation
        # Low effective rank + high spectral concentration = need more enhancement
        # High spectral gap = well-separated modes, can handle more enhancement

        rank_factor = 1.0 - effective_rank  # Higher when rank is low
        concentration_factor = spectral_concentration  # Higher when concentrated
        gap_factor = spectral_gap  # Higher when modes are well-separated

        # Combine factors to determine enhancement strength
        enhancement_strength = (rank_factor * 0.4 + concentration_factor * 0.3 + gap_factor * 0.3)

        # Apply rank preservation constraint
        if effective_rank < min_rank_preservation:
            # Reduce enhancement to prevent excessive rank collapse
            enhancement_strength *= (effective_rank / min_rank_preservation)

        # Compute adaptive temperature
        # Lower temperature = stronger enhancement of cross-frame attention
        temp_reduction = enhancement_strength * 0.5  # Max 50% temperature reduction
        adaptive_temp = base_temp * (1.0 - temp_reduction)

        # Ensure temperature doesn't go too low
        adaptive_temp = torch.clamp(adaptive_temp, min=base_temp * 0.3, max=base_temp * 1.5)

        return adaptive_temp

    except Exception as e:
        # Fallback to original temperature if SVD fails
        print(f"SVD computation failed: {e}")
        return base_temp


@torch.compiler.disable()
def get_feta_scores(query, key):
    img_q, img_k = query, key

    num_frames = get_num_frames()

    B, S, N, C = img_q.shape

    # Calculate spatial dimension
    spatial_dim = S // num_frames

    # Add time dimension between spatial and head dims
    query_image = img_q.reshape(B, spatial_dim, num_frames, N, C)
    key_image = img_k.reshape(B, spatial_dim, num_frames, N, C)

    # Expand time dimension
    query_image = query_image.expand(-1, -1, num_frames, -1, -1)  # [B, S, T, N, C]
    key_image = key_image.expand(-1, -1, num_frames, -1, -1)  # [B, S, T, N, C]

    # Reshape to match feta_score input format: [(B S) N T C]
    query_image = rearrange(query_image, "b s t n c -> (b s) n t c")  # torch.Size([3200, 24, 5, 128])
    key_image = rearrange(key_image, "b s t n c -> (b s) n t c")

    return feta_score(query_image, key_image, C, num_frames)


@torch.compiler.disable()
def feta_score(query_image, key_image, head_dim, num_frames):
    scale = head_dim ** -0.5
    query_image = query_image * scale
    attn_temp = query_image @ key_image.transpose(-2, -1)  # translate attn to float32
    attn_temp = attn_temp.to(torch.float32)

    temperature = 1.0
    eigenspace_threshold = 0.1
    min_rank_preservation = 0.8
    adaptive_scaling = True

    enhanced_scores = eigenspace_enhanced_attention(
        attn_temp, temperature, eigenspace_threshold,
        min_rank_preservation, adaptive_scaling
    )

    return enhanced_scores


# if __name__ == '__main__':
#     import time
#     batch_size = 1
#     num_frames = get_num_frames()
#     S = num_frames * 256
#     head_dim = 24
#     C = 128
#     query_image = torch.randn(batch_size, S, head_dim, C)
#     key_image = torch.randn(batch_size, S, head_dim, C)
#
#     start_time = time.time()
#     enhance_scores = get_feta_scores(query_image, key_image)
#     print(f'Enhance scores: {enhance_scores.shape}')
#     print('Time taken:', time.time() - start_time)