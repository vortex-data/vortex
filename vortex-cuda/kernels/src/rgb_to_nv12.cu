// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/// Converts planar RGB to NV12 for NVENC input.
///
/// Input: three separate R, G, B planes of width*height uint8 values (one byte per pixel).
/// Output: NV12 layout — Y plane (width*height bytes) followed by interleaved UV plane
///         (width * height/2 bytes, with U and V subsampled 2x2).
///
/// Uses BT.601 color space conversion:
///   Y  =  0.299*R + 0.587*G + 0.114*B
///   Cb = -0.169*R - 0.331*G + 0.500*B + 128
///   Cr =  0.500*R - 0.419*G - 0.081*B + 128

extern "C" __global__ void rgb_to_nv12(
    const uint8_t* __restrict__ R,
    const uint8_t* __restrict__ G,
    const uint8_t* __restrict__ B,
    uint8_t* __restrict__ nv12,
    uint32_t width,
    uint32_t height)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    uint32_t idx = y * width + x;

    float r = (float)R[idx];
    float g = (float)G[idx];
    float b = (float)B[idx];

    // BT.601 luma
    float yf = 0.299f * r + 0.587f * g + 0.114f * b;
    nv12[idx] = (uint8_t)fminf(fmaxf(yf + 0.5f, 0.0f), 255.0f);

    // Chroma: only compute for top-left pixel of each 2x2 block
    if ((x & 1) == 0 && (y & 1) == 0) {
        // Average the 2x2 block for chroma subsampling
        float r_sum = r, g_sum = g, b_sum = b;
        int count = 1;

        if (x + 1 < width) {
            uint32_t idx_r = idx + 1;
            r_sum += (float)R[idx_r];
            g_sum += (float)G[idx_r];
            b_sum += (float)B[idx_r];
            count++;
        }
        if (y + 1 < height) {
            uint32_t idx_b = idx + width;
            r_sum += (float)R[idx_b];
            g_sum += (float)G[idx_b];
            b_sum += (float)B[idx_b];
            count++;
        }
        if (x + 1 < width && y + 1 < height) {
            uint32_t idx_br = idx + width + 1;
            r_sum += (float)R[idx_br];
            g_sum += (float)G[idx_br];
            b_sum += (float)B[idx_br];
            count++;
        }

        float inv = 1.0f / (float)count;
        float r_avg = r_sum * inv;
        float g_avg = g_sum * inv;
        float b_avg = b_sum * inv;

        // BT.601 chroma
        float cb = -0.169f * r_avg - 0.331f * g_avg + 0.500f * b_avg + 128.0f;
        float cr =  0.500f * r_avg - 0.419f * g_avg - 0.081f * b_avg + 128.0f;

        // UV plane starts at offset width * height, interleaved U, V
        uint32_t uv_offset = width * height;
        uint32_t uv_row = y / 2;
        uint32_t uv_col = x / 2;
        uint32_t uv_idx = uv_offset + uv_row * width + uv_col * 2;

        nv12[uv_idx]     = (uint8_t)fminf(fmaxf(cb + 0.5f, 0.0f), 255.0f);
        nv12[uv_idx + 1] = (uint8_t)fminf(fmaxf(cr + 0.5f, 0.0f), 255.0f);
    }
}
