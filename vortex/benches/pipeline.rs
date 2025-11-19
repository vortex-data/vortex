// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmark suite for hand-rolled BP -> FoR -> ALP decode pipeline.
//!
//! This benchmark compares different decompression strategies:
//! - Batch decompression with separate buffers for each stage
//! - Pipeline decompression with chunked processing
//! - In-place batch decompression reusing buffers
//! - In-place pipeline decompression with minimal memory usage
//!
//! The pipeline consists of three stages:
//! 1. Bitpacking decompression (10-bit values to 32-bit)
//! 2. Frame of Reference (FoR) decompression
//! 3. Adaptive Lossless Floating-point (ALP) decompression
//!
//! # Pipeline Decompression Performance Analysis
//!
//! ## Setup
//!
//! Benchmarks were run on an M4 Max MacBook (2024) with 128GB RAM. The M4 Max has 128KB L1 data
//! cache (per performance core).
//!
//! ## Benchmark Results
//!
//! Testing across multiple data sizes shows consistent performance patterns:
//!
//! | Size (elements) | Batch (us) | In-Place Batch (us) | Pipeline (us) | In-Place Pipeline (us) |
//! | --------------- | ---------- | ------------------- | ------------- | ---------------------- |
//! | 1,024           | 0.134      | 0.129               | 0.133         | 0.131                  |
//! | 16,384          | 3.187      | 2.124               | 2.166         | 2.082                  |
//! | 65,536          | 13.87      | 12.66               | 9.582         | 10.37                  |
//! | 73,728          | 15.58      | 14.20               | 10.66         | 11.66                  |
//! | 86,016          | 18.24      | 16.58               | 12.41         | 13.62                  |
//! | 100,352         | 21.33      | 19.29               | 14.37         | 15.91                  |
//!
//! Pipeline processing achieves 33% better performance than batch at 100K elements. The
//! performance gap emerges at 16K elements and remains stable for larger sizes.
//!
//! ## Cache Locality Advantage
//!
//! The pipeline approach processes data in 1,024-element chunks (4KB). Each chunk fits entirely
//! in L1 data cache on the M4 Max. L1 cache provides sub-nanosecond latency whereas L2 will have
//! latency in the nanosecond range (it is not consistent because L2 is shared in Apple
//! processors [1]).
//!
//! Processing all three stages while data resides in L1 eliminates cache misses. The batch
//! approach must reload the entire dataset from L2/L3 for each stage. For 100K elements (400KB),
//! batch processing performs three full passes through memory. Pipeline processing performs the
//! same memory reads but maintains temporal locality within each 4KB chunk.
//!
//! Measured memory bandwidth utilization confirms this advantage. Pipeline processing achieves
//! 26.0 GB/s effective bandwidth versus 17.8 GB/s for batch processing when processing 1MB of
//! data. The 46% improvement comes from keeping data in L1 throughout the transformation chain.
//!
//! ## In-Place Performance Penalty
//!
//! In-place processing reuses the output buffer for all intermediate stages. This creates
//! store-to-load forwarding delays. When the processor writes to an address and immediately reads
//! from it, the load must wait for the store to complete. ARM processors typically incur a 4-5
//! cycle penalty for this pattern [2].
//!
//! Regular pipeline writes to separate buffers:
//!
//! ```text
//! Read from buffer A -> Process -> Write to buffer B
//! Read from buffer B -> Process -> Write to buffer C
//! ```
//!
//! In-place pipeline creates dependencies:
//!
//! ```text
//! Read from buffer X -> Process -> Write to buffer X
//! Read from buffer X (must wait for write) -> Process -> Write to buffer X
//! ```
//!
//! Each 1,024-element chunk encounters this penalty twice. Once in the FoR stage and once in the
//! ALP stage. The measured 8-10% performance penalty aligns with the theoretical overhead of 2,048
//! store-to-load delays per chunk.
//!
//! ---
//!
//! [1] <https://www.reddit.com/r/hardware/comments/1gyh42k/david_huang_tests_apple_m4_pro/>
//! [2] <https://chipsandcheese.com/p/cortex-x2-arm-aims-high>

#![allow(
    clippy::unwrap_used,
    clippy::uninit_vec,
    clippy::cast_possible_truncation
)]

use std::time::Duration;

use divan::Bencher;
use fastlanes::BitPacking;
use rand::Rng;
use vortex_alp::{ALPFloat, Exponents};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Constants
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Size of each chunk.
const N: usize = 1024;

/// The width of each bitpacked value.
const W: usize = 10;

/// The width of the unpacked i32 values.
const T: usize = 32;

/// The bitpacked stride that makes up 1024 bits.
const S: usize = N * W / T;

/// Benchmark sizes to test for performance benchmarks.
const BENCHMARK_SIZES: [usize; 8] = [
    1024,   // 1K
    8192,   // 8K
    16384,  // 16K
    65536,  // 64K
    73728,  // 72K
    86016,  // 84K
    100352, // 98K
    262144, // 256K
];

/// Sizes to test for correctness verification.
const VERIFICATION_SIZES: [usize; 2] = [
    1024,  // 1K - minimum size
    16384, // 16K - medium size
];

////////////////////////////////////////////////////////////////////////////////////////////////////
// Main
////////////////////////////////////////////////////////////////////////////////////////////////////

fn main() {
    divan::main();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Data Structures
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Input data for decompression benchmarks.
///
/// Contains the compressed data and metadata needed for decompression.
struct InputData {
    /// Bitpacked compressed data.
    bitpacked: Vec<u32>,
    /// Reference value for FoR decompression.
    reference: i32,
    /// Exponent values for ALP decompression.
    exponents: Exponents,
    /// Original values for verification.
    original: Vec<f32>,
    /// ALP-encoded values for intermediate verification.
    alp_encoded: Vec<i32>,
    /// Patch information for ALP decompression verification.
    patches: Patches,
}

/// Pre-allocated buffers for benchmark operations.
///
/// These buffers are allocated once and reused across benchmark iterations
/// to avoid measuring allocation overhead.
struct BenchmarkBuffers {
    /// Intermediate buffer for unpacked bitpacked data.
    bitpacked_output: Vec<u32>,
    /// Intermediate buffer for FoR-decoded data.
    for_decoded: Vec<i32>,
    /// Output buffer for batch decompression.
    alp_decoded: Vec<f32>,
    /// Output buffer for pipeline decompression.
    pipeline_output: Vec<f32>,
    /// Output buffer for in-place batch decompression.
    alp_decoded_inplace_batch: Vec<f32>,
    /// Output buffer for in-place pipeline decompression.
    alp_decoded_inplace_pipeline: Vec<f32>,
}

/// Patch information for ALP encoding.
///
/// Some values cannot be accurately represented in ALP encoding and require patches.
pub struct Patches {
    /// Indices of values that need patches.
    indices: Vec<u64>,
    /// Original values at the patch indices.
    values: Vec<f32>,
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Setup Functions
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Set up test data and buffers for benchmarks.
///
/// Creates compressed data using the full pipeline (ALP -> FoR -> Bitpacking)
/// and allocates all necessary buffers for decompression.
fn setup(size: usize) -> (InputData, BenchmarkBuffers) {
    let original = create_random_values(size);
    let (alp_encoded, exponents, patches) = alp_compress(&original);
    let (for_encoded, reference) = for_compress(&alp_encoded);
    let bitpacked = bitpack_10(cast_i32_as_u32(&for_encoded));

    let input_data = InputData {
        bitpacked,
        reference,
        exponents,
        original,
        alp_encoded,
        patches,
    };

    let benchmark_buffers = BenchmarkBuffers {
        bitpacked_output: vec![0u32; size],
        for_decoded: vec![0i32; size],
        alp_decoded: vec![0.0f32; size],
        pipeline_output: vec![0.0f32; size],
        alp_decoded_inplace_batch: vec![0.0f32; size],
        alp_decoded_inplace_pipeline: vec![0.0f32; size],
    };

    (input_data, benchmark_buffers)
}

/// Create random float values for testing.
///
/// Generates values in the range [0.0, 10.24) which compress well with ALP.
fn create_random_values(len: usize) -> Vec<f32> {
    assert!(len.is_multiple_of(N));

    let mut rng = rand::rng();
    (0..len)
        .map(|_| rng.random_range(0..1024))
        .map(|x| x as f32 / 100.0)
        .collect()
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Batch Decompression Functions
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Batch decompression with separate buffers for each stage.
///
/// This is the straightforward approach with clear separation between stages.
fn decompress_batch(
    bitpacked: &[u32],
    reference: i32,
    exponents: Exponents,
    bitpacked_output: &mut [u32],
    for_decoded: &mut [i32],
    alp_decoded: &mut [f32],
) {
    unpack_10(bitpacked, bitpacked_output);
    for_decompress(cast_u32_as_i32(bitpacked_output), reference, for_decoded);
    alp_decompress(for_decoded, exponents, alp_decoded);
}

/// In-place batch decompression that reuses a single buffer for all stages.
///
/// Minimizes memory usage by reinterpreting the same buffer for different stages.
fn decompress_in_place_batch(
    bitpacked: &[u32],
    reference: i32,
    exponents: Exponents,
    output: &mut [f32],
) {
    // Reinterpret the output buffer as u32 for the first stage.
    // SAFETY: f32 and u32 have the same size (4 bytes) and alignment.
    let buffer_u32 =
        unsafe { std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut u32, output.len()) };

    // Stage 1: Unpack bitpacked data into buffer (as u32).
    unpack_10(bitpacked, buffer_u32);

    // Stage 2: FoR decode in-place (reinterpret as i32).
    let buffer_i32 = cast_u32_as_i32_mut(buffer_u32);
    for_decompress_inplace(buffer_i32, reference);

    // Stage 3: ALP decode in-place (transmute i32 → f32).
    f32::decode_slice_inplace(buffer_i32, exponents);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Pipeline Decompression Functions
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Pipeline decompression that processes data chunk by chunk.
///
/// Processes data in chunks to improve cache locality while using separate buffers
/// for each stage to maintain clarity.
fn decompress_pipeline(
    bitpacked: &[u32],
    reference: i32,
    exponents: Exponents,
    unpack_buffer: &mut [u32],
    for_buffer: &mut [i32],
    output: &mut [f32],
) {
    debug_assert!(bitpacked.len().is_multiple_of(S));
    debug_assert_eq!(output.len(), bitpacked.len() * T / W);
    debug_assert!(unpack_buffer.len() >= N);
    debug_assert!(for_buffer.len() >= N);

    // Use only the first N elements of the pre-allocated buffers.
    let unpack_chunk = &mut unpack_buffer[..N];
    let for_chunk = &mut for_buffer[..N];

    let mut input_offset = 0;
    let mut output_offset = 0;

    // Process each 1024-element chunk.
    while input_offset < bitpacked.len() {
        // Stage 1: Bitpacking decompression.
        // SAFETY: Bounds are verified by debug_assert and loop conditions.
        unsafe {
            let input = bitpacked.get_unchecked(input_offset..input_offset + S);
            BitPacking::unchecked_unpack(W, input, unpack_chunk);
        }

        // Stage 2: FoR decompression.
        // SAFETY: Buffer sizes are verified to be N.
        unsafe {
            for i in 0..N {
                let unpacked = *unpack_chunk.get_unchecked(i) as i32;
                *for_chunk.get_unchecked_mut(i) = unpacked.wrapping_add(reference);
            }
        }

        // Stage 3: ALP decompression.
        // SAFETY: Buffer sizes and output bounds are verified.
        unsafe {
            let output_chunk = output.get_unchecked_mut(output_offset..output_offset + N);
            for i in 0..N {
                let for_decoded = *for_chunk.get_unchecked(i);
                *output_chunk.get_unchecked_mut(i) = f32::decode_single(for_decoded, exponents);
            }
        }

        input_offset += S;
        output_offset += N;
    }
}

/// Pipeline decompression that processes data chunk by chunk with an extra copy.
///
/// This version intentionally adds an extra copy step to measure the performance impact.
/// It writes to an intermediate ALP buffer before copying to the final output.
fn decompress_pipeline_extra_copy(
    bitpacked: &[u32],
    reference: i32,
    exponents: Exponents,
    unpack_buffer: &mut [u32],
    for_buffer: &mut [i32],
    alp_buffer: &mut [f32],
    output: &mut [f32],
) {
    debug_assert!(bitpacked.len().is_multiple_of(S));
    debug_assert_eq!(output.len(), bitpacked.len() * T / W);
    debug_assert!(unpack_buffer.len() >= N);
    debug_assert!(for_buffer.len() >= N);
    debug_assert!(alp_buffer.len() >= N);

    // Use only the first N elements of the pre-allocated buffers.
    let unpack_chunk = &mut unpack_buffer[..N];
    let for_chunk = &mut for_buffer[..N];
    let alp_chunk = &mut alp_buffer[..N];

    let mut input_offset = 0;
    let mut output_offset = 0;

    // Process each 1024-element chunk.
    while input_offset < bitpacked.len() {
        // Stage 1: Bitpacking decompression.
        // SAFETY: Bounds are verified by debug_assert and loop conditions.
        unsafe {
            let input = bitpacked.get_unchecked(input_offset..input_offset + S);
            BitPacking::unchecked_unpack(W, input, unpack_chunk);
        }

        // Stage 2: FoR decompression.
        // SAFETY: Buffer sizes are verified to be N.
        unsafe {
            for i in 0..N {
                let unpacked = *unpack_chunk.get_unchecked(i) as i32;
                *for_chunk.get_unchecked_mut(i) = unpacked.wrapping_add(reference);
            }
        }

        // Stage 3: ALP decompression into intermediate buffer.
        // SAFETY: Buffer sizes are verified to be N.
        unsafe {
            for i in 0..N {
                let for_decoded = *for_chunk.get_unchecked(i);
                *alp_chunk.get_unchecked_mut(i) = f32::decode_single(for_decoded, exponents);
            }
        }

        // Stage 4: Copy from intermediate ALP buffer to final output.
        // SAFETY: Buffer sizes are verified to be N.
        let output_chunk = unsafe { output.get_unchecked_mut(output_offset..output_offset + N) };
        output_chunk.copy_from_slice(alp_chunk);

        input_offset += S;
        output_offset += N;
    }
}

/// In-place pipeline decompression that processes data chunk by chunk directly in the output buffer.
///
/// Combines the benefits of pipeline processing with minimal memory usage.
fn decompress_in_place_pipeline(
    bitpacked: &[u32],
    reference: i32,
    exponents: Exponents,
    output: &mut [f32],
) {
    debug_assert!(bitpacked.len().is_multiple_of(S));
    debug_assert_eq!(output.len(), bitpacked.len() * T / W);

    let mut input_offset = 0;
    let mut output_offset = 0;

    while input_offset < bitpacked.len() {
        // Get the current chunk of the output buffer to work on.
        // SAFETY: Output bounds are verified by debug_assert.
        let output_chunk = unsafe { output.get_unchecked_mut(output_offset..output_offset + N) };

        // Reinterpret the output chunk as u32 for unpacking.
        // SAFETY: f32 and u32 have the same size and alignment.
        let chunk_u32 =
            unsafe { std::slice::from_raw_parts_mut(output_chunk.as_mut_ptr() as *mut u32, N) };

        // Stage 1: Unpack directly into the output buffer (as u32).
        // SAFETY: Input bounds are verified.
        unsafe {
            let input = bitpacked.get_unchecked(input_offset..input_offset + S);
            BitPacking::unchecked_unpack(W, input, chunk_u32);
        }

        // Stage 2: FoR decompression in-place.
        let chunk_i32 = cast_u32_as_i32_mut(chunk_u32);
        unsafe {
            for i in 0..N {
                *chunk_i32.get_unchecked_mut(i) =
                    chunk_i32.get_unchecked(i).wrapping_add(reference);
            }
        }

        // Stage 3: ALP decompression.
        // SAFETY: Buffer sizes are verified.
        unsafe {
            for i in 0..N {
                let for_decoded = *chunk_i32.get_unchecked(i);
                *output_chunk.get_unchecked_mut(i) = f32::decode_single(for_decoded, exponents);
            }
        }

        input_offset += S;
        output_offset += N;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Bitpacking Functions
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Pack 32-bit values into 10-bit bitpacked representation.
fn bitpack_10(values: &[u32]) -> Vec<u32> {
    let len = values.len();
    debug_assert!(len.is_multiple_of(N));

    let mut bitpacked = Vec::with_capacity(len * W / T);
    // SAFETY: We're setting the length to the exact capacity we just allocated.
    // The memory will be immediately initialized by BitPacking::unchecked_pack.
    unsafe { bitpacked.set_len(len * W / T) };

    let mut input_offset = 0;
    let mut output_offset = 0;

    while input_offset < len {
        // SAFETY: Loop bounds ensure we have N elements available.
        unsafe {
            let input = values.get_unchecked(input_offset..input_offset + N);
            let output = bitpacked.get_unchecked_mut(output_offset..output_offset + S);
            BitPacking::unchecked_pack(W, input, output);
        }

        input_offset += N;
        output_offset += S;
    }

    bitpacked
}

/// Unpack 10-bit bitpacked values into 32-bit representation.
fn unpack_10(bitpacked: &[u32], unpacked: &mut [u32]) {
    debug_assert!(bitpacked.len().is_multiple_of(S));
    let len = bitpacked.len() * T / W;
    debug_assert_eq!(unpacked.len(), len);

    let mut input_offset = 0;
    let mut output_offset = 0;

    while output_offset < len {
        // SAFETY: Loop bounds and assertions ensure valid indices.
        unsafe {
            let input = bitpacked.get_unchecked(input_offset..input_offset + S);
            let output = unpacked.get_unchecked_mut(output_offset..output_offset + N);
            BitPacking::unchecked_unpack(W, input, output);
        }

        input_offset += S;
        output_offset += N;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// FoR (Frame of Reference) Functions
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Compress values using Frame of Reference encoding.
///
/// Subtracts the minimum value from all values to reduce the range.
fn for_compress(values: &[i32]) -> (Vec<i32>, i32) {
    let min = values.iter().min().copied().unwrap();
    (values.iter().map(|x| x.wrapping_sub(min)).collect(), min)
}

/// Decompress Frame of Reference encoded values.
///
/// Adds the reference value back to restore original values.
fn for_decompress(for_values: &[i32], reference: i32, output: &mut [i32]) {
    debug_assert_eq!(for_values.len(), output.len());
    let len = for_values.len();

    // SAFETY: Length equality is verified by debug_assert.
    unsafe {
        for i in 0..len {
            *output.get_unchecked_mut(i) = for_values.get_unchecked(i).wrapping_add(reference);
        }
    }
}

/// In-place Frame of Reference decompression.
///
/// Modifies values in-place by adding the reference value.
fn for_decompress_inplace(values: &mut [i32], reference: i32) {
    for i in 0..values.len() {
        values[i] = values[i].wrapping_add(reference);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// ALP (Adaptive Lossless floating-Point) Functions
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Compress floating-point values using ALP encoding.
///
/// Returns the encoded integers, exponents, and patches for values that cannot be accurately encoded.
fn alp_compress(values: &[f32]) -> (Vec<i32>, Exponents, Patches) {
    let (exponents, encoded, patch_indices, patch_values, _) = f32::encode(values, None);

    let indices = patch_indices.into_iter().collect();
    let values = patch_values.into_iter().collect();

    let alp_vec: Vec<i32> = encoded.into_iter().collect();
    (alp_vec, exponents, Patches { indices, values })
}

/// Decompress ALP-encoded values back to floating-point.
fn alp_decompress(encoded: &[i32], exponents: Exponents, output: &mut [f32]) {
    f32::decode_into(encoded, exponents, output)
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Utility Functions
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Cast i32 slice to u32 slice.
///
/// This is safe because i32 and u32 have the same size and alignment.
fn cast_i32_as_u32(slice: &[i32]) -> &[u32] {
    // SAFETY: i32 and u32 have the same size and alignment, so this transmute is safe.
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u32, slice.len()) }
}

/// Cast u32 slice to i32 slice.
///
/// This is safe because u32 and i32 have the same size and alignment.
fn cast_u32_as_i32(slice: &[u32]) -> &[i32] {
    // SAFETY: i32 and u32 have the same size and alignment, so this transmute is safe.
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const i32, slice.len()) }
}

/// Cast mutable u32 slice to mutable i32 slice.
///
/// This is safe because u32 and i32 have the same size and alignment.
fn cast_u32_as_i32_mut(slice: &mut [u32]) -> &mut [i32] {
    // SAFETY: i32 and u32 have the same size and alignment, so this transmute is safe.
    unsafe { std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut i32, slice.len()) }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Verification Functions
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Verify that FoR and ALP decompression produced correct results.
///
/// Checks that FoR decoding matches expected values and ALP decoding (with patches) matches originals.
fn verify(
    function_name: &str,
    for_decoded: &[i32],
    alp_decoded: &[f32],
    alp_encoded: &[i32],
    original: &[f32],
    patches: &Patches,
) {
    // Verify FoR decompression.
    for i in 0..for_decoded.len() {
        assert_eq!(
            for_decoded[i], alp_encoded[i],
            "{}: FoR decode mismatch at index {}: decoded={}, expected={}",
            function_name, i, for_decoded[i], alp_encoded[i]
        );
    }

    // Verify ALP decompression.
    // ALP may have patches for values that couldn't be accurately encoded.
    for i in 0..alp_decoded.len() {
        if let Some(patch_idx) = patches.indices.iter().position(|&idx| idx == i as u64) {
            // This index has a patch - verify the patch value matches the original.
            assert_eq!(
                patches.values[patch_idx], original[i],
                "{}: Patch value mismatch at index {}: patch={}, expected={}",
                function_name, i, patches.values[patch_idx], original[i]
            );
        } else {
            // For non-patched values, verify ALP decoding matches the original.
            assert_eq!(
                alp_decoded[i], original[i],
                "{}: ALP decode mismatch at index {}: decoded={}, expected={}",
                function_name, i, alp_decoded[i], original[i]
            );
        }
    }
}

/// Compare outputs from different decompression functions.
///
/// Ensures that all decompression strategies produce identical results.
fn compare_outputs(function_name: &str, expected: &[f32], actual: &[f32]) {
    assert_eq!(
        expected.len(),
        actual.len(),
        "{}: Output length mismatch: expected={}, actual={}",
        function_name,
        expected.len(),
        actual.len()
    );

    for i in 0..expected.len() {
        assert_eq!(
            expected[i], actual[i],
            "{}: Output mismatch at index {}: expected={}, actual={}",
            function_name, i, expected[i], actual[i]
        );
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Benchmarks
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Performance benchmarks for decompression strategies.
#[divan::bench_group(min_time = Duration::from_secs(1))]
mod decompress_benchmarks {
    use super::*;

    #[divan::bench(consts = BENCHMARK_SIZES)]
    fn batch<const SIZE: usize>(bencher: Bencher) {
        let (input_data, mut buffers) = setup(SIZE);

        bencher.bench_local(|| {
            decompress_batch(
                &input_data.bitpacked,
                input_data.reference,
                input_data.exponents,
                &mut buffers.bitpacked_output,
                &mut buffers.for_decoded,
                &mut buffers.alp_decoded,
            );
        });
    }

    #[divan::bench(consts = BENCHMARK_SIZES)]
    fn pipeline<const SIZE: usize>(bencher: Bencher) {
        let (input_data, mut buffers) = setup(SIZE);

        bencher.bench_local(|| {
            decompress_pipeline(
                &input_data.bitpacked,
                input_data.reference,
                input_data.exponents,
                &mut buffers.bitpacked_output,
                &mut buffers.for_decoded,
                &mut buffers.pipeline_output,
            );
        });
    }

    #[divan::bench(consts = BENCHMARK_SIZES)]
    fn pipeline_extra_copy<const SIZE: usize>(bencher: Bencher) {
        let (input_data, mut buffers) = setup(SIZE);

        bencher.bench_local(|| {
            decompress_pipeline_extra_copy(
                &input_data.bitpacked,
                input_data.reference,
                input_data.exponents,
                &mut buffers.bitpacked_output,
                &mut buffers.for_decoded,
                &mut buffers.alp_decoded,
                &mut buffers.pipeline_output,
            );
        });
    }

    #[divan::bench(consts = BENCHMARK_SIZES)]
    fn in_place_batch<const SIZE: usize>(bencher: Bencher) {
        let (input_data, mut buffers) = setup(SIZE);

        bencher.bench_local(|| {
            decompress_in_place_batch(
                &input_data.bitpacked,
                input_data.reference,
                input_data.exponents,
                &mut buffers.alp_decoded_inplace_batch,
            );
        });
    }

    #[divan::bench(consts = BENCHMARK_SIZES)]
    fn in_place_pipeline<const SIZE: usize>(bencher: Bencher) {
        let (input_data, mut buffers) = setup(SIZE);

        bencher.bench_local(|| {
            decompress_in_place_pipeline(
                &input_data.bitpacked,
                input_data.reference,
                input_data.exponents,
                &mut buffers.alp_decoded_inplace_pipeline,
            );
        });
    }
}

/// Correctness verification benchmarks.
///
/// These benchmarks verify that all decompression strategies produce identical
/// and correct results. They run with smaller sizes for quick verification.
#[divan::bench_group(min_time = Duration::from_millis(100))]
mod correctness_verification {
    use super::*;

    #[divan::bench(consts = VERIFICATION_SIZES)]
    fn verify_all_methods<const SIZE: usize>(bencher: Bencher) {
        bencher.bench_local(|| {
            let (input_data, mut buffers) = setup(SIZE);

            // Run batch decompression (our reference implementation).
            decompress_batch(
                &input_data.bitpacked,
                input_data.reference,
                input_data.exponents,
                &mut buffers.bitpacked_output,
                &mut buffers.for_decoded,
                &mut buffers.alp_decoded,
            );

            // Verify batch decompression is correct.
            verify(
                "batch",
                &buffers.for_decoded,
                &buffers.alp_decoded,
                &input_data.alp_encoded,
                &input_data.original,
                &input_data.patches,
            );

            // Run pipeline decompression and compare with batch.
            decompress_pipeline(
                &input_data.bitpacked,
                input_data.reference,
                input_data.exponents,
                &mut buffers.bitpacked_output,
                &mut buffers.for_decoded,
                &mut buffers.pipeline_output,
            );
            compare_outputs("pipeline", &buffers.alp_decoded, &buffers.pipeline_output);

            // Run in-place batch decompression and compare with batch.
            decompress_in_place_batch(
                &input_data.bitpacked,
                input_data.reference,
                input_data.exponents,
                &mut buffers.alp_decoded_inplace_batch,
            );
            compare_outputs(
                "in_place_batch",
                &buffers.alp_decoded,
                &buffers.alp_decoded_inplace_batch,
            );

            // Run in-place pipeline decompression and compare with batch.
            decompress_in_place_pipeline(
                &input_data.bitpacked,
                input_data.reference,
                input_data.exponents,
                &mut buffers.alp_decoded_inplace_pipeline,
            );
            compare_outputs(
                "in_place_pipeline",
                &buffers.alp_decoded,
                &buffers.alp_decoded_inplace_pipeline,
            );
        });
    }
}
