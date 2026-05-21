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
//! Benchmarks were run on an AMD Ryzen 9 7950X (Zen 4) with 64GB RAM. The 7950X has 32KB L1 data
//! cache per core and 1MB L2 cache per core.
//!
//! ## Benchmark Results with Filtering
//!
//! Testing across multiple data sizes shows consistent performance patterns:
//!
//! | Size (elements) | Batch (µs) | Pipeline (µs) | Pipeline+Copy (µs) | In-Place Batch (µs) | In-Place Pipeline (µs) |
//! | --------------- | ---------- | ------------- | ------------------ | ------------------- | ---------------------- |
//! | 1,024           | 0.229      | 0.215         | 0.215              | 0.219               | 0.215                  |
//! | 16,384          | 3.708      | 3.535         | 3.386              | 3.559               | 3.375                  |
//! | 65,536          | 15.30      | 13.42         | 13.62              | 14.21               | 13.38                  |
//! | 73,728          | 18.02      | 15.05         | 15.08              | 15.94               | 15.04                  |
//! | 86,016          | 20.97      | 17.65         | 17.55              | 19.32               | 17.54                  |
//! | 100,352         | 25.12      | 21.42         | 20.48              | 22.03               | 20.48                  |
//!
//! ## Cache Locality Advantage
//!
//! The pipeline approach processes data in 1,024-element chunks (4KB). Each chunk fits entirely
//! in the 32KB L1 data cache on Zen 4. L1 cache provides 4-cycle latency while L2 has 14-cycle
//! latency [1].
//!
//! Processing all three stages while data resides in L1 eliminates cache misses. The batch
//! approach must reload the entire dataset from L2/L3 for each stage. For 100K elements (400KB),
//! batch processing performs three full passes through memory. Pipeline processing performs the
//! same memory reads but maintains temporal locality within each 4KB chunk.
//!
//! Measured memory bandwidth utilization shows the advantage. Pipeline processing achieves
//! 19.8 GB/s effective bandwidth versus 16.5 GB/s for batch processing at 100K elements. The
//! 20% bandwidth improvement comes from keeping data in L1 throughout the transformation chain.
//!
//! ## Extra Copy Performance Advantage
//!
//! The pipeline with extra copy outperforms the regular pipeline despite doing more work. This
//! counterintuitive result comes from better cache utilization during filtering.
//!
//! Regular pipeline writes ALP output directly to the final buffer then filters in place:
//!
//! ```text
//! ALP decode -> Write to output[offset] -> Filter output[offset] in place
//! ```
//!
//! Pipeline with extra copy uses an intermediate buffer:
//!
//! ```text
//! ALP decode -> Write to temp[0:1024] -> Filter temp[0:1024] -> Copy kept elements to output
//! ```
//!
//! The intermediate buffer (4KB) stays hot in L1 cache during filtering. Regular pipeline may
//! evict output buffer data from L1 as it advances through chunks. When filtering accesses the
//! output buffer, some data has moved to L2.
//!
//! The extra copy only moves kept elements after filtering. With the 0xDEADBEEF mask keeping
//! about 50% of data, this reduces memory traffic. The cost of copying 512 elements from L1
//! is less than the penalty of filtering data that has been evicted to L2.
//!
//! ## In-Place Performance Penalty with Filtering
//!
//! **Note that this section is only relevant to ARM processors, as we saw performance degradation
//! for in-place processing only on ARM and not on x86.** This section is an archive from previous
//! benchmarks we ran on an Apple M4 Max processor.
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
//! [1] <https://chipsandcheese.com/2022/11/08/amds-zen-4-part-2-memory-subsystem-and-conclusion/>
//! [2] <https://chipsandcheese.com/p/cortex-x2-arm-aims-high>

#![expect(clippy::unwrap_used, clippy::uninit_vec)]

use divan::Bencher;
use fastlanes::BitPacking;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_alp::ALPFloat;
use vortex_alp::Exponents;
use vortex_error::vortex_panic;

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
const BENCHMARK_SIZES: [usize; 6] = [
    1024,   // 1K
    8192,   // 8K
    16384,  // 16K
    65536,  // 64K
    86016,  // 84K
    100352, // 98K
];

/// Sizes to test for correctness verification.
const VERIFICATION_SIZES: [usize; 2] = [
    1024,  // 1K - minimum size
    16384, // 16K - medium size
];

/// The number of samples (each will run 100 times).
const SAMPLE_SIZE: u32 = 4096;

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

    let mut rng = StdRng::seed_from_u64(0);
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

    // Cast f32 output to u32 for filtering.
    // SAFETY: f32 and u32 have the same size and alignment.
    let alp_as_u32 = unsafe {
        std::slice::from_raw_parts_mut(alp_decoded.as_mut_ptr() as *mut u32, alp_decoded.len())
    };
    let _kept = filter_scalar(alp_as_u32);
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

    // Cast f32 output to u32 for filtering.
    // SAFETY: f32 and u32 have the same size and alignment.
    let output_as_u32 =
        unsafe { std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut u32, output.len()) };
    let _kept = filter_scalar(output_as_u32);
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
    let mut output_write_offset = 0; // Track where to write filtered output.

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

        // Stage 3: ALP decompression directly into output buffer.
        // We decompress into the output buffer starting at output_write_offset.
        // SAFETY: Buffer sizes and output bounds are verified.
        unsafe {
            let output_chunk =
                output.get_unchecked_mut(output_write_offset..output_write_offset + N);
            for i in 0..N {
                let for_decoded = *for_chunk.get_unchecked(i);
                *output_chunk.get_unchecked_mut(i) = f32::decode_single(for_decoded, exponents);
            }
        }

        // Stage 4: Filter the chunk in the output buffer.
        // Note: filter_scalar modifies the data in-place, compacting it.
        let output_chunk =
            unsafe { output.get_unchecked_mut(output_write_offset..output_write_offset + N) };
        let kept_count = filter_scalar(output_chunk);

        // The filtered data is now compacted at output_write_offset.
        output_write_offset += kept_count;
        input_offset += S;
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
    let mut output_write_offset = 0; // Track where to write filtered output.

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

        // Stage 4: Filter the intermediate ALP buffer.
        let kept_count = filter_scalar(alp_chunk);

        // Stage 5: Copy filtered data from intermediate ALP buffer to final output.
        // SAFETY: Buffer sizes are verified and kept_count <= N.
        let output_chunk = unsafe {
            output.get_unchecked_mut(output_write_offset..output_write_offset + kept_count)
        };
        output_chunk.copy_from_slice(&alp_chunk[..kept_count]);

        output_write_offset += kept_count;
        input_offset += S;
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
    let mut output_write_offset = 0; // Track where to write filtered output.

    while input_offset < bitpacked.len() {
        // Get the current chunk of the output buffer to work on.
        // SAFETY: Output bounds are verified by debug_assert.
        let output_chunk =
            unsafe { output.get_unchecked_mut(output_write_offset..output_write_offset + N) };

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

        // Stage 4: Filter the chunk in-place.
        let kept_count = filter_scalar(output_chunk);

        output_write_offset += kept_count;
        input_offset += S;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Filter Functions
////////////////////////////////////////////////////////////////////////////////////////////////////

// Hardcoded mask for now.

fn filter_scalar<T: Copy>(data: &mut [T]) -> usize {
    let len = data.len();
    assert!(len.is_multiple_of(usize::BITS as usize));

    let iters = len / 64;

    let mut read_ptr = data.as_ptr();
    let mut write_ptr = data.as_mut_ptr();
    let initial_write_ptr = write_ptr;

    for _ in 0..iters {
        let mut word: usize = std::hint::black_box(0xDEADBEEF);

        while word != 0 {
            let bit_pos = word.trailing_zeros();
            word &= word - 1; // Clear the bit at `bit_pos`.
            let span = word.trailing_ones();
            word >>= span;

            unsafe {
                std::ptr::copy(read_ptr.add(bit_pos as usize), write_ptr, span as usize);
                write_ptr = write_ptr.add(span as usize);
            }
        }

        unsafe { read_ptr = read_ptr.add(usize::BITS as usize) };
    }

    // Return the number of elements kept.
    unsafe { write_ptr.offset_from(initial_write_ptr) as usize }
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
    unsafe { std::slice::from_raw_parts(slice.as_ptr().cast(), slice.len()) }
}

/// Cast u32 slice to i32 slice.
///
/// This is safe because u32 and i32 have the same size and alignment.
fn cast_u32_as_i32(slice: &[u32]) -> &[i32] {
    // SAFETY: i32 and u32 have the same size and alignment, so this transmute is safe.
    unsafe { std::slice::from_raw_parts(slice.as_ptr().cast(), slice.len()) }
}

/// Cast mutable u32 slice to mutable i32 slice.
///
/// This is safe because u32 and i32 have the same size and alignment.
fn cast_u32_as_i32_mut(slice: &mut [u32]) -> &mut [i32] {
    // SAFETY: i32 and u32 have the same size and alignment, so this transmute is safe.
    unsafe { std::slice::from_raw_parts_mut(slice.as_mut_ptr().cast(), slice.len()) }
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
/// Filtering should produce the same results whether applied chunk-by-chunk
/// or all at once. Both expected and actual should already be filtered.
fn compare_outputs(function_name: &str, expected: &[f32], actual: &[f32], expected_len: usize) {
    // Both buffers should have the same allocated size.
    assert_eq!(actual.len(), expected.len());

    // Only compare the filtered portion of the data.
    let expected_slice = &expected[..expected_len];
    let actual_slice = &actual[..expected_len];

    for i in 0..expected_len {
        if expected_slice[i] != actual_slice[i] {
            // Debug output to understand the mismatch.
            eprintln!(
                "Mismatch at index {}: expected={}, actual={}",
                i, expected_slice[i], actual_slice[i]
            );
            if i > 0 {
                eprintln!(
                    "  Previous values: expected[{}]={}, actual[{}]={}",
                    i - 1,
                    expected_slice[i - 1],
                    i - 1,
                    actual_slice[i - 1]
                );
            }
            if i + 1 < expected_len {
                eprintln!(
                    "  Next values: expected[{}]={}, actual[{}]={}",
                    i + 1,
                    expected_slice[i + 1],
                    i + 1,
                    actual_slice[i + 1]
                );
            }
            vortex_panic!(
                "{}: Output mismatch at index {}: expected={}, actual={}",
                function_name,
                i,
                expected_slice[i],
                actual_slice[i]
            );
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Benchmarks
////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(not(codspeed))]
#[divan::bench(consts = BENCHMARK_SIZES, sample_size = SAMPLE_SIZE)]
fn batch<const SIZE: usize>(bencher: Bencher) {
    let (input_data, mut buffers) = setup(SIZE);

    bencher.bench_local(|| {
        let input_data = std::hint::black_box(&input_data);
        let buffers = std::hint::black_box(&mut buffers);

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

#[cfg(not(codspeed))]
#[divan::bench(consts = BENCHMARK_SIZES, sample_size = SAMPLE_SIZE)]
fn pipeline<const SIZE: usize>(bencher: Bencher) {
    let (input_data, mut buffers) = setup(SIZE);
    bencher.bench_local(|| {
        let input_data = std::hint::black_box(&input_data);
        let buffers = std::hint::black_box(&mut buffers);

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

#[cfg(not(codspeed))]
#[divan::bench(consts = BENCHMARK_SIZES, sample_size = SAMPLE_SIZE)]
fn pipeline_extra_copy<const SIZE: usize>(bencher: Bencher) {
    let (input_data, mut buffers) = setup(SIZE);
    bencher.bench_local(|| {
        let input_data = std::hint::black_box(&input_data);
        let buffers = std::hint::black_box(&mut buffers);

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

#[cfg(not(codspeed))]
#[divan::bench(consts = BENCHMARK_SIZES, sample_size = SAMPLE_SIZE)]
fn in_place_batch<const SIZE: usize>(bencher: Bencher) {
    let (input_data, mut buffers) = setup(SIZE);
    bencher.bench_local(|| {
        let input_data = std::hint::black_box(&input_data);
        let buffers = std::hint::black_box(&mut buffers);

        decompress_in_place_batch(
            &input_data.bitpacked,
            input_data.reference,
            input_data.exponents,
            &mut buffers.alp_decoded_inplace_batch,
        );
    });
}

#[cfg(not(codspeed))]
#[divan::bench(consts = BENCHMARK_SIZES, sample_size = SAMPLE_SIZE)]
fn in_place_pipeline<const SIZE: usize>(bencher: Bencher) {
    let (input_data, mut buffers) = setup(SIZE);
    bencher.bench_local(|| {
        let input_data = std::hint::black_box(&input_data);
        let buffers = std::hint::black_box(&mut buffers);

        decompress_in_place_pipeline(
            &input_data.bitpacked,
            input_data.reference,
            input_data.exponents,
            &mut buffers.alp_decoded_inplace_pipeline,
        );
    });
}

// Correctness verification benchmarks.
//
// These benchmarks verify that all decompression strategies produce identical
// and correct results. They run with smaller sizes for quick verification.

#[cfg(not(codspeed))]
#[divan::bench(consts = VERIFICATION_SIZES)]
fn verify_all_methods<const SIZE: usize>(bencher: Bencher) {
    bencher
        .with_inputs(|| setup(SIZE))
        .bench_refs(|(input_data, buffers)| {
            // Create a filtered version of the original values for comparison.
            // SAFETY: f32 and u32 have the same size and alignment.
            let original_as_u32 = unsafe {
                std::slice::from_raw_parts_mut(
                    input_data.original.as_mut_ptr() as *mut u32,
                    input_data.original.len(),
                )
            };
            let expected_filtered_len = filter_scalar(original_as_u32);

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
            // Note: for_decoded is not filtered, but alp_decoded is filtered.
            verify(
                "batch",
                &buffers.for_decoded,
                &buffers.alp_decoded,
                &input_data.alp_encoded,
                &input_data.original, // This is now filtered.
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
            compare_outputs(
                "pipeline",
                &buffers.alp_decoded,
                &buffers.pipeline_output,
                expected_filtered_len,
            );

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
                expected_filtered_len,
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
                expected_filtered_len,
            );
        });
}
