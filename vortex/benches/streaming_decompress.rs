// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Streaming chunked decompression (`decompress_chunks`) against level-wise execution and against
//! hand-written fused kernels, for the encoding trees that stream.
//!
//! Every tree is measured six ways, `{materialize, sum} x {streaming, levelwise, fused}`:
//!
//! - `streaming`: the vtable path. `materialize` streams each chunk into the canonical builder
//!   (`execute_via_chunks`), and `sum` folds each chunk straight out of L1.
//! - `levelwise`: the executor with its streaming shortcut disabled, which materializes one buffer
//!   per encoding level. `sum` then reads the result back.
//! - `fused`: a kernel written by hand for exactly this tree, monomorphized end to end with no
//!   dynamic dispatch. It decodes each 1024-row block through every level and writes or folds it
//!   once. This is the bound the generic mechanism is measured against.
//!
//! Trees hold 64Ki rows, the size of a typical scan split, with no nulls and no slicing so the
//! fused kernels can stay minimal. The real scan stacks at the end, found in TPC-H, have no fused
//! kernel and compare streaming with level-wise execution only.

use std::f64::consts::PI;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ops::Range;
use std::sync::LazyLock;

use divan::Bencher;
use fastlanes::BitPacking;
use fastlanes::Delta as FastLanesDelta;
use fastlanes::FoR as FastLanesFoR;
use fastlanes::RLE as FastLanesRLE;
use fastlanes::Transpose;
use mimalloc::MiMalloc;
use vortex::VortexSessionDefault;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::DecimalArray;
use vortex::array::arrays::Dict;
use vortex::array::arrays::DictArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::SliceArray;
use vortex::array::arrays::dict::DictArraySlotsExt;
use vortex::array::builtins::ArrayBuiltins;
use vortex::array::chunk_iter::ChunkMut;
use vortex::array::chunk_iter::ChunkSink;
use vortex::array::chunk_iter::ChunkValue;
use vortex::array::chunk_iter::execute_via_chunks;
use vortex::array::chunk_iter::set_chunked_execute_enabled;
use vortex::array::dtype::DType;
use vortex::array::dtype::DecimalDType;
use vortex::array::dtype::NativeDecimalType;
use vortex::array::dtype::NativePType;
use vortex::array::dtype::Nullability;
use vortex::array::dtype::PType;
use vortex::array::patches::Patches;
use vortex::array::scalar::Scalar;
use vortex::array::scalar_fn::fns::cast::Cast;
use vortex::array::validity::Validity;
use vortex::buffer::Buffer;
use vortex::encodings::alp::ALP;
use vortex::encodings::alp::ALPArrayExt;
use vortex::encodings::alp::ALPArraySlotsExt;
use vortex::encodings::alp::ALPFloat;
use vortex::encodings::alp::Exponents;
use vortex::encodings::alp::alp_encode;
use vortex::encodings::decimal_byte_parts::DecimalByteParts;
use vortex::encodings::decimal_byte_parts::DecimalBytePartsArraySlotsExt;
use vortex::encodings::fastlanes::BitPacked;
use vortex::encodings::fastlanes::BitPackedArrayExt;
use vortex::encodings::fastlanes::Delta;
use vortex::encodings::fastlanes::DeltaArraySlotsExt;
use vortex::encodings::fastlanes::FoR;
use vortex::encodings::fastlanes::FoRArrayExt;
use vortex::encodings::fastlanes::FoRArraySlotsExt;
use vortex::encodings::fastlanes::RLE;
use vortex::encodings::fastlanes::RLEArraySlotsExt;
use vortex::encodings::fastlanes::RLEData;
use vortex::encodings::fastlanes::bitpack_compress::bitpack_encode;
use vortex::encodings::runend::RunEnd;
use vortex::encodings::runend::RunEndArraySlotsExt;
use vortex::encodings::sparse::Sparse;
use vortex::encodings::sparse::SparseExt;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex_array::ExecutionCtx;
use vortex_array::VortexSessionExecute;
use vortex_mask::Mask;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    check_fused_kernels();
    // The `levelwise` variants measure the executor without its streaming shortcut. The
    // `streaming` variants call the streaming entry points directly, so this does not affect them.
    set_chunked_execute_enabled(false);
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::default);

const LEN_U32: u32 = 64 * 1024;
const LEN: usize = LEN_U32 as usize;
const BLOCK: usize = 1024;

fn ctx() -> ExecutionCtx {
    SESSION.create_execution_ctx()
}

fn primitive(array: &ArrayRef) -> PrimitiveArray {
    array
        .clone()
        .execute::<PrimitiveArray>(&mut ctx())
        .vortex_expect("execute")
}

// ---------------------------------------------------------------------------------------------
// Shared measurement
// ---------------------------------------------------------------------------------------------

/// Folds values into a checksum, so every variant reads every value once.
trait Fold: ChunkValue {
    fn fold(acc: u64, values: &[Self]) -> u64;
}

impl Fold for u32 {
    #[inline]
    fn fold(acc: u64, values: &[Self]) -> u64 {
        values
            .iter()
            .fold(acc, |acc, &v| acc.wrapping_add(u64::from(v)))
    }
}

impl Fold for i64 {
    #[inline]
    fn fold(acc: u64, values: &[Self]) -> u64 {
        acc.wrapping_add(values.iter().fold(0i64, |acc, &v| acc.wrapping_add(v)) as u64)
    }
}

impl Fold for f64 {
    #[inline]
    fn fold(acc: u64, values: &[Self]) -> u64 {
        acc.wrapping_add(values.iter().sum::<f64>().to_bits())
    }
}

impl Fold for i128 {
    #[inline]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the checksum keeps the low bits"
    )]
    fn fold(acc: u64, values: &[Self]) -> u64 {
        acc.wrapping_add(values.iter().fold(0i128, |acc, &v| acc.wrapping_add(v)) as u64)
    }
}

struct SumSink<T> {
    total: u64,
    _values: PhantomData<T>,
}

impl<T: Fold> ChunkSink for SumSink<T> {
    #[inline]
    fn accept(&mut self, chunk: ChunkMut<'_>, _rows: Range<usize>) -> VortexResult<()> {
        // Fold full chunks as whole blocks, with a constant length, as the fused kernels do.
        self.total = match chunk.as_block::<T>() {
            Some(block) => T::fold(self.total, block),
            None => T::fold(self.total, chunk.as_slice::<T>()),
        };
        Ok(())
    }
}

fn materialize_streaming(bencher: Bencher, array: ArrayRef) {
    bencher
        .with_inputs(|| (array.clone(), ctx()))
        .bench_values(|(array, mut ctx)| {
            execute_via_chunks(&array, &mut ctx)
                .vortex_expect("bench")
                .len()
        });
}

fn materialize_levelwise(bencher: Bencher, array: ArrayRef) {
    bencher
        .with_inputs(|| (array.clone(), ctx()))
        .bench_values(|(array, mut ctx)| {
            array
                .execute::<PrimitiveArray>(&mut ctx)
                .vortex_expect("bench")
                .len()
        });
}

fn sum_streaming<T: Fold>(bencher: Bencher, array: ArrayRef) {
    bencher
        .with_inputs(|| (array.clone(), ctx()))
        .bench_values(|(array, mut ctx)| {
            let mut sink = SumSink::<T> {
                total: 0,
                _values: PhantomData,
            };
            array
                .decompress_chunks(&mut ctx, &mut sink)
                .vortex_expect("bench");
            sink.total
        });
}

fn sum_levelwise<T: Fold + NativePType>(bencher: Bencher, array: ArrayRef) {
    bencher
        .with_inputs(|| (array.clone(), ctx()))
        .bench_values(|(array, mut ctx)| {
            let values = array
                .execute::<PrimitiveArray>(&mut ctx)
                .vortex_expect("bench");
            T::fold(0, values.as_slice::<T>())
        });
}

fn materialize_fused<T>(bencher: Bencher, array: ArrayRef, kernel: fn(&ArrayRef) -> Vec<T>) {
    bencher
        .with_inputs(|| array.clone())
        .bench_values(|array| kernel(&array).len());
}

fn sum_fused(bencher: Bencher, array: ArrayRef, kernel: fn(&ArrayRef) -> u64) {
    bencher
        .with_inputs(|| array.clone())
        .bench_values(|array| kernel(&array));
}

/// Generate the six benchmarks for one tree.
macro_rules! tree_benches {
    ($tree:ident : $T:ty, $setup:path, $fused_materialize:path, $fused_sum:path) => {
        mod $tree {
            use super::*;

            #[divan::bench]
            fn materialize_streaming(bencher: Bencher) {
                super::materialize_streaming(bencher, $setup());
            }

            #[divan::bench]
            fn materialize_levelwise(bencher: Bencher) {
                super::materialize_levelwise(bencher, $setup());
            }

            #[divan::bench]
            fn materialize_fused(bencher: Bencher) {
                super::materialize_fused::<$T>(bencher, $setup(), $fused_materialize);
            }

            #[divan::bench]
            fn sum_streaming(bencher: Bencher) {
                super::sum_streaming::<$T>(bencher, $setup());
            }

            #[divan::bench]
            fn sum_levelwise(bencher: Bencher) {
                super::sum_levelwise::<$T>(bencher, $setup());
            }

            #[divan::bench]
            fn sum_fused(bencher: Bencher) {
                super::sum_fused(bencher, $setup(), $fused_sum);
            }
        }
    };
}

// ---------------------------------------------------------------------------------------------
// Fused kernel building blocks
// ---------------------------------------------------------------------------------------------

/// Allocate `LEN` values and let `decode(block_index, block)` write each block in place.
fn decode_blocks<T: Copy>(mut decode: impl FnMut(usize, &mut [T; BLOCK])) -> Vec<T> {
    let mut out = Vec::<T>::with_capacity(LEN);
    let (blocks, _) = out.spare_capacity_mut()[..LEN].as_chunks_mut::<BLOCK>();
    for (index, block) in blocks.iter_mut().enumerate() {
        // SAFETY: `decode` writes every value of the block before anything reads it.
        decode(index, unsafe {
            &mut *(block as *mut [MaybeUninit<T>; BLOCK]).cast::<[T; BLOCK]>()
        });
    }
    // SAFETY: every block was written above.
    unsafe { out.set_len(LEN) };
    out
}

/// Let `decode(block_index, block)` write each block into one scratch block and fold it.
fn fold_blocks<T: Fold>(mut decode: impl FnMut(usize, &mut [T; BLOCK])) -> u64 {
    let mut scratch = [T::default(); BLOCK];
    let mut total = 0;
    for index in 0..LEN / BLOCK {
        decode(index, &mut scratch);
        total = T::fold(total, &scratch);
    }
    total
}

/// Define a tree's fused `materialize` and `sum` kernels from the same per-block decode.
///
/// The `prepare` statements run once per call and bind whatever the decode reads, and `decode`
/// writes block `index` into `block`. Both kernels inline the decode with no dynamic dispatch.
macro_rules! fused_kernels {
    (
        $T:ty, $materialize:ident, $sum:ident,
        |$array:ident| { $($prepare:tt)* },
        |$index:ident, $block:ident| $decode:expr
    ) => {
        fn $materialize($array: &ArrayRef) -> Vec<$T> {
            $($prepare)*
            decode_blocks(|$index, $block: &mut [$T; BLOCK]| $decode)
        }

        fn $sum($array: &ArrayRef) -> u64 {
            $($prepare)*
            fold_blocks(|$index, $block: &mut [$T; BLOCK]| $decode)
        }
    };
}

/// The packed words of block `index` of a bit-packed array.
#[inline]
fn packed_block<T>(packed: &[T], width: usize, index: usize) -> &[T] {
    let per_block = BLOCK * width / (8 * size_of::<T>());
    &packed[index * per_block..][..per_block]
}

/// Patches as row-sorted `(row, value)` pairs.
fn patch_list<T: NativePType>(patches: &Patches) -> Vec<(usize, T)> {
    let values = primitive(patches.values());
    usizes(patches.indices())
        .into_iter()
        .map(|row| row - patches.offset())
        .zip(values.as_slice::<T>().iter().copied())
        .collect()
}

/// Write the patches that fall in block `index` over it, advancing `cursor`.
#[inline]
fn patch_block<T: Copy>(block: &mut [T], index: usize, patches: &[(usize, T)], cursor: &mut usize) {
    let start = index * BLOCK;
    while let Some(&(row, value)) = patches.get(*cursor)
        && row < start + BLOCK
    {
        block[row - start] = value;
        *cursor += 1;
    }
}

/// Read an unsigned integer array, of any width, as `usize`s.
fn usizes(array: &ArrayRef) -> Vec<usize> {
    let array = primitive(
        &array
            .cast(DType::Primitive(PType::U64, Nullability::NonNullable))
            .vortex_expect("cast"),
    );
    array
        .as_slice::<u64>()
        .iter()
        .map(|&v| usize::try_from(v).ok().vortex_expect("usize"))
        .collect()
}

fn constant_reference<T: NativePType>(for_: &ArrayRef) -> T {
    for_.as_::<FoR>()
        .constant_reference()
        .and_then(|reference| reference.as_primitive().as_::<T>())
        .vortex_expect("reference")
}

// ---------------------------------------------------------------------------------------------
// FoR(BitPacked) u32
// ---------------------------------------------------------------------------------------------

fn for_bp_u32() -> ArrayRef {
    let values = PrimitiveArray::from_iter((0..LEN_U32).map(|i| (i * 7) % 1000));
    let bp = bitpack_encode(&values, 10, None, &mut ctx()).vortex_expect("bitpack");
    FoR::try_new(bp.into_array(), Scalar::from(1_000_000u32))
        .vortex_expect("for")
        .into_array()
}

fused_kernels!(
    u32,
    for_bp_u32_materialize,
    for_bp_u32_sum,
    |array| {
        let reference = constant_reference::<u32>(array);
        let for_ = array.as_::<FoR>();
        let bp = for_.encoded().as_::<BitPacked>();
        let (packed, width) = (bp.packed_slice::<u32>(), bp.bit_width() as usize);
    },
    |index, block| {
        // SAFETY: the block holds one chunk packed at `width`, and `block` holds a full chunk.
        unsafe {
            u32::unchecked_unfor_pack(width, packed_block(packed, width, index), reference, block)
        }
    }
);

tree_benches!(for_bp_u32: u32, for_bp_u32, for_bp_u32_materialize, for_bp_u32_sum);

// ---------------------------------------------------------------------------------------------
// BitPacked u32 with patches
// ---------------------------------------------------------------------------------------------

fn bp_patches_u32() -> ArrayRef {
    let values = PrimitiveArray::from_iter((0..LEN_U32).map(|i| {
        if i % 97 == 0 {
            100_000 + i
        } else {
            (i * 7) % 1000
        }
    }));
    bitpack_encode(&values, 10, None, &mut ctx())
        .vortex_expect("bitpack")
        .into_array()
}

fused_kernels!(
    u32,
    bp_patches_u32_materialize,
    bp_patches_u32_sum,
    |array| {
        let bp = array.as_::<BitPacked>();
        let (packed, width) = (bp.packed_slice::<u32>(), bp.bit_width() as usize);
        let patches = patch_list::<u32>(&bp.patches().vortex_expect("patches"));
        let mut cursor = 0;
    },
    |index, block| {
        // SAFETY: the block holds one chunk packed at `width`, and `block` holds a full chunk.
        unsafe { u32::unchecked_unpack(width, packed_block(packed, width, index), block) };
        patch_block(block, index, &patches, &mut cursor);
    }
);

tree_benches!(
    bp_patches_u32: u32,
    bp_patches_u32,
    bp_patches_u32_materialize,
    bp_patches_u32_sum
);

// ---------------------------------------------------------------------------------------------
// Sparse i64: a fill value with a patch every 64 rows
// ---------------------------------------------------------------------------------------------

fn sparse_i64() -> ArrayRef {
    let rows = (0..LEN_U32).step_by(64);
    let indices = PrimitiveArray::from_iter(rows.clone()).into_array();
    let values = PrimitiveArray::from_iter(rows.map(|i| i64::from(i) * 3)).into_array();
    Sparse::try_new(indices, values, LEN, Scalar::from(7i64))
        .vortex_expect("sparse")
        .into_array()
}

fused_kernels!(
    i64,
    sparse_i64_materialize,
    sparse_i64_sum,
    |array| {
        let sparse = array.as_::<Sparse>();
        let fill = sparse
            .fill_scalar()
            .as_primitive()
            .as_::<i64>()
            .vortex_expect("fill");
        let patches = patch_list::<i64>(&sparse.patches());
        let mut cursor = 0;
    },
    |index, block| {
        block.fill(fill);
        patch_block(block, index, &patches, &mut cursor);
    }
);

tree_benches!(sparse_i64: i64, sparse_i64, sparse_i64_materialize, sparse_i64_sum);

// ---------------------------------------------------------------------------------------------
// RLE u32 with bit-packed indices: runs of 16
// ---------------------------------------------------------------------------------------------

fn rle_bp_u32() -> ArrayRef {
    let mut ctx = ctx();
    let values = PrimitiveArray::from_iter((0..LEN_U32).map(|i| (i / 16) % 300 * 7));
    let rle = RLEData::encode(values.as_view(), &mut ctx).vortex_expect("rle");
    let indices = primitive(rle.indices());
    let indices = bitpack_encode(&indices, 6, None, &mut ctx).vortex_expect("bitpack");
    RLE::try_new(
        rle.values().clone(),
        indices.into_array(),
        rle.values_idx_offsets().clone(),
        0,
        LEN,
    )
    .vortex_expect("rle")
    .into_array()
}

fused_kernels!(
    u32,
    rle_bp_u32_materialize,
    rle_bp_u32_sum,
    |array| {
        let rle = array.as_::<RLE>();
        let values = primitive(rle.values());
        let values = values.as_slice::<u32>();
        let offsets = usizes(rle.values_idx_offsets());
        let bp = rle.indices().as_::<BitPacked>();
        let (packed, width) = (bp.packed_slice::<u16>(), bp.bit_width() as usize);
        let mut indices = [0u16; BLOCK];
    },
    |index, block| {
        // SAFETY: the block holds one chunk packed at `width`, and `indices` holds a full chunk.
        unsafe { u16::unchecked_unpack(width, packed_block(packed, width, index), &mut indices) };
        let start = offsets[index] - offsets[0];
        let end = offsets
            .get(index + 1)
            .map_or(values.len(), |&next| next - offsets[0]);
        if end - start == 1 {
            block.fill(values[start]);
        } else {
            // SAFETY: the encoder only emits indices into the block's own run values.
            unsafe { u32::decode_unchecked(&values[start..end], &indices, block) };
        }
    }
);

tree_benches!(rle_bp_u32: u32, rle_bp_u32, rle_bp_u32_materialize, rle_bp_u32_sum);

// ---------------------------------------------------------------------------------------------
// RunEnd i64: runs of 16
// ---------------------------------------------------------------------------------------------

fn runend_i64() -> ArrayRef {
    let values =
        PrimitiveArray::from_iter((0..i64::from(LEN_U32)).map(|i| (i / 16) % 300 * 7 - 1000));
    RunEnd::encode(values.into_array(), &mut ctx())
        .vortex_expect("runend")
        .into_array()
}

fused_kernels!(
    i64,
    runend_i64_materialize,
    runend_i64_sum,
    |array| {
        let runend = array.as_::<RunEnd>();
        let ends = usizes(runend.ends());
        let values = primitive(runend.values());
        let values = values.as_slice::<i64>();
        let mut run = 0;
    },
    |index, block| {
        let start = index * BLOCK;
        let mut row = start;
        while row < start + BLOCK {
            let end = ends[run].min(start + BLOCK);
            block[row - start..end - start].fill(values[run]);
            row = end;
            if ends[run] <= row {
                run += 1;
            }
        }
    }
);

tree_benches!(runend_i64: i64, runend_i64, runend_i64_materialize, runend_i64_sum);

// ---------------------------------------------------------------------------------------------
// Dict i64 with bit-packed u8 codes over 100 values
// ---------------------------------------------------------------------------------------------

fn dict_bp_i64() -> ArrayRef {
    let codes = PrimitiveArray::from_iter(
        (0..LEN_U32).map(|i| u8::try_from((i * 7) % 100).vortex_expect("code")),
    );
    let codes = bitpack_encode(&codes, 7, None, &mut ctx()).vortex_expect("bitpack");
    let values = PrimitiveArray::from_iter((0..100i64).map(|v| v * 1_000_003 - 50_000_000));
    DictArray::try_new(codes.into_array(), values.into_array())
        .vortex_expect("dict")
        .into_array()
}

fused_kernels!(
    i64,
    dict_bp_i64_materialize,
    dict_bp_i64_sum,
    |array| {
        let dict = array.as_::<Dict>();
        let values = primitive(dict.values());
        let values = values.as_slice::<i64>();
        let bp = dict.codes().as_::<BitPacked>();
        let (packed, width) = (bp.packed_slice::<u8>(), bp.bit_width() as usize);
        let mut codes = [0u8; BLOCK];
    },
    |index, block| {
        // SAFETY: the block holds one chunk packed at `width`, and `codes` holds a full chunk.
        unsafe { u8::unchecked_unpack(width, packed_block(packed, width, index), &mut codes) };
        for (out, &code) in block.iter_mut().zip(codes.iter()) {
            *out = values[usize::from(code)];
        }
    }
);

tree_benches!(dict_bp_i64: i64, dict_bp_i64, dict_bp_i64_materialize, dict_bp_i64_sum);

// ---------------------------------------------------------------------------------------------
// Delta u32 with FoR(BitPacked) deltas. The special kernel unpacks each block with its
// reference, undeltas it against the block's lane bases and untransposes it, all in L1.
// ---------------------------------------------------------------------------------------------

const U32_LANES: usize = 32;

fn delta_for_bp_u32() -> ArrayRef {
    let mut ctx = ctx();
    let mut acc = 1_000_000u32;
    let values = PrimitiveArray::from_iter((0..LEN_U32).map(|i| {
        acc += (i * 7) % 13;
        acc
    }));
    let delta = Delta::try_from_primitive_array(&values, &mut ctx).vortex_expect("delta");
    let deltas = FoR::encode(primitive(delta.deltas()), &mut ctx).vortex_expect("for");
    let bp = bitpack_encode(&primitive(deltas.encoded()), 4, None, &mut ctx).vortex_expect("bp");
    let deltas = FoR::try_new(
        bp.into_array(),
        deltas.constant_reference().vortex_expect("reference"),
    )
    .vortex_expect("for");
    Delta::try_new(delta.bases().clone(), deltas.into_array(), 0, LEN)
        .vortex_expect("delta")
        .into_array()
}

fused_kernels!(
    u32,
    delta_for_bp_u32_materialize,
    delta_for_bp_u32_sum,
    |array| {
        let delta = array.as_::<Delta>();
        let bases = primitive(delta.bases());
        let bases = bases.as_slice::<u32>();
        let reference = constant_reference::<u32>(delta.deltas());
        let for_ = delta.deltas().as_::<FoR>();
        let bp = for_.encoded().as_::<BitPacked>();
        let (packed, width) = (bp.packed_slice::<u32>(), bp.bit_width() as usize);
        let mut deltas = [0u32; BLOCK];
        let mut transposed = [0u32; BLOCK];
    },
    |index, block| {
        // SAFETY: the block holds one chunk packed at `width`, and `deltas` holds a full chunk.
        unsafe {
            u32::unchecked_unfor_pack(
                width,
                packed_block(packed, width, index),
                reference,
                &mut deltas,
            )
        };
        let lane_bases = bases[index * U32_LANES..]
            .first_chunk::<U32_LANES>()
            .vortex_expect("lane bases");
        u32::undelta::<U32_LANES>(&deltas, lane_bases, &mut transposed);
        u32::untranspose(&transposed, block);
    }
);

tree_benches!(
    delta_for_bp_u32: u32,
    delta_for_bp_u32,
    delta_for_bp_u32_materialize,
    delta_for_bp_u32_sum
);

// ---------------------------------------------------------------------------------------------
// ALP f64 over FoR(BitPacked), with patches
// ---------------------------------------------------------------------------------------------

fn alp_for_bp_f64() -> ArrayRef {
    let mut ctx = ctx();
    let values = PrimitiveArray::from_iter((0..LEN_U32).map(|i| {
        if i % 997 == 500 {
            PI * f64::from(i)
        } else {
            1000.0 + f64::from(i % 1000) * 0.01
        }
    }));
    let alp = alp_encode(values.as_view(), None, &mut ctx).vortex_expect("alp");
    let encoded = FoR::encode(primitive(alp.encoded()), &mut ctx).vortex_expect("for");
    let bp = bitpack_encode(&primitive(encoded.encoded()), 10, None, &mut ctx).vortex_expect("bp");
    let encoded = FoR::try_new(
        bp.into_array(),
        encoded.constant_reference().vortex_expect("reference"),
    )
    .vortex_expect("for");
    ALP::try_new(encoded.into_array(), alp.exponents(), alp.patches())
        .vortex_expect("alp")
        .into_array()
}

fused_kernels!(
    f64,
    alp_for_bp_f64_materialize,
    alp_for_bp_f64_sum,
    |array| {
        let alp = array.as_::<ALP>();
        let exponents: Exponents = alp.exponents();
        let patches = patch_list::<f64>(&alp.patches().vortex_expect("patches"));
        // The unpack adds the reference with wrapping arithmetic on the unsigned bits.
        let reference = constant_reference::<i64>(alp.encoded()) as u64;
        let for_ = alp.encoded().as_::<FoR>();
        let bp = for_.encoded().as_::<BitPacked>();
        let (packed, width) = (bp.packed_slice::<u64>(), bp.bit_width() as usize);
        let mut cursor = 0;
    },
    |index, block| {
        // SAFETY: `f64`, `i64` and `u64` share size and alignment, and any bits are valid.
        let bits = unsafe { &mut *(block as *mut [f64; BLOCK]).cast::<[u64; BLOCK]>() };
        // SAFETY: the block holds one chunk packed at `width`, and `bits` holds a full chunk.
        unsafe {
            u64::unchecked_unfor_pack(width, packed_block(packed, width, index), reference, bits)
        };
        // SAFETY: as above.
        let ints = unsafe { &mut *(block as *mut [f64; BLOCK]).cast::<[i64; BLOCK]>() };
        f64::decode_slice_inplace(ints, exponents);
        patch_block(block, index, &patches, &mut cursor);
    }
);

tree_benches!(
    alp_for_bp_f64: f64,
    alp_for_bp_f64,
    alp_for_bp_f64_materialize,
    alp_for_bp_f64_sum
);

// ---------------------------------------------------------------------------------------------
// Real scan stacks. In a census of the arrays the TPC-H SF1 queries execute, these are the trees
// that stream: a scan filters or slices FoR(BitPacked) and casts it to the nullability of its
// table schema, and FoR pushes both into its values, leaving FoR(Cast(Filter|Slice(BitPacked))).
// They have no fused kernel; the question is whether streaming beats level-wise execution.
// ---------------------------------------------------------------------------------------------

/// FoR over a cast of `values` that only makes them nullable, as scans leave it.
fn for_cast_i64(values: ArrayRef) -> ArrayRef {
    let cast = Cast::new(values, DType::Primitive(PType::I64, Nullability::Nullable));
    FoR::try_new(cast.into_array(), Scalar::from(1_000_000i64))
        .vortex_expect("for")
        .into_array()
}

/// Bit-packed i64 values, with a patch every 97 rows when `patched`.
fn bp_i64(patched: bool) -> ArrayRef {
    let values = PrimitiveArray::from_iter((0..i64::from(LEN_U32)).map(|i| {
        if patched && i % 97 == 0 {
            100_000 + i
        } else {
            (i * 7) % 1000
        }
    }));
    bitpack_encode(&values, 10, None, &mut ctx())
        .vortex_expect("bitpack")
        .into_array()
}

/// Rows kept out of every 16, so the filter's selectivity is `keep / 16`.
const FILTER_KEEP: &[usize] = &[1, 4, 12];

fn for_cast_filter_bp_i64(keep: usize) -> ArrayRef {
    let mask = Mask::from_iter((0..LEN).map(|i| i % 16 < keep));
    for_cast_i64(bp_i64(false).filter(mask).vortex_expect("filter"))
}

fn for_cast_slice_bp_i64() -> ArrayRef {
    for_cast_i64(SliceArray::new(bp_i64(true), 517..LEN - 300).into_array())
}

mod for_cast_filter_bp_i64 {
    use super::*;

    #[divan::bench(args = FILTER_KEEP)]
    fn materialize_streaming(bencher: Bencher, keep: usize) {
        super::materialize_streaming(bencher, for_cast_filter_bp_i64(keep));
    }

    #[divan::bench(args = FILTER_KEEP)]
    fn materialize_levelwise(bencher: Bencher, keep: usize) {
        super::materialize_levelwise(bencher, for_cast_filter_bp_i64(keep));
    }

    #[divan::bench(args = FILTER_KEEP)]
    fn sum_streaming(bencher: Bencher, keep: usize) {
        super::sum_streaming::<i64>(bencher, for_cast_filter_bp_i64(keep));
    }

    #[divan::bench(args = FILTER_KEEP)]
    fn sum_levelwise(bencher: Bencher, keep: usize) {
        super::sum_levelwise::<i64>(bencher, for_cast_filter_bp_i64(keep));
    }
}

mod for_cast_slice_bp_i64 {
    use super::*;

    #[divan::bench]
    fn materialize_streaming(bencher: Bencher) {
        super::materialize_streaming(bencher, for_cast_slice_bp_i64());
    }

    #[divan::bench]
    fn materialize_levelwise(bencher: Bencher) {
        super::materialize_levelwise(bencher, for_cast_slice_bp_i64());
    }

    #[divan::bench]
    fn sum_streaming(bencher: Bencher) {
        super::sum_streaming::<i64>(bencher, for_cast_slice_bp_i64());
    }

    #[divan::bench]
    fn sum_levelwise(bencher: Bencher) {
        super::sum_levelwise::<i64>(bencher, for_cast_slice_bp_i64());
    }
}

// ---------------------------------------------------------------------------------------------
// DecimalByteParts: decimals whose most significant parts are FoR(BitPacked), as the compressor
// leaves them. Without lower parts (up to 18 digits), the parts are the values; with one (up to
// 38 digits), streaming assembles each chunk of parts with its lower words.
// ---------------------------------------------------------------------------------------------

/// `values` as byte parts, the most significant part bit-packed under a frame of reference.
fn dbp(decimal: DecimalArray) -> ArrayRef {
    let parts = DecimalByteParts::encode(&decimal, &mut ctx()).vortex_expect("dbp");
    let msp = primitive(parts.msp());
    let min = msp
        .as_slice::<i64>()
        .iter()
        .copied()
        .min()
        .unwrap_or_default();
    let shifted = PrimitiveArray::from_iter(msp.as_slice::<i64>().iter().map(|&v| v - min));
    let bp = bitpack_encode(&shifted, 10, None, &mut ctx()).vortex_expect("bitpack");
    let msp = FoR::try_new(bp.into_array(), Scalar::from(min)).vortex_expect("for");
    DecimalByteParts::try_new_with_lower_parts(
        msp.into_array(),
        parts.lower_parts().to_vec(),
        decimal.decimal_dtype(),
    )
    .vortex_expect("dbp")
    .into_array()
}

fn dbp_i64() -> ArrayRef {
    let values = (0..i64::from(LEN_U32)).map(|i| (i * 7) % 1000 - 271);
    dbp(DecimalArray::new(
        values.collect::<Buffer<i64>>(),
        DecimalDType::new(18, 2),
        Validity::NonNullable,
    ))
}

fn dbp_i128() -> ArrayRef {
    let values = (0..i128::from(LEN_U32)).map(|i| ((i * 7) % 1000 - 271) * (1 << 64) + i * 7919);
    dbp(DecimalArray::new(
        values.collect::<Buffer<i128>>(),
        DecimalDType::new(38, 2),
        Validity::NonNullable,
    ))
}

fn materialize_levelwise_decimal(bencher: Bencher, array: ArrayRef) {
    bencher
        .with_inputs(|| (array.clone(), ctx()))
        .bench_values(|(array, mut ctx)| {
            array
                .execute::<DecimalArray>(&mut ctx)
                .vortex_expect("bench")
                .len()
        });
}

fn sum_levelwise_decimal<T: Fold + NativeDecimalType>(bencher: Bencher, array: ArrayRef) {
    bencher
        .with_inputs(|| (array.clone(), ctx()))
        .bench_values(|(array, mut ctx)| {
            let values = array
                .execute::<DecimalArray>(&mut ctx)
                .vortex_expect("bench");
            T::fold(0, &values.buffer::<T>())
        });
}

/// Generate the four benchmarks for one decimal tree, which has no fused kernel.
macro_rules! decimal_benches {
    ($tree:ident : $T:ty) => {
        mod $tree {
            use super::*;

            #[divan::bench]
            fn materialize_streaming(bencher: Bencher) {
                super::materialize_streaming(bencher, super::$tree());
            }

            #[divan::bench]
            fn materialize_levelwise(bencher: Bencher) {
                super::materialize_levelwise_decimal(bencher, super::$tree());
            }

            #[divan::bench]
            fn sum_streaming(bencher: Bencher) {
                super::sum_streaming::<$T>(bencher, super::$tree());
            }

            #[divan::bench]
            fn sum_levelwise(bencher: Bencher) {
                super::sum_levelwise_decimal::<$T>(bencher, super::$tree());
            }
        }
    };
}

decimal_benches!(dbp_i64: i64);
decimal_benches!(dbp_i128: i128);

/// Every fused kernel must agree with execution, or its timing means nothing.
fn check_fused_kernels() {
    fn check<T: NativePType>(array: ArrayRef, kernel: fn(&ArrayRef) -> Vec<T>) {
        assert_eq!(
            kernel(&array),
            primitive(&array).as_slice::<T>(),
            "fused kernel disagrees with execution for {}",
            array.encoding_id()
        );
    }
    check(for_bp_u32(), for_bp_u32_materialize);
    check(bp_patches_u32(), bp_patches_u32_materialize);
    check(sparse_i64(), sparse_i64_materialize);
    check(rle_bp_u32(), rle_bp_u32_materialize);
    check(runend_i64(), runend_i64_materialize);
    check(dict_bp_i64(), dict_bp_i64_materialize);
    check(delta_for_bp_u32(), delta_for_bp_u32_materialize);
    check(alp_for_bp_f64(), alp_for_bp_f64_materialize);

    // The real stacks have no fused kernel, so check that they stream as the benchmarks assume.
    for array in FILTER_KEEP
        .iter()
        .map(|&keep| for_cast_filter_bp_i64(keep))
        .chain([for_cast_slice_bp_i64()])
    {
        assert!(
            array.supports_decompress_chunks(),
            "{} does not stream",
            array.encoding_id()
        );
    }
    // The decimal trees are deep enough that the executor streams them.
    for array in [dbp_i64(), dbp_i128()] {
        assert!(
            array.should_execute_via_chunks(),
            "{} does not stream",
            array.dtype()
        );
    }
}
