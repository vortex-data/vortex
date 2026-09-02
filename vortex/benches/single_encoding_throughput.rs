// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]

use std::sync::LazyLock;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use divan::Bencher;
#[cfg(not(codspeed))]
use divan::counter::BytesCount;
use mimalloc::MiMalloc;
use rand::RngExt;
use rand::SeedableRng;
use rand::prelude::IndexedRandom;
use rand::rngs::StdRng;
use vortex::VortexSessionDefault;
use vortex::array::Canonical;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::arrays::Primitive;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::builders::dict::dict_encode;
use vortex::array::builtins::ArrayBuiltins;
use vortex::array::dtype::Nullability;
use vortex::compressor::BtrBlocksCompressorBuilder;
use vortex::dtype::NativePType;
use vortex::dtype::PType;
use vortex::dtype::half::f16;
use vortex::encodings::alp::RDEncoder;
use vortex::encodings::alp::RDEncoderExt;
use vortex::encodings::alp::alp_encode;
use vortex::encodings::block_residual::BlockResidual;
use vortex::encodings::block_residual::OrderedFloat;
use vortex::encodings::block_residual::OrderedFloatArraySlotsExt;
use vortex::encodings::fastlanes::BitPacked;
use vortex::encodings::fastlanes::Delta;
use vortex::encodings::fastlanes::DeltaData;
use vortex::encodings::fastlanes::FoR;
use vortex::encodings::fastlanes::FoRArrayExt;
use vortex::encodings::fastlanes::FoRArraySlotsExt;
use vortex::encodings::fastlanes::bitpack_compress::bitpack_encode_unchecked;
use vortex::encodings::fastlanes::delta_compress;
use vortex::encodings::float_quant::FloatQuant;
use vortex::encodings::float_quant::FloatQuantArraySlotsExt;
use vortex::encodings::float_quant::analyze_float_quant;
use vortex::encodings::fsst::fsst_compress;
use vortex::encodings::fsst::fsst_train_compressor;
use vortex::encodings::pco::Pco;
use vortex::encodings::runend::RunEnd;
use vortex::encodings::sequence::sequence_encode;
use vortex::encodings::zigzag::zigzag_encode;
use vortex::encodings::zstd::Zstd;
use vortex::encodings::zstd::ZstdData;
use vortex::scalar::Scalar;
use vortex_array::VortexSessionExecute;
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_btrblocks::SchemeExt;
use vortex_btrblocks::schemes::float::FloatQuantScheme;
use vortex_btrblocks::schemes::float::OrderedBlockResidualScheme;
use vortex_btrblocks::schemes::integer::BlockResidualScheme;
use vortex_error::VortexResult;
use vortex_sequence::Sequence;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::default);

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

// Sizes are chosen to keep each CodSpeed run well under 1ms; zstd and pco get
// smaller inputs because they are much slower per element.
const NUM_VALUES: u64 = 4096;
const PCO_NUM_VALUES: u64 = 1024;
const PCO_COMPRESSION_LEVEL: usize = 8;
const PCO_VALUES_PER_PAGE: usize = 8192;
#[cfg(feature = "zstd")]
const ZSTD_NUM_VALUES: u64 = 128;
const STRING_NUM_VALUES: usize = 2048;
#[cfg(feature = "zstd")]
const ZSTD_STRING_NUM_VALUES: usize = 200;
// Uniqueness fractions keep ~5 unique strings, as in the original 100k * 0.00005 workload.
const STRING_UNIQUENESS: f64 = 0.0025;
#[cfg(feature = "zstd")]
const ZSTD_STRING_UNIQUENESS: f64 = 0.025;

// Helper function to conditionally add counter based on codspeed cfg
fn with_byte_counter<'a, 'b>(bencher: Bencher<'a, 'b>, bytes: u64) -> Bencher<'a, 'b> {
    #[cfg(not(codspeed))]
    return bencher.counter(BytesCount::new(bytes));
    #[cfg(codspeed)]
    {
        _ = bytes; // Consume the bytes value to avoid unused variable warning.
        return bencher;
    }
}

fn canonicalize(array: impl IntoArray, ctx: &mut ExecutionCtx) -> VortexResult<Canonical> {
    array.into_array().execute::<Canonical>(ctx)
}

fn bench_compressor(bencher: Bencher, array: PrimitiveArray, compressor: BtrBlocksCompressor) {
    with_byte_counter(bencher, array.nbytes())
        .with_inputs(|| (array.clone().into_array(), SESSION.create_execution_ctx()))
        .bench_values(|(array, mut ctx)| compressor.compress(&array, &mut ctx).unwrap());
}

fn default_with_float_quant() -> BtrBlocksCompressorBuilder {
    let builder = BtrBlocksCompressorBuilder::default();

    #[cfg(not(feature = "unstable_encodings"))]
    let builder = builder.with_new_scheme(&FloatQuantScheme);

    builder
}

fn proposed_default_builder() -> BtrBlocksCompressorBuilder {
    let builder = BtrBlocksCompressorBuilder::default();

    #[cfg(not(feature = "unstable_encodings"))]
    let builder = builder
        .with_new_scheme(&FloatQuantScheme)
        .with_new_scheme(&OrderedBlockResidualScheme)
        .with_new_scheme(&BlockResidualScheme);

    builder
}

// Setup functions
fn setup_primitive_arrays(len: u64) -> (PrimitiveArray, PrimitiveArray, PrimitiveArray) {
    let mut ctx = SESSION.create_execution_ctx();
    let mut rng = StdRng::seed_from_u64(0);
    let uint_array = PrimitiveArray::from_iter((0..len).map(|_| rng.random_range(42u32..256)));
    let int_array = uint_array
        .clone()
        .into_array()
        .cast(PType::I32.into())
        .unwrap()
        .execute::<PrimitiveArray>(&mut ctx)
        .unwrap();
    let float_array = uint_array
        .clone()
        .into_array()
        .cast(PType::F64.into())
        .unwrap()
        .execute::<PrimitiveArray>(&mut ctx)
        .unwrap();
    (uint_array, int_array, float_array)
}

fn setup_widened_f32_array() -> PrimitiveArray {
    let mut rng = StdRng::seed_from_u64(1);
    PrimitiveArray::from_iter((0..NUM_VALUES).map(|index| {
        let trend = (index % 10_000) as f32 * 0.001;
        f64::from(trend + rng.random_range(-1.0_f32..1.0))
    }))
}

fn setup_quantized_f32_array() -> PrimitiveArray {
    PrimitiveArray::from_iter((0..NUM_VALUES).map(|index| {
        let mantissa = (index.wrapping_mul(7_919) as u32 & 0x7fff) << 8;
        f32::from_bits(0x3f80_0000 | mantissa)
    }))
}

fn setup_quantized_f16_array() -> PrimitiveArray {
    PrimitiveArray::from_iter((0..NUM_VALUES).map(|index| {
        let mantissa = (index.wrapping_mul(7_919) as u16) & 0x03f0;
        f16::from_bits(0x3c00 | mantissa)
    }))
}

fn setup_general_f16_array() -> PrimitiveArray {
    PrimitiveArray::from_iter(
        (0..NUM_VALUES).map(|index| f16::from_bits(index.wrapping_mul(7_919) as u16)),
    )
}

fn setup_general_f32_array() -> PrimitiveArray {
    PrimitiveArray::from_iter((0..NUM_VALUES).map(|index| {
        let mantissa = index.wrapping_mul(7_919) as u32 & 0x007f_ffff;
        f32::from_bits(0x3f80_0000 | mantissa)
    }))
}

fn setup_general_f64_array() -> PrimitiveArray {
    PrimitiveArray::from_iter((0..NUM_VALUES).map(|index| {
        let mantissa = index.wrapping_mul(0x9e37_79b9_7f4a_7c15) & 0x000f_ffff_ffff_ffff;
        f64::from_bits(0x3ff0_0000_0000_0000 | mantissa)
    }))
}

fn setup_float_quant_near_miss_f32_array() -> PrimitiveArray {
    PrimitiveArray::from_iter((0..NUM_VALUES).map(|index| {
        let scrambled = (index as u32).wrapping_mul(2_654_435_761);
        let sign = (scrambled & 1) << 31;
        let exponent = ((scrambled >> 1) % 254 + 1) << 23;
        let mantissa = scrambled & 0x007f_fffc;
        f32::from_bits(sign | exponent | mantissa)
    }))
}

fn setup_float_quant_near_miss_f64_array() -> PrimitiveArray {
    PrimitiveArray::from_iter((0..NUM_VALUES).map(|index| {
        let scrambled = index.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        let sign = (scrambled & 1) << 63;
        let exponent = ((scrambled >> 1) % 2_046 + 1) << 52;
        let mantissa = scrambled & 0x000f_ffff_ffff_fffc;
        f64::from_bits(sign | exponent | mantissa)
    }))
}

fn setup_nonzero_secondary_array() -> PrimitiveArray {
    let widened = setup_widened_f32_array();
    PrimitiveArray::from_iter(
        widened
            .as_slice::<f64>()
            .iter()
            .enumerate()
            .map(|(index, value)| {
                if index % 10 == 0 {
                    f64::from_bits(value.to_bits() | 1)
                } else {
                    *value
                }
            }),
    )
}

fn setup_nonzero_secondary_f16_array() -> PrimitiveArray {
    let quantized = setup_quantized_f16_array();
    PrimitiveArray::from_iter(quantized.as_slice::<f16>().iter().enumerate().map(
        |(index, value)| {
            if index % 10 == 0 {
                f16::from_bits(value.to_bits() | 1)
            } else {
                *value
            }
        },
    ))
}

fn setup_nonzero_secondary_f32_array() -> PrimitiveArray {
    let quantized = setup_quantized_f32_array();
    PrimitiveArray::from_iter(quantized.as_slice::<f32>().iter().enumerate().map(
        |(index, value)| {
            if index % 10 == 0 {
                f32::from_bits(value.to_bits() | 1)
            } else {
                *value
            }
        },
    ))
}

fn setup_secondary_width_array(width: u8) -> PrimitiveArray {
    let widened = setup_widened_f32_array();
    let low_mask = (1_u64 << width) - 1;
    PrimitiveArray::from_iter(
        widened
            .as_slice::<f64>()
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let low = (index as u64).wrapping_mul(2_654_435_761) & low_mask;
                f64::from_bits(value.to_bits() | low)
            }),
    )
}

fn setup_random_walk_array() -> PrimitiveArray {
    let mut rng = StdRng::seed_from_u64(2);
    let mut value = 1_000.0_f64;
    PrimitiveArray::from_iter((0..NUM_VALUES).map(|_| {
        value += rng.random_range(-0.01_f64..0.01);
        value
    }))
}

fn setup_block_local_u64_array() -> PrimitiveArray {
    PrimitiveArray::from_iter((0..NUM_VALUES).map(|index| {
        let block = index / 1_024;
        let residual = index.wrapping_mul(2_654_435_761) % 1_024;
        block * 1_000_000 + residual
    }))
}

fn setup_block_local_u32_array() -> PrimitiveArray {
    PrimitiveArray::from_iter((0..NUM_VALUES).map(|index| {
        let block = index / 1_024;
        let residual = index.wrapping_mul(2_654_435_761) % 1_024;
        (block * 1_000_000 + residual) as u32
    }))
}

fn setup_block_local_i32_array() -> PrimitiveArray {
    PrimitiveArray::from_iter((0..NUM_VALUES).map(|index| {
        let block = index / 1_024;
        let residual = index.wrapping_mul(2_654_435_761) % 1_024;
        (block as i32 - 1_000) * 1_000_000 + residual as i32
    }))
}

fn setup_patch_density_u32_array(stride: u64) -> PrimitiveArray {
    PrimitiveArray::from_iter((0..NUM_VALUES).map(|index| {
        if index % stride == 0 {
            u32::MAX - index as u32
        } else {
            42
        }
    }))
}

fn setup_ordered_f32_array() -> PrimitiveArray {
    PrimitiveArray::from_iter((0..NUM_VALUES).map(|index| {
        let block = index / 1_024;
        let residual = index.wrapping_mul(7_919) % 1_024;
        f32::from_bits(0x3f80_0000 + (block as u32 * 0x1_0000) + residual as u32)
    }))
}

fn setup_ordered_f16_array() -> PrimitiveArray {
    PrimitiveArray::from_iter((0..NUM_VALUES).map(|index| {
        let block = (index / 1_024) % 8;
        let residual = index.wrapping_mul(7_919) % 64;
        f16::from_bits(0x3c00 + (block * 64 + residual) as u16)
    }))
}

fn setup_block_local_i16_array() -> PrimitiveArray {
    PrimitiveArray::from_iter((0..NUM_VALUES).map(|index| {
        let block = (index / 1_024) % 128;
        let residual = index.wrapping_mul(2_654_435_761) % 128;
        (block * 128 + residual) as i16
    }))
}

fn setup_block_local_integer_array<T: NativePType>() -> PrimitiveArray {
    match T::PTYPE {
        PType::U8 => PrimitiveArray::from_iter((0..NUM_VALUES).map(|index| {
            let block = (index / 1_024) % 16;
            let residual = index.wrapping_mul(2_654_435_761) % 16;
            (block * 16 + residual) as u8
        })),
        PType::U16 => PrimitiveArray::from_iter((0..NUM_VALUES).map(|index| {
            let block = (index / 1_024) % 256;
            let residual = index.wrapping_mul(2_654_435_761) % 256;
            (block * 256 + residual) as u16
        })),
        PType::U32 => setup_block_local_u32_array(),
        PType::U64 => setup_block_local_u64_array(),
        PType::I8 => PrimitiveArray::from_iter((0..NUM_VALUES).map(|index| {
            let block = (index / 1_024) % 16;
            let residual = index.wrapping_mul(2_654_435_761) % 16;
            ((block * 16 + residual) as i16 - 128) as i8
        })),
        PType::I16 => setup_block_local_i16_array(),
        PType::I32 => setup_block_local_i32_array(),
        PType::I64 => PrimitiveArray::from_iter((0..NUM_VALUES).map(|index| {
            let block = index / 1_024;
            let residual = index.wrapping_mul(2_654_435_761) % 1_024;
            (block as i64 - 1_000) * 1_000_000_000_000 + residual as i64
        })),
        ptype => unreachable!("unsupported block residual benchmark type {ptype}"),
    }
}

fn encode_for_bitpacked_tree(array: &PrimitiveArray, bit_width: u8) -> vortex::array::ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let encoded = FoR::encode(array.clone(), &mut ctx).unwrap();
    let bitpacked = BitPacked::encode(encoded.encoded(), bit_width, &mut ctx).unwrap();
    FoR::try_new(bitpacked.into_array(), encoded.reference_scalar().clone())
        .unwrap()
        .into_array()
}

fn ordered_values(array: &PrimitiveArray) -> PrimitiveArray {
    let ordered = OrderedFloat::from_primitive(array.as_view()).unwrap();
    ordered
        .encoded()
        .clone()
        .execute::<PrimitiveArray>(&mut SESSION.create_execution_ctx())
        .unwrap()
}

fn encode_ordered_block_residual(array: &PrimitiveArray) -> vortex::array::ArrayRef {
    let ordered = OrderedFloat::from_primitive(array.as_view()).unwrap();
    let residuals = BlockResidual::from_primitive(ordered.encoded().as_::<Primitive>()).unwrap();
    OrderedFloat::try_new(residuals.into_array(), array.ptype())
        .unwrap()
        .into_array()
}

fn encode_float_quant_tree(array: &PrimitiveArray) -> vortex::array::ArrayRef {
    let analysis = analyze_float_quant(array.as_view()).unwrap();
    assert_eq!(analysis.secondary_bit_width, 0);
    let primary =
        FloatQuant::primary_for_primitive(array.as_view(), analysis.k, analysis.primary_min)
            .unwrap();
    // SAFETY: The analysis computes this width from the exact primary range.
    let primary = unsafe { bitpack_encode_unchecked(primary, analysis.primary_bit_width) }.unwrap();
    let reference = if array.ptype() == PType::F32 {
        Scalar::from(u32::try_from(analysis.primary_min).unwrap())
    } else {
        debug_assert_eq!(array.ptype(), PType::F64);
        Scalar::from(analysis.primary_min)
    };
    let primary = FoR::try_new(primary.into_array(), reference).unwrap();
    FloatQuant::try_new(primary.into_array(), None, array.ptype(), analysis.k)
        .unwrap()
        .into_array()
}

fn encode_float_quant_nonzero_secondary_tree(array: &PrimitiveArray) -> vortex::array::ArrayRef {
    let analysis = analyze_float_quant(array.as_view()).unwrap();
    assert_ne!(analysis.secondary_bit_width, 0);
    let split = FloatQuant::from_primitive(array.as_view(), analysis.k).unwrap();
    let primary = split
        .primary()
        .clone()
        .execute::<PrimitiveArray>(&mut SESSION.create_execution_ctx())
        .unwrap();
    let secondary = split
        .secondary()
        .unwrap()
        .clone()
        .execute::<PrimitiveArray>(&mut SESSION.create_execution_ctx())
        .unwrap();
    let biased_primary = PrimitiveArray::from_iter(
        primary
            .as_slice::<u64>()
            .iter()
            .map(|value| value - analysis.primary_min),
    );
    // SAFETY: The analysis computes this width from the exact primary range.
    let primary = unsafe { bitpack_encode_unchecked(biased_primary, analysis.primary_bit_width) }
        .unwrap()
        .into_array();
    let primary = FoR::try_new(primary, Scalar::from(analysis.primary_min))
        .unwrap()
        .into_array();
    // SAFETY: The analysis computes the exact secondary width.
    let secondary = unsafe { bitpack_encode_unchecked(secondary, analysis.secondary_bit_width) }
        .unwrap()
        .into_array();
    FloatQuant::try_new(primary, Some(secondary), PType::F64, analysis.k)
        .unwrap()
        .into_array()
}

fn encode_float_quant_scheme_tree(array: &PrimitiveArray) -> vortex::array::ArrayRef {
    BtrBlocksCompressorBuilder::empty()
        .with_new_scheme(&FloatQuantScheme)
        .build()
        .compress(
            &array.clone().into_array(),
            &mut SESSION.create_execution_ctx(),
        )
        .unwrap()
}

fn encode_float_quant_nonzero_secondary_scheme_tree(
    array: &PrimitiveArray,
) -> vortex::array::ArrayRef {
    let encoded = encode_float_quant_scheme_tree(array);
    let float_quant = encoded.as_::<FloatQuant>();
    assert!(float_quant.secondary().is_some());
    encoded
}

fn encode_prior_default(array: &PrimitiveArray) -> vortex::array::ArrayRef {
    BtrBlocksCompressorBuilder::default()
        .exclude_schemes([
            FloatQuantScheme.id(),
            OrderedBlockResidualScheme.id(),
            BlockResidualScheme.id(),
        ])
        .build()
        .compress(
            &array.clone().into_array(),
            &mut SESSION.create_execution_ctx(),
        )
        .unwrap()
}

fn encode_proposed_default(array: &PrimitiveArray) -> vortex::array::ArrayRef {
    proposed_default_builder()
        .build()
        .compress(
            &array.clone().into_array(),
            &mut SESSION.create_execution_ctx(),
        )
        .unwrap()
}

#[expect(clippy::cast_possible_truncation)]
fn gen_varbin_words(len: usize, uniqueness: f64) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(0);
    let uniq_cnt = (len as f64 * uniqueness) as usize;
    let dict: Vec<String> = (0..uniq_cnt)
        .map(|_| {
            (0..8)
                .map(|_| (rng.random_range(b'a'..=b'z')) as char)
                .collect()
        })
        .collect();
    (0..len)
        .map(|_| dict.choose(&mut rng).unwrap().clone())
        .collect()
}

// Primitive compression benchmarks
#[divan::bench(name = "bitpacked_compress_u32")]
fn bench_bitpacked_compress_u32(bencher: Bencher) {
    use vortex::encodings::fastlanes::bitpack_compress::bitpack_encode_unchecked;

    let (uint_array, ..) = setup_primitive_arrays(NUM_VALUES);
    let bit_width = 8;

    with_byte_counter(bencher, NUM_VALUES * 4)
        .with_inputs(|| uint_array.clone())
        .bench_values(|a| unsafe { bitpack_encode_unchecked(a, bit_width).unwrap() });
}

#[divan::bench(name = "bitpacked_decompress_u32")]
fn bench_bitpacked_decompress_u32(bencher: Bencher) {
    use vortex::encodings::fastlanes::bitpack_compress::bitpack_encode;

    let (uint_array, ..) = setup_primitive_arrays(NUM_VALUES);
    let bit_width = 8;
    let compressed = bitpack_encode(
        &uint_array,
        bit_width,
        None,
        &mut SESSION.create_execution_ctx(),
    )
    .unwrap()
    .into_array();

    with_byte_counter(bencher, NUM_VALUES * 4)
        .with_inputs(|| (&compressed, SESSION.create_execution_ctx()))
        .bench_refs(|(a, ctx)| canonicalize((**a).clone(), ctx));
}

#[divan::bench(name = "runend_compress_u32")]
fn bench_runend_compress_u32(bencher: Bencher) {
    let (uint_array, ..) = setup_primitive_arrays(NUM_VALUES);

    with_byte_counter(bencher, NUM_VALUES * 4)
        .with_inputs(|| (uint_array.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(a, mut ctx)| RunEnd::encode(a.into_array(), &mut ctx).unwrap());
}

#[divan::bench(name = "runend_decompress_u32")]
fn bench_runend_decompress_u32(bencher: Bencher) {
    let (uint_array, ..) = setup_primitive_arrays(NUM_VALUES);
    let compressed =
        RunEnd::encode(uint_array.into_array(), &mut SESSION.create_execution_ctx()).unwrap();

    with_byte_counter(bencher, NUM_VALUES * 4)
        .with_inputs(|| (&compressed, SESSION.create_execution_ctx()))
        .bench_refs(|(a, ctx)| canonicalize((**a).clone(), ctx));
}

#[divan::bench(name = "delta_compress_u32")]
fn bench_delta_compress_u32(bencher: Bencher) {
    let (uint_array, ..) = setup_primitive_arrays(NUM_VALUES);

    with_byte_counter(bencher, NUM_VALUES * 4)
        .with_inputs(|| (&uint_array, SESSION.create_execution_ctx()))
        .bench_refs(|(a, ctx)| {
            let (_bases, _deltas) = delta_compress(a, ctx).unwrap();
            DeltaData::try_new(0).unwrap()
        });
}

#[divan::bench(name = "delta_decompress_u32")]
fn bench_delta_decompress_u32(bencher: Bencher) {
    let (uint_array, ..) = setup_primitive_arrays(NUM_VALUES);
    let (bases, deltas) = delta_compress(&uint_array, &mut SESSION.create_execution_ctx()).unwrap();
    let compressed = Delta::try_new(bases.into_array(), deltas.into_array(), 0, uint_array.len())
        .unwrap()
        .into_array();

    with_byte_counter(bencher, NUM_VALUES * 4)
        .with_inputs(|| (&compressed, SESSION.create_execution_ctx()))
        .bench_refs(|(a, ctx)| canonicalize((**a).clone(), ctx));
}

#[divan::bench(name = "for_compress_i32")]
fn bench_for_compress_i32(bencher: Bencher) {
    let (_, int_array, _) = setup_primitive_arrays(NUM_VALUES);

    with_byte_counter(bencher, NUM_VALUES * 4)
        .with_inputs(|| (int_array.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(a, mut ctx)| FoR::encode(a, &mut ctx).unwrap());
}

#[divan::bench(name = "for_decompress_i32")]
fn bench_for_decompress_i32(bencher: Bencher) {
    let (_, int_array, _) = setup_primitive_arrays(NUM_VALUES);
    let compressed = FoR::encode(int_array, &mut SESSION.create_execution_ctx()).unwrap();

    with_byte_counter(bencher, NUM_VALUES * 4)
        .with_inputs(|| (&compressed, SESSION.create_execution_ctx()))
        .bench_refs(|(a, ctx)| canonicalize((**a).clone(), ctx));
}

#[divan::bench(name = "dict_compress_u32")]
fn bench_dict_compress_u32(bencher: Bencher) {
    let (uint_array, ..) = setup_primitive_arrays(NUM_VALUES);
    let array = uint_array.into_array();

    with_byte_counter(bencher, NUM_VALUES * 4)
        .with_inputs(|| (&array, SESSION.create_execution_ctx()))
        .bench_refs(|(a, ctx)| dict_encode(a, ctx).unwrap());
}

#[divan::bench(name = "dict_decompress_u32")]
fn bench_dict_decompress_u32(bencher: Bencher) {
    let (uint_array, ..) = setup_primitive_arrays(NUM_VALUES);
    let compressed = dict_encode(
        &uint_array.into_array(),
        &mut SESSION.create_execution_ctx(),
    )
    .unwrap();

    with_byte_counter(bencher, NUM_VALUES * 4)
        .with_inputs(|| (&compressed, SESSION.create_execution_ctx()))
        .bench_refs(|(a, ctx)| canonicalize((**a).clone(), ctx));
}

#[divan::bench(name = "zigzag_compress_i32")]
fn bench_zigzag_compress_i32(bencher: Bencher) {
    let (_, int_array, _) = setup_primitive_arrays(NUM_VALUES);

    with_byte_counter(bencher, NUM_VALUES * 4)
        .with_inputs(|| int_array.clone())
        .bench_values(|a| zigzag_encode(a.as_view()).unwrap());
}

#[divan::bench(name = "zigzag_decompress_i32")]
fn bench_zigzag_decompress_i32(bencher: Bencher) {
    let (_, int_array, _) = setup_primitive_arrays(NUM_VALUES);
    let compressed = zigzag_encode(int_array.as_view()).unwrap().into_array();

    with_byte_counter(bencher, NUM_VALUES * 4)
        .with_inputs(|| (&compressed, SESSION.create_execution_ctx()))
        .bench_refs(|(a, ctx)| canonicalize((**a).clone(), ctx));
}

#[expect(clippy::cast_possible_truncation)]
#[divan::bench(name = "sequence_compress_u32")]
fn bench_sequence_compress_u32(bencher: Bencher) {
    let seq_array = PrimitiveArray::from_iter(0..NUM_VALUES as u32);

    with_byte_counter(bencher, NUM_VALUES * 4)
        .with_inputs(|| (seq_array.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(a, mut ctx)| sequence_encode(a.as_view(), &mut ctx).unwrap().unwrap());
}

#[expect(clippy::cast_possible_truncation)]
#[divan::bench(name = "sequence_decompress_u32")]
fn bench_sequence_decompress_u32(bencher: Bencher) {
    let compressed = Sequence::try_new_typed(0, 1, Nullability::NonNullable, NUM_VALUES as usize)
        .unwrap()
        .into_array();

    with_byte_counter(bencher, NUM_VALUES * 4)
        .with_inputs(|| (&compressed, SESSION.create_execution_ctx()))
        .bench_refs(|(a, ctx)| canonicalize((**a).clone(), ctx));
}

#[divan::bench(name = "alp_compress_f64")]
fn bench_alp_compress_f64(bencher: Bencher) {
    let (_, _, float_array) = setup_primitive_arrays(NUM_VALUES);

    with_byte_counter(bencher, NUM_VALUES * 8)
        .with_inputs(|| (&float_array, SESSION.create_execution_ctx()))
        .bench_refs(|(a, ctx)| alp_encode(a.as_view(), None, ctx).unwrap());
}

#[divan::bench(name = "alp_decompress_f64")]
fn bench_alp_decompress_f64(bencher: Bencher) {
    let (_, _, float_array) = setup_primitive_arrays(NUM_VALUES);
    let compressed = alp_encode(
        float_array.as_view(),
        None,
        &mut SESSION.create_execution_ctx(),
    )
    .unwrap();

    with_byte_counter(bencher, NUM_VALUES * 8)
        .with_inputs(|| (&compressed, SESSION.create_execution_ctx()))
        .bench_refs(|(a, ctx)| canonicalize((**a).clone(), ctx));
}

#[divan::bench(name = "alp_rd_compress_f64")]
fn bench_alp_rd_compress_f64(bencher: Bencher) {
    let (_, _, float_array) = setup_primitive_arrays(NUM_VALUES);

    with_byte_counter(bencher, NUM_VALUES * 8)
        .with_inputs(|| &float_array)
        .bench_refs(|a| {
            let encoder = RDEncoder::new(a.as_slice::<f64>());
            encoder.encode(a.as_view())
        });
}

#[divan::bench(name = "alp_rd_decompress_f64")]
fn bench_alp_rd_decompress_f64(bencher: Bencher) {
    let (_, _, float_array) = setup_primitive_arrays(NUM_VALUES);
    let encoder = RDEncoder::new(float_array.as_slice::<f64>());
    let compressed = encoder.encode(float_array.as_view());

    with_byte_counter(bencher, NUM_VALUES * 8)
        .with_inputs(|| (&compressed, SESSION.create_execution_ctx()))
        .bench_refs(|(a, ctx)| canonicalize((**a).clone(), ctx));
}

#[divan::bench(name = "ordered_float_compress_f64")]
fn bench_ordered_float_compress_f64(bencher: Bencher) {
    let float_array = setup_random_walk_array();

    with_byte_counter(bencher, NUM_VALUES * 8)
        .with_inputs(|| &float_array)
        .bench_refs(|array| OrderedFloat::from_primitive(array.as_view()).unwrap());
}

#[divan::bench(name = "ordered_float_decompress_f64")]
fn bench_ordered_float_decompress_f64(bencher: Bencher) {
    let float_array = setup_random_walk_array();
    let encoded = OrderedFloat::from_primitive(float_array.as_view())
        .unwrap()
        .into_array();

    with_byte_counter(bencher, NUM_VALUES * 8)
        .with_inputs(|| (&encoded, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| canonicalize((**array).clone(), ctx));
}

#[divan::bench(name = "ordered_float_scalar_at_f64")]
fn bench_ordered_float_scalar_at_f64(bencher: Bencher) {
    let encoded = OrderedFloat::from_primitive(setup_random_walk_array().as_view())
        .unwrap()
        .into_array();
    let next_index = AtomicUsize::new(0);

    bencher
        .with_inputs(|| {
            (
                &encoded,
                SESSION.create_execution_ctx(),
                next_index.fetch_add(2_654_435_761, Ordering::Relaxed) % encoded.len(),
            )
        })
        .bench_values(|(array, mut ctx, index)| array.execute_scalar(index, &mut ctx).unwrap());
}

#[divan::bench(name = "ordered_float_compress_f16")]
fn bench_ordered_float_compress_f16(bencher: Bencher) {
    let float_array = setup_ordered_f16_array();

    with_byte_counter(bencher, NUM_VALUES * 2)
        .with_inputs(|| &float_array)
        .bench_refs(|array| OrderedFloat::from_primitive(array.as_view()).unwrap());
}

#[divan::bench(name = "ordered_float_decompress_f16")]
fn bench_ordered_float_decompress_f16(bencher: Bencher) {
    let encoded = OrderedFloat::from_primitive(setup_ordered_f16_array().as_view())
        .unwrap()
        .into_array();

    with_byte_counter(bencher, NUM_VALUES * 2)
        .with_inputs(|| (&encoded, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| canonicalize((**array).clone(), ctx));
}

#[divan::bench(name = "ordered_float_scalar_at_f16")]
fn bench_ordered_float_scalar_at_f16(bencher: Bencher) {
    let encoded = OrderedFloat::from_primitive(setup_ordered_f16_array().as_view())
        .unwrap()
        .into_array();
    let next_index = AtomicUsize::new(0);

    bencher
        .with_inputs(|| {
            (
                &encoded,
                SESSION.create_execution_ctx(),
                next_index.fetch_add(2_654_435_761, Ordering::Relaxed) % encoded.len(),
            )
        })
        .bench_values(|(array, mut ctx, index)| array.execute_scalar(index, &mut ctx).unwrap());
}

#[divan::bench(name = "ordered_float_compress_f32")]
fn bench_ordered_float_compress_f32(bencher: Bencher) {
    let float_array = setup_ordered_f32_array();

    with_byte_counter(bencher, NUM_VALUES * 4)
        .with_inputs(|| &float_array)
        .bench_refs(|array| OrderedFloat::from_primitive(array.as_view()).unwrap());
}

#[divan::bench(name = "ordered_float_decompress_f32")]
fn bench_ordered_float_decompress_f32(bencher: Bencher) {
    let encoded = OrderedFloat::from_primitive(setup_ordered_f32_array().as_view())
        .unwrap()
        .into_array();

    with_byte_counter(bencher, NUM_VALUES * 4)
        .with_inputs(|| (&encoded, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| canonicalize((**array).clone(), ctx));
}

#[divan::bench(name = "ordered_float_scalar_at_f32")]
fn bench_ordered_float_scalar_at_f32(bencher: Bencher) {
    let encoded = OrderedFloat::from_primitive(setup_ordered_f32_array().as_view())
        .unwrap()
        .into_array();
    let next_index = AtomicUsize::new(0);

    bencher
        .with_inputs(|| {
            (
                &encoded,
                SESSION.create_execution_ctx(),
                next_index.fetch_add(2_654_435_761, Ordering::Relaxed) % encoded.len(),
            )
        })
        .bench_values(|(array, mut ctx, index)| array.execute_scalar(index, &mut ctx).unwrap());
}

#[divan::bench(name = "block_residual_compress_u64")]
fn bench_block_residual_compress_u64(bencher: Bencher) {
    let float_array = setup_random_walk_array();
    let ordered = ordered_values(&float_array);

    with_byte_counter(bencher, NUM_VALUES * 8)
        .with_inputs(|| &ordered)
        .bench_refs(|array| BlockResidual::from_primitive(array.as_view()).unwrap());
}

#[divan::bench(name = "block_residual_decompress_u64")]
fn bench_block_residual_decompress_u64(bencher: Bencher) {
    let float_array = setup_random_walk_array();
    let ordered = ordered_values(&float_array);
    let encoded = BlockResidual::from_primitive(ordered.as_view())
        .unwrap()
        .into_array();

    with_byte_counter(bencher, NUM_VALUES * 8)
        .with_inputs(|| (&encoded, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| canonicalize((**array).clone(), ctx));
}

#[divan::bench(types = [u8, u16, u32, u64, i8, i16, i32, i64])]
fn block_local_block_residual_compress<T: NativePType>(bencher: Bencher) {
    let array = setup_block_local_integer_array::<T>();
    let byte_width = u64::try_from(T::PTYPE.byte_width()).unwrap();

    with_byte_counter(bencher, NUM_VALUES * byte_width)
        .with_inputs(|| &array)
        .bench_refs(|array| BlockResidual::from_primitive(array.as_view()).unwrap());
}

#[divan::bench(types = [u8, u16, u32, u64, i8, i16, i32, i64])]
fn block_local_block_residual_decompress<T: NativePType>(bencher: Bencher) {
    let encoded = BlockResidual::from_primitive(setup_block_local_integer_array::<T>().as_view())
        .unwrap()
        .into_array();
    let byte_width = u64::try_from(T::PTYPE.byte_width()).unwrap();

    with_byte_counter(bencher, NUM_VALUES * byte_width)
        .with_inputs(|| (&encoded, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| canonicalize((**array).clone(), ctx));
}

#[divan::bench(types = [u8, u16, u32, u64, i8, i16, i32, i64])]
fn block_local_block_residual_scalar_at<T: NativePType>(bencher: Bencher) {
    let encoded = BlockResidual::from_primitive(setup_block_local_integer_array::<T>().as_view())
        .unwrap()
        .into_array();
    let next_index = AtomicUsize::new(0);

    bencher
        .with_inputs(|| {
            (
                &encoded,
                SESSION.create_execution_ctx(),
                next_index.fetch_add(2_654_435_761, Ordering::Relaxed) % encoded.len(),
            )
        })
        .bench_values(|(array, mut ctx, index)| array.execute_scalar(index, &mut ctx).unwrap());
}

#[divan::bench(name = "block_residual_slice_patched_u32")]
fn bench_block_residual_slice_patched_u32(bencher: Bencher) {
    const SLICE_LEN: usize = 100;

    let encoded = BlockResidual::from_primitive(setup_patch_density_u32_array(16).as_view())
        .unwrap()
        .into_array();
    let next_index = AtomicUsize::new(0);

    bencher
        .with_inputs(|| {
            let start = next_index.fetch_add(2_654_435_761, Ordering::Relaxed)
                % (encoded.len() - SLICE_LEN);
            (&encoded, start)
        })
        .bench_values(|(array, start)| array.slice(start..start + SLICE_LEN).unwrap());
}

#[divan::bench(name = "block_local_for_bitpacked_compress_u64")]
fn bench_block_local_for_bitpacked_compress_u64(bencher: Bencher) {
    let array = setup_block_local_u64_array();

    with_byte_counter(bencher, NUM_VALUES * 8)
        .with_inputs(|| &array)
        .bench_refs(|array| encode_for_bitpacked_tree(array, 31));
}

#[divan::bench(name = "block_local_for_bitpacked_decompress_u64")]
fn bench_block_local_for_bitpacked_decompress_u64(bencher: Bencher) {
    let encoded = encode_for_bitpacked_tree(&setup_block_local_u64_array(), 31);

    with_byte_counter(bencher, NUM_VALUES * 8)
        .with_inputs(|| (&encoded, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| canonicalize((**array).clone(), ctx));
}

#[divan::bench(name = "block_local_for_bitpacked_scalar_at_u64")]
fn bench_block_local_for_bitpacked_scalar_at_u64(bencher: Bencher) {
    let encoded = encode_for_bitpacked_tree(&setup_block_local_u64_array(), 31);
    let next_index = AtomicUsize::new(0);

    bencher
        .with_inputs(|| {
            (
                &encoded,
                SESSION.create_execution_ctx(),
                next_index.fetch_add(2_654_435_761, Ordering::Relaxed) % encoded.len(),
            )
        })
        .bench_values(|(array, mut ctx, index)| array.execute_scalar(index, &mut ctx).unwrap());
}

#[divan::bench(name = "block_local_for_bitpacked_compress_u32")]
fn bench_block_local_for_bitpacked_compress_u32(bencher: Bencher) {
    let array = setup_block_local_u32_array();

    with_byte_counter(bencher, NUM_VALUES * 4)
        .with_inputs(|| &array)
        .bench_refs(|array| encode_for_bitpacked_tree(array, 31));
}

#[divan::bench(name = "block_local_for_bitpacked_decompress_u32")]
fn bench_block_local_for_bitpacked_decompress_u32(bencher: Bencher) {
    let encoded = encode_for_bitpacked_tree(&setup_block_local_u32_array(), 31);

    with_byte_counter(bencher, NUM_VALUES * 4)
        .with_inputs(|| (&encoded, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| canonicalize((**array).clone(), ctx));
}

#[divan::bench(name = "block_local_for_bitpacked_scalar_at_u32")]
fn bench_block_local_for_bitpacked_scalar_at_u32(bencher: Bencher) {
    let encoded = encode_for_bitpacked_tree(&setup_block_local_u32_array(), 31);
    let next_index = AtomicUsize::new(0);

    bencher
        .with_inputs(|| {
            (
                &encoded,
                SESSION.create_execution_ctx(),
                next_index.fetch_add(2_654_435_761, Ordering::Relaxed) % encoded.len(),
            )
        })
        .bench_values(|(array, mut ctx, index)| array.execute_scalar(index, &mut ctx).unwrap());
}

#[divan::bench(name = "block_local_for_bitpacked_compress_i32")]
fn bench_block_local_for_bitpacked_compress_i32(bencher: Bencher) {
    let array = setup_block_local_i32_array();

    with_byte_counter(bencher, NUM_VALUES * 4)
        .with_inputs(|| &array)
        .bench_refs(|array| encode_for_bitpacked_tree(array, 31));
}

#[divan::bench(name = "block_local_for_bitpacked_decompress_i32")]
fn bench_block_local_for_bitpacked_decompress_i32(bencher: Bencher) {
    let encoded = encode_for_bitpacked_tree(&setup_block_local_i32_array(), 31);

    with_byte_counter(bencher, NUM_VALUES * 4)
        .with_inputs(|| (&encoded, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| canonicalize((**array).clone(), ctx));
}

#[divan::bench(name = "block_local_for_bitpacked_scalar_at_i32")]
fn bench_block_local_for_bitpacked_scalar_at_i32(bencher: Bencher) {
    let encoded = encode_for_bitpacked_tree(&setup_block_local_i32_array(), 31);
    let next_index = AtomicUsize::new(0);

    bencher
        .with_inputs(|| {
            (
                &encoded,
                SESSION.create_execution_ctx(),
                next_index.fetch_add(2_654_435_761, Ordering::Relaxed) % encoded.len(),
            )
        })
        .bench_values(|(array, mut ctx, index)| array.execute_scalar(index, &mut ctx).unwrap());
}

#[divan::bench(args = [256, 64, 16, 4, 1])]
fn patch_density_block_residual_decompress_u32(bencher: Bencher, stride: u64) {
    let encoded = BlockResidual::from_primitive(setup_patch_density_u32_array(stride).as_view())
        .unwrap()
        .into_array();

    with_byte_counter(bencher, NUM_VALUES * 4)
        .with_inputs(|| (&encoded, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| canonicalize((**array).clone(), ctx));
}

#[divan::bench(args = [256, 64, 16, 4, 1])]
fn patch_density_block_residual_scalar_at_u32(bencher: Bencher, stride: u64) {
    let encoded = BlockResidual::from_primitive(setup_patch_density_u32_array(stride).as_view())
        .unwrap()
        .into_array();
    let next_index = AtomicUsize::new(0);

    bencher
        .with_inputs(|| {
            (
                &encoded,
                SESSION.create_execution_ctx(),
                next_index.fetch_add(2_654_435_761, Ordering::Relaxed) % encoded.len(),
            )
        })
        .bench_values(|(array, mut ctx, index)| array.execute_scalar(index, &mut ctx).unwrap());
}

#[divan::bench(args = [256, 64, 16, 4, 1])]
fn patch_density_prior_default_compress_u32(bencher: Bencher, stride: u64) {
    let input = setup_patch_density_u32_array(stride);
    let compressor = BtrBlocksCompressorBuilder::default()
        .exclude_schemes([BlockResidualScheme.id()])
        .build();
    bench_compressor(bencher, input, compressor);
}

#[divan::bench(args = [256, 64, 16, 4, 1])]
fn patch_density_default_compress_u32(bencher: Bencher, stride: u64) {
    bench_compressor(
        bencher,
        setup_patch_density_u32_array(stride),
        proposed_default_builder().build(),
    );
}

#[divan::bench(args = [256, 64, 16, 4, 1])]
fn patch_density_prior_default_decompress_u32(bencher: Bencher, stride: u64) {
    let encoded = encode_prior_default(&setup_patch_density_u32_array(stride));

    with_byte_counter(bencher, NUM_VALUES * 4)
        .with_inputs(|| (&encoded, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| canonicalize((**array).clone(), ctx));
}

#[divan::bench(args = [256, 64, 16, 4, 1])]
fn patch_density_default_decompress_u32(bencher: Bencher, stride: u64) {
    let encoded = encode_proposed_default(&setup_patch_density_u32_array(stride));

    with_byte_counter(bencher, NUM_VALUES * 4)
        .with_inputs(|| (&encoded, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| canonicalize((**array).clone(), ctx));
}

#[divan::bench(name = "block_local_for_bitpacked_compress_i16")]
fn bench_block_local_for_bitpacked_compress_i16(bencher: Bencher) {
    let array = setup_block_local_i16_array();

    with_byte_counter(bencher, NUM_VALUES * 2)
        .with_inputs(|| &array)
        .bench_refs(|array| encode_for_bitpacked_tree(array, 14));
}

#[divan::bench(name = "block_local_for_bitpacked_decompress_i16")]
fn bench_block_local_for_bitpacked_decompress_i16(bencher: Bencher) {
    let encoded = encode_for_bitpacked_tree(&setup_block_local_i16_array(), 14);

    with_byte_counter(bencher, NUM_VALUES * 2)
        .with_inputs(|| (&encoded, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| canonicalize((**array).clone(), ctx));
}

#[divan::bench(name = "block_local_for_bitpacked_scalar_at_i16")]
fn bench_block_local_for_bitpacked_scalar_at_i16(bencher: Bencher) {
    let encoded = encode_for_bitpacked_tree(&setup_block_local_i16_array(), 14);
    let next_index = AtomicUsize::new(0);

    bencher
        .with_inputs(|| {
            (
                &encoded,
                SESSION.create_execution_ctx(),
                next_index.fetch_add(2_654_435_761, Ordering::Relaxed) % encoded.len(),
            )
        })
        .bench_values(|(array, mut ctx, index)| array.execute_scalar(index, &mut ctx).unwrap());
}

#[divan::bench(name = "block_local_pcodec_compress_i16")]
fn bench_block_local_pcodec_compress_i16(bencher: Bencher) {
    let array = setup_block_local_i16_array();

    with_byte_counter(bencher, NUM_VALUES * 2)
        .with_inputs(|| (&array, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| {
            Pco::from_primitive(
                array.as_view(),
                PCO_COMPRESSION_LEVEL,
                PCO_VALUES_PER_PAGE,
                ctx,
            )
            .unwrap()
        });
}

#[divan::bench(name = "block_local_pcodec_decompress_i16")]
fn bench_block_local_pcodec_decompress_i16(bencher: Bencher) {
    let array = setup_block_local_i16_array();
    let compressed = Pco::from_primitive(
        array.as_view(),
        PCO_COMPRESSION_LEVEL,
        PCO_VALUES_PER_PAGE,
        &mut SESSION.create_execution_ctx(),
    )
    .unwrap();

    with_byte_counter(bencher, NUM_VALUES * 2)
        .with_inputs(|| (&compressed, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| canonicalize((**array).clone(), ctx));
}

#[divan::bench(name = "ordered_block_residual_compress_f64")]
fn bench_ordered_block_residual_compress_f64(bencher: Bencher) {
    let float_array = setup_random_walk_array();

    with_byte_counter(bencher, NUM_VALUES * 8)
        .with_inputs(|| &float_array)
        .bench_refs(|array| encode_ordered_block_residual(array));
}

#[divan::bench(name = "ordered_block_residual_decompress_f64")]
fn bench_ordered_block_residual_decompress_f64(bencher: Bencher) {
    let float_array = setup_random_walk_array();
    let encoded = encode_ordered_block_residual(&float_array);

    with_byte_counter(bencher, NUM_VALUES * 8)
        .with_inputs(|| (&encoded, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| canonicalize((**array).clone(), ctx));
}

#[divan::bench(name = "ordered_block_residual_scalar_at_f64")]
fn bench_ordered_block_residual_scalar_at_f64(bencher: Bencher) {
    let encoded = encode_ordered_block_residual(&setup_random_walk_array());
    let next_index = AtomicUsize::new(0);

    bencher
        .with_inputs(|| {
            (
                &encoded,
                SESSION.create_execution_ctx(),
                next_index.fetch_add(2_654_435_761, Ordering::Relaxed) % encoded.len(),
            )
        })
        .bench_values(|(array, mut ctx, index)| array.execute_scalar(index, &mut ctx).unwrap());
}

#[divan::bench(name = "ordered_block_residual_prior_default_scalar_at_f64")]
fn bench_ordered_block_residual_prior_default_scalar_at_f64(bencher: Bencher) {
    let encoded = encode_prior_default(&setup_random_walk_array());
    let next_index = AtomicUsize::new(0);

    bencher
        .with_inputs(|| {
            (
                &encoded,
                SESSION.create_execution_ctx(),
                next_index.fetch_add(2_654_435_761, Ordering::Relaxed) % encoded.len(),
            )
        })
        .bench_values(|(array, mut ctx, index)| array.execute_scalar(index, &mut ctx).unwrap());
}

#[divan::bench(name = "ordered_block_residual_compress_f16")]
fn bench_ordered_block_residual_compress_f16(bencher: Bencher) {
    let float_array = setup_ordered_f16_array();

    with_byte_counter(bencher, NUM_VALUES * 2)
        .with_inputs(|| &float_array)
        .bench_refs(|array| encode_ordered_block_residual(array));
}

#[divan::bench(name = "ordered_block_residual_decompress_f16")]
fn bench_ordered_block_residual_decompress_f16(bencher: Bencher) {
    let encoded = encode_ordered_block_residual(&setup_ordered_f16_array());

    with_byte_counter(bencher, NUM_VALUES * 2)
        .with_inputs(|| (&encoded, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| canonicalize((**array).clone(), ctx));
}

#[divan::bench(name = "ordered_block_residual_scalar_at_f16")]
fn bench_ordered_block_residual_scalar_at_f16(bencher: Bencher) {
    let encoded = encode_ordered_block_residual(&setup_ordered_f16_array());
    let next_index = AtomicUsize::new(0);

    bencher
        .with_inputs(|| {
            (
                &encoded,
                SESSION.create_execution_ctx(),
                next_index.fetch_add(2_654_435_761, Ordering::Relaxed) % encoded.len(),
            )
        })
        .bench_values(|(array, mut ctx, index)| array.execute_scalar(index, &mut ctx).unwrap());
}

#[divan::bench(name = "ordered_block_residual_compress_f32")]
fn bench_ordered_block_residual_compress_f32(bencher: Bencher) {
    let float_array = setup_ordered_f32_array();

    with_byte_counter(bencher, NUM_VALUES * 4)
        .with_inputs(|| &float_array)
        .bench_refs(|array| encode_ordered_block_residual(array));
}

#[divan::bench(name = "ordered_block_residual_decompress_f32")]
fn bench_ordered_block_residual_decompress_f32(bencher: Bencher) {
    let encoded = encode_ordered_block_residual(&setup_ordered_f32_array());

    with_byte_counter(bencher, NUM_VALUES * 4)
        .with_inputs(|| (&encoded, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| canonicalize((**array).clone(), ctx));
}

#[divan::bench(name = "ordered_block_residual_scalar_at_f32")]
fn bench_ordered_block_residual_scalar_at_f32(bencher: Bencher) {
    let encoded = encode_ordered_block_residual(&setup_ordered_f32_array());
    let next_index = AtomicUsize::new(0);

    bencher
        .with_inputs(|| {
            (
                &encoded,
                SESSION.create_execution_ctx(),
                next_index.fetch_add(2_654_435_761, Ordering::Relaxed) % encoded.len(),
            )
        })
        .bench_values(|(array, mut ctx, index)| array.execute_scalar(index, &mut ctx).unwrap());
}

#[divan::bench(name = "ordered_block_residual_scheme_compress_f64")]
fn bench_ordered_block_residual_scheme_compress_f64(bencher: Bencher) {
    let float_array = setup_random_walk_array();
    let compressor = BtrBlocksCompressorBuilder::empty()
        .with_new_scheme(&OrderedBlockResidualScheme)
        .build();

    with_byte_counter(bencher, NUM_VALUES * 8)
        .with_inputs(|| {
            (
                float_array.clone().into_array(),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(array, mut ctx)| compressor.compress(&array, &mut ctx).unwrap());
}

#[divan::bench(name = "float_quant_split_compress_f16")]
fn bench_float_quant_split_compress_f16(bencher: Bencher) {
    let float_array = setup_quantized_f16_array();

    with_byte_counter(bencher, NUM_VALUES * 2)
        .with_inputs(|| &float_array)
        .bench_refs(|array| {
            FloatQuant::from_primitive_constant_secondary(array.as_view(), 4).unwrap()
        });
}

#[divan::bench(name = "float_quant_split_decompress_f16")]
fn bench_float_quant_split_decompress_f16(bencher: Bencher) {
    let encoded =
        FloatQuant::from_primitive_constant_secondary(setup_quantized_f16_array().as_view(), 4)
            .unwrap()
            .into_array();

    with_byte_counter(bencher, NUM_VALUES * 2)
        .with_inputs(|| (&encoded, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| canonicalize((**array).clone(), ctx));
}

#[divan::bench(name = "float_quant_split_scalar_at_f16")]
fn bench_float_quant_split_scalar_at_f16(bencher: Bencher) {
    let encoded =
        FloatQuant::from_primitive_constant_secondary(setup_quantized_f16_array().as_view(), 4)
            .unwrap()
            .into_array();
    let next_index = AtomicUsize::new(0);

    bencher
        .with_inputs(|| {
            (
                &encoded,
                SESSION.create_execution_ctx(),
                next_index.fetch_add(2_654_435_761, Ordering::Relaxed) % encoded.len(),
            )
        })
        .bench_values(|(array, mut ctx, index)| array.execute_scalar(index, &mut ctx).unwrap());
}

#[divan::bench(name = "float_quant_scheme_compress_f16")]
fn bench_float_quant_scheme_compress_f16(bencher: Bencher) {
    let compressor = BtrBlocksCompressorBuilder::empty()
        .with_new_scheme(&FloatQuantScheme)
        .build();
    bench_compressor(bencher, setup_quantized_f16_array(), compressor);
}

#[divan::bench(name = "float_quant_tree_decompress_f16")]
fn bench_float_quant_tree_decompress_f16(bencher: Bencher) {
    let encoded = encode_float_quant_scheme_tree(&setup_quantized_f16_array());

    with_byte_counter(bencher, NUM_VALUES * 2)
        .with_inputs(|| (&encoded, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| canonicalize((**array).clone(), ctx));
}

#[divan::bench(name = "float_quant_tree_scalar_at_f16")]
fn bench_float_quant_tree_scalar_at_f16(bencher: Bencher) {
    let encoded = encode_float_quant_scheme_tree(&setup_quantized_f16_array());
    let next_index = AtomicUsize::new(0);

    bencher
        .with_inputs(|| {
            (
                &encoded,
                SESSION.create_execution_ctx(),
                next_index.fetch_add(2_654_435_761, Ordering::Relaxed) % encoded.len(),
            )
        })
        .bench_values(|(array, mut ctx, index)| array.execute_scalar(index, &mut ctx).unwrap());
}

#[divan::bench(name = "float_quant_prior_default_compress_f16")]
fn bench_float_quant_prior_default_compress_f16(bencher: Bencher) {
    let compressor = BtrBlocksCompressorBuilder::default()
        .exclude_schemes([
            FloatQuantScheme.id(),
            OrderedBlockResidualScheme.id(),
            BlockResidualScheme.id(),
        ])
        .build();
    bench_compressor(bencher, setup_quantized_f16_array(), compressor);
}

#[divan::bench(name = "float_quant_proposed_default_compress_f16")]
fn bench_float_quant_proposed_default_compress_f16(bencher: Bencher) {
    bench_compressor(
        bencher,
        setup_quantized_f16_array(),
        proposed_default_builder().build(),
    );
}

#[divan::bench(name = "float_quant_prior_default_decompress_f16")]
fn bench_float_quant_prior_default_decompress_f16(bencher: Bencher) {
    let encoded = encode_prior_default(&setup_quantized_f16_array());

    with_byte_counter(bencher, NUM_VALUES * 2)
        .with_inputs(|| (&encoded, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| canonicalize((**array).clone(), ctx));
}

#[divan::bench(name = "float_quant_proposed_default_decompress_f16")]
fn bench_float_quant_proposed_default_decompress_f16(bencher: Bencher) {
    let encoded = encode_proposed_default(&setup_quantized_f16_array());

    with_byte_counter(bencher, NUM_VALUES * 2)
        .with_inputs(|| (&encoded, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| canonicalize((**array).clone(), ctx));
}

#[divan::bench(name = "float_quant_prior_default_reject_f16")]
fn bench_float_quant_prior_default_reject_f16(bencher: Bencher) {
    let compressor = BtrBlocksCompressorBuilder::default()
        .exclude_schemes([
            FloatQuantScheme.id(),
            OrderedBlockResidualScheme.id(),
            BlockResidualScheme.id(),
        ])
        .build();
    bench_compressor(bencher, setup_general_f16_array(), compressor);
}

#[divan::bench(name = "float_quant_proposed_default_reject_f16")]
fn bench_float_quant_proposed_default_reject_f16(bencher: Bencher) {
    bench_compressor(
        bencher,
        setup_general_f16_array(),
        proposed_default_builder().build(),
    );
}

macro_rules! float_quant_rejection_benches {
    ($scheme_only:ident, $without_scheme:ident, $with_scheme:ident, $setup:ident) => {
        #[divan::bench]
        fn $scheme_only(bencher: Bencher) {
            let compressor = BtrBlocksCompressorBuilder::empty()
                .with_new_scheme(&FloatQuantScheme)
                .build();
            bench_compressor(bencher, $setup(), compressor);
        }

        #[divan::bench]
        fn $without_scheme(bencher: Bencher) {
            let compressor = BtrBlocksCompressorBuilder::default()
                .exclude_schemes([FloatQuantScheme.id()])
                .build();
            bench_compressor(bencher, $setup(), compressor);
        }

        #[divan::bench]
        fn $with_scheme(bencher: Bencher) {
            bench_compressor(bencher, $setup(), default_with_float_quant().build());
        }
    };
}

float_quant_rejection_benches!(
    float_quant_scheme_reject_f16,
    float_quant_default_without_scheme_reject_f16,
    float_quant_default_with_scheme_reject_f16,
    setup_general_f16_array
);

float_quant_rejection_benches!(
    float_quant_scheme_reject_f32,
    float_quant_default_without_scheme_reject_f32,
    float_quant_default_with_scheme_reject_f32,
    setup_general_f32_array
);

float_quant_rejection_benches!(
    float_quant_scheme_reject_f64,
    float_quant_default_without_scheme_reject_f64,
    float_quant_default_with_scheme_reject_f64,
    setup_general_f64_array
);

float_quant_rejection_benches!(
    float_quant_scheme_near_miss_f32,
    float_quant_default_without_scheme_near_miss_f32,
    float_quant_default_with_scheme_near_miss_f32,
    setup_float_quant_near_miss_f32_array
);

float_quant_rejection_benches!(
    float_quant_scheme_near_miss_f64,
    float_quant_default_without_scheme_near_miss_f64,
    float_quant_default_with_scheme_near_miss_f64,
    setup_float_quant_near_miss_f64_array
);

#[divan::bench(name = "float_quant_split_compress_f64")]
fn bench_float_quant_split_compress_f64(bencher: Bencher) {
    let float_array = setup_widened_f32_array();

    with_byte_counter(bencher, NUM_VALUES * 8)
        .with_inputs(|| &float_array)
        .bench_refs(|array| {
            FloatQuant::from_primitive_constant_secondary(array.as_view(), 29).unwrap()
        });
}

#[divan::bench(name = "float_quant_split_decompress_f64")]
fn bench_float_quant_split_decompress_f64(bencher: Bencher) {
    let float_array = setup_widened_f32_array();
    let encoded = FloatQuant::from_primitive_constant_secondary(float_array.as_view(), 29)
        .unwrap()
        .into_array();

    with_byte_counter(bencher, NUM_VALUES * 8)
        .with_inputs(|| (&encoded, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| canonicalize((**array).clone(), ctx));
}

#[divan::bench(name = "float_quant_split_scalar_at_f64")]
fn bench_float_quant_split_scalar_at_f64(bencher: Bencher) {
    let encoded =
        FloatQuant::from_primitive_constant_secondary(setup_widened_f32_array().as_view(), 29)
            .unwrap()
            .into_array();
    let next_index = AtomicUsize::new(0);

    bencher
        .with_inputs(|| {
            (
                &encoded,
                SESSION.create_execution_ctx(),
                next_index.fetch_add(2_654_435_761, Ordering::Relaxed) % encoded.len(),
            )
        })
        .bench_values(|(array, mut ctx, index)| array.execute_scalar(index, &mut ctx).unwrap());
}

#[divan::bench(name = "float_quant_materialized_tree_compress_f64")]
fn bench_float_quant_tree_compress_f64(bencher: Bencher) {
    let float_array = setup_widened_f32_array();

    with_byte_counter(bencher, NUM_VALUES * 8)
        .with_inputs(|| &float_array)
        .bench_refs(|array| encode_float_quant_tree(array));
}

#[divan::bench(name = "float_quant_scheme_compress_f64")]
fn bench_float_quant_scheme_compress_f64(bencher: Bencher) {
    let compressor = BtrBlocksCompressorBuilder::empty()
        .with_new_scheme(&FloatQuantScheme)
        .build();
    bench_compressor(bencher, setup_widened_f32_array(), compressor);
}

#[divan::bench(name = "float_quant_tree_decompress_f64")]
fn bench_float_quant_tree_decompress_f64(bencher: Bencher) {
    let float_array = setup_widened_f32_array();
    let encoded = encode_float_quant_tree(&float_array);

    with_byte_counter(bencher, NUM_VALUES * 8)
        .with_inputs(|| (&encoded, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| canonicalize((**array).clone(), ctx));
}

#[divan::bench(name = "float_quant_tree_scalar_at_f64")]
fn bench_float_quant_tree_scalar_at_f64(bencher: Bencher) {
    let encoded = encode_float_quant_tree(&setup_widened_f32_array());
    let next_index = AtomicUsize::new(0);

    bencher
        .with_inputs(|| {
            (
                &encoded,
                SESSION.create_execution_ctx(),
                next_index.fetch_add(2_654_435_761, Ordering::Relaxed) % encoded.len(),
            )
        })
        .bench_values(|(array, mut ctx, index)| array.execute_scalar(index, &mut ctx).unwrap());
}

#[divan::bench(name = "float_quant_split_compress_f32")]
fn bench_float_quant_split_compress_f32(bencher: Bencher) {
    let float_array = setup_quantized_f32_array();
    let k = analyze_float_quant(float_array.as_view()).unwrap().k;

    with_byte_counter(bencher, NUM_VALUES * 4)
        .with_inputs(|| &float_array)
        .bench_refs(|array| {
            FloatQuant::from_primitive_constant_secondary(array.as_view(), k).unwrap()
        });
}

#[divan::bench(name = "float_quant_split_decompress_f32")]
fn bench_float_quant_split_decompress_f32(bencher: Bencher) {
    let float_array = setup_quantized_f32_array();
    let k = analyze_float_quant(float_array.as_view()).unwrap().k;
    let encoded = FloatQuant::from_primitive_constant_secondary(float_array.as_view(), k)
        .unwrap()
        .into_array();

    with_byte_counter(bencher, NUM_VALUES * 4)
        .with_inputs(|| (&encoded, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| canonicalize((**array).clone(), ctx));
}

#[divan::bench(name = "float_quant_split_scalar_at_f32")]
fn bench_float_quant_split_scalar_at_f32(bencher: Bencher) {
    let float_array = setup_quantized_f32_array();
    let k = analyze_float_quant(float_array.as_view()).unwrap().k;
    let encoded = FloatQuant::from_primitive_constant_secondary(float_array.as_view(), k)
        .unwrap()
        .into_array();
    let next_index = AtomicUsize::new(0);

    bencher
        .with_inputs(|| {
            (
                &encoded,
                SESSION.create_execution_ctx(),
                next_index.fetch_add(2_654_435_761, Ordering::Relaxed) % encoded.len(),
            )
        })
        .bench_values(|(array, mut ctx, index)| array.execute_scalar(index, &mut ctx).unwrap());
}

#[divan::bench(name = "float_quant_materialized_tree_compress_f32")]
fn bench_float_quant_tree_compress_f32(bencher: Bencher) {
    let float_array = setup_quantized_f32_array();

    with_byte_counter(bencher, NUM_VALUES * 4)
        .with_inputs(|| &float_array)
        .bench_refs(|array| encode_float_quant_tree(array));
}

#[divan::bench(name = "float_quant_alp_rd_compress_f32")]
fn bench_float_quant_alp_rd_compress_f32(bencher: Bencher) {
    let float_array = setup_quantized_f32_array();

    with_byte_counter(bencher, NUM_VALUES * 4)
        .with_inputs(|| &float_array)
        .bench_refs(|array| {
            let encoder = RDEncoder::new(array.as_slice::<f32>());
            encoder.encode(array.as_view())
        });
}

#[divan::bench(name = "float_quant_tree_decompress_f32")]
fn bench_float_quant_tree_decompress_f32(bencher: Bencher) {
    let encoded = encode_float_quant_tree(&setup_quantized_f32_array());

    with_byte_counter(bencher, NUM_VALUES * 4)
        .with_inputs(|| (&encoded, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| canonicalize((**array).clone(), ctx));
}

#[divan::bench(name = "float_quant_tree_scalar_at_f32")]
fn bench_float_quant_tree_scalar_at_f32(bencher: Bencher) {
    let encoded = encode_float_quant_tree(&setup_quantized_f32_array());
    let next_index = AtomicUsize::new(0);

    bencher
        .with_inputs(|| {
            (
                &encoded,
                SESSION.create_execution_ctx(),
                next_index.fetch_add(2_654_435_761, Ordering::Relaxed) % encoded.len(),
            )
        })
        .bench_values(|(array, mut ctx, index)| array.execute_scalar(index, &mut ctx).unwrap());
}

#[divan::bench(name = "float_quant_analyze_f32")]
fn bench_float_quant_analyze_f32(bencher: Bencher) {
    let float_array = setup_quantized_f32_array();

    with_byte_counter(bencher, NUM_VALUES * 4)
        .with_inputs(|| &float_array)
        .bench_refs(|array| analyze_float_quant(array.as_view()).unwrap());
}

#[divan::bench(name = "float_quant_scheme_compress_f32")]
fn bench_float_quant_scheme_compress_f32(bencher: Bencher) {
    let compressor = BtrBlocksCompressorBuilder::empty()
        .with_new_scheme(&FloatQuantScheme)
        .build();
    bench_compressor(bencher, setup_quantized_f32_array(), compressor);
}

#[divan::bench(name = "float_quant_prior_default_compress_f32")]
fn bench_float_quant_prior_default_compress_f32(bencher: Bencher) {
    let compressor = BtrBlocksCompressorBuilder::default()
        .exclude_schemes([
            FloatQuantScheme.id(),
            OrderedBlockResidualScheme.id(),
            BlockResidualScheme.id(),
        ])
        .build();
    bench_compressor(bencher, setup_quantized_f32_array(), compressor);
}

#[divan::bench(name = "float_quant_default_compress_f32")]
fn bench_float_quant_default_compress_f32(bencher: Bencher) {
    let compressor = default_with_float_quant()
        .exclude_schemes([OrderedBlockResidualScheme.id(), BlockResidualScheme.id()])
        .build();
    bench_compressor(bencher, setup_quantized_f32_array(), compressor);
}

#[divan::bench(name = "float_quant_proposed_default_compress_f32")]
fn bench_float_quant_proposed_default_compress_f32(bencher: Bencher) {
    let compressor = proposed_default_builder().build();
    bench_compressor(bencher, setup_quantized_f32_array(), compressor);
}

#[divan::bench(name = "float_quant_prior_default_reject_f32")]
fn bench_float_quant_prior_default_reject_f32(bencher: Bencher) {
    let compressor = BtrBlocksCompressorBuilder::default()
        .exclude_schemes([
            FloatQuantScheme.id(),
            OrderedBlockResidualScheme.id(),
            BlockResidualScheme.id(),
        ])
        .build();
    bench_compressor(bencher, setup_general_f32_array(), compressor);
}

#[divan::bench(name = "float_quant_proposed_default_reject_f32")]
fn bench_float_quant_proposed_default_reject_f32(bencher: Bencher) {
    let compressor = proposed_default_builder().build();
    bench_compressor(bencher, setup_general_f32_array(), compressor);
}

macro_rules! float_quant_nonzero_secondary_benches {
    (
        $split_compress:ident,
        $split_decompress:ident,
        $split_scalar:ident,
        $scheme_compress:ident,
        $tree_decompress:ident,
        $tree_scalar:ident,
        $setup:ident,
        $byte_width:expr
    ) => {
        #[divan::bench]
        fn $split_compress(bencher: Bencher) {
            let float_array = $setup();
            let k = analyze_float_quant(float_array.as_view()).unwrap().k;

            with_byte_counter(bencher, NUM_VALUES * $byte_width)
                .with_inputs(|| &float_array)
                .bench_refs(|array| FloatQuant::from_primitive(array.as_view(), k).unwrap());
        }

        #[divan::bench]
        fn $split_decompress(bencher: Bencher) {
            let float_array = $setup();
            let k = analyze_float_quant(float_array.as_view()).unwrap().k;
            let encoded = FloatQuant::from_primitive(float_array.as_view(), k)
                .unwrap()
                .into_array();

            with_byte_counter(bencher, NUM_VALUES * $byte_width)
                .with_inputs(|| (&encoded, SESSION.create_execution_ctx()))
                .bench_refs(|(array, ctx)| canonicalize((**array).clone(), ctx));
        }

        #[divan::bench]
        fn $split_scalar(bencher: Bencher) {
            let float_array = $setup();
            let k = analyze_float_quant(float_array.as_view()).unwrap().k;
            let encoded = FloatQuant::from_primitive(float_array.as_view(), k)
                .unwrap()
                .into_array();
            let next_index = AtomicUsize::new(0);

            bencher
                .with_inputs(|| {
                    (
                        &encoded,
                        SESSION.create_execution_ctx(),
                        next_index.fetch_add(2_654_435_761, Ordering::Relaxed) % encoded.len(),
                    )
                })
                .bench_values(|(array, mut ctx, index)| {
                    array.execute_scalar(index, &mut ctx).unwrap()
                });
        }

        #[divan::bench]
        fn $scheme_compress(bencher: Bencher) {
            let compressor = BtrBlocksCompressorBuilder::empty()
                .with_new_scheme(&FloatQuantScheme)
                .build();
            bench_compressor(bencher, $setup(), compressor);
        }

        #[divan::bench]
        fn $tree_decompress(bencher: Bencher) {
            let encoded = encode_float_quant_nonzero_secondary_scheme_tree(&$setup());

            with_byte_counter(bencher, NUM_VALUES * $byte_width)
                .with_inputs(|| (&encoded, SESSION.create_execution_ctx()))
                .bench_refs(|(array, ctx)| canonicalize((**array).clone(), ctx));
        }

        #[divan::bench]
        fn $tree_scalar(bencher: Bencher) {
            let encoded = encode_float_quant_nonzero_secondary_scheme_tree(&$setup());
            let next_index = AtomicUsize::new(0);

            bencher
                .with_inputs(|| {
                    (
                        &encoded,
                        SESSION.create_execution_ctx(),
                        next_index.fetch_add(2_654_435_761, Ordering::Relaxed) % encoded.len(),
                    )
                })
                .bench_values(|(array, mut ctx, index)| {
                    array.execute_scalar(index, &mut ctx).unwrap()
                });
        }
    };
}

float_quant_nonzero_secondary_benches!(
    float_quant_nonzero_secondary_split_compress_f16,
    float_quant_nonzero_secondary_split_decompress_f16,
    float_quant_nonzero_secondary_split_scalar_at_f16,
    float_quant_nonzero_secondary_scheme_compress_f16,
    float_quant_nonzero_secondary_tree_decompress_f16,
    float_quant_nonzero_secondary_tree_scalar_at_f16,
    setup_nonzero_secondary_f16_array,
    2
);

float_quant_nonzero_secondary_benches!(
    float_quant_nonzero_secondary_split_compress_f32,
    float_quant_nonzero_secondary_split_decompress_f32,
    float_quant_nonzero_secondary_split_scalar_at_f32,
    float_quant_nonzero_secondary_scheme_compress_f32,
    float_quant_nonzero_secondary_tree_decompress_f32,
    float_quant_nonzero_secondary_tree_scalar_at_f32,
    setup_nonzero_secondary_f32_array,
    4
);

#[divan::bench(name = "float_quant_nonzero_secondary_split_compress_f64")]
fn bench_float_quant_nonzero_secondary_split_compress_f64(bencher: Bencher) {
    let float_array = setup_nonzero_secondary_array();
    let k = analyze_float_quant(float_array.as_view()).unwrap().k;

    with_byte_counter(bencher, NUM_VALUES * 8)
        .with_inputs(|| &float_array)
        .bench_refs(|array| FloatQuant::from_primitive(array.as_view(), k).unwrap());
}

#[divan::bench(name = "float_quant_nonzero_secondary_split_decompress_f64")]
fn bench_float_quant_nonzero_secondary_split_decompress_f64(bencher: Bencher) {
    let float_array = setup_nonzero_secondary_array();
    let k = analyze_float_quant(float_array.as_view()).unwrap().k;
    let encoded = FloatQuant::from_primitive(float_array.as_view(), k)
        .unwrap()
        .into_array();

    with_byte_counter(bencher, NUM_VALUES * 8)
        .with_inputs(|| (&encoded, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| canonicalize((**array).clone(), ctx));
}

#[divan::bench(name = "float_quant_nonzero_secondary_split_scalar_at_f64")]
fn bench_float_quant_nonzero_secondary_split_scalar_at_f64(bencher: Bencher) {
    let float_array = setup_nonzero_secondary_array();
    let k = analyze_float_quant(float_array.as_view()).unwrap().k;
    let encoded = FloatQuant::from_primitive(float_array.as_view(), k)
        .unwrap()
        .into_array();
    let next_index = AtomicUsize::new(0);

    bencher
        .with_inputs(|| {
            (
                &encoded,
                SESSION.create_execution_ctx(),
                next_index.fetch_add(2_654_435_761, Ordering::Relaxed) % encoded.len(),
            )
        })
        .bench_values(|(array, mut ctx, index)| array.execute_scalar(index, &mut ctx).unwrap());
}

#[divan::bench(name = "float_quant_nonzero_secondary_tree_compress_f64")]
fn bench_float_quant_nonzero_secondary_tree_compress_f64(bencher: Bencher) {
    let float_array = setup_nonzero_secondary_array();

    with_byte_counter(bencher, NUM_VALUES * 8)
        .with_inputs(|| &float_array)
        .bench_refs(|array| encode_float_quant_nonzero_secondary_tree(array));
}

#[divan::bench(name = "float_quant_nonzero_secondary_scheme_compress_f64")]
fn bench_float_quant_nonzero_secondary_scheme_compress_f64(bencher: Bencher) {
    let compressor = BtrBlocksCompressorBuilder::empty()
        .with_new_scheme(&FloatQuantScheme)
        .build();
    bench_compressor(bencher, setup_nonzero_secondary_array(), compressor);
}

#[divan::bench(name = "float_quant_nonzero_secondary_default_compress_f64")]
fn bench_float_quant_nonzero_secondary_default_compress_f64(bencher: Bencher) {
    bench_compressor(
        bencher,
        setup_nonzero_secondary_array(),
        proposed_default_builder().build(),
    );
}

#[divan::bench(name = "float_quant_nonzero_secondary_tree_decompress_f64")]
fn bench_float_quant_nonzero_secondary_tree_decompress_f64(bencher: Bencher) {
    let encoded = encode_float_quant_nonzero_secondary_tree(&setup_nonzero_secondary_array());

    with_byte_counter(bencher, NUM_VALUES * 8)
        .with_inputs(|| (&encoded, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| canonicalize((**array).clone(), ctx));
}

#[divan::bench(name = "float_quant_nonzero_secondary_default_decompress_f64")]
fn bench_float_quant_nonzero_secondary_default_decompress_f64(bencher: Bencher) {
    let input = setup_nonzero_secondary_array().into_array();
    let encoded = proposed_default_builder()
        .build()
        .compress(&input, &mut SESSION.create_execution_ctx())
        .unwrap();
    assert!(encoded.is::<FloatQuant>());

    with_byte_counter(bencher, NUM_VALUES * 8)
        .with_inputs(|| (&encoded, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| canonicalize((**array).clone(), ctx));
}

#[divan::bench(args = [1, 4, 8, 16])]
fn float_quant_secondary_width_decompress_f64(bencher: Bencher, width: u8) {
    let encoded = encode_float_quant_nonzero_secondary_tree(&setup_secondary_width_array(width));

    with_byte_counter(bencher, NUM_VALUES * 8)
        .with_inputs(|| (&encoded, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| canonicalize((**array).clone(), ctx));
}

#[divan::bench(name = "float_quant_nonzero_secondary_tree_scalar_at_f64")]
fn bench_float_quant_nonzero_secondary_tree_scalar_at_f64(bencher: Bencher) {
    let encoded = encode_float_quant_nonzero_secondary_tree(&setup_nonzero_secondary_array());
    let next_index = AtomicUsize::new(0);

    bencher
        .with_inputs(|| {
            (
                &encoded,
                SESSION.create_execution_ctx(),
                next_index.fetch_add(2_654_435_761, Ordering::Relaxed) % encoded.len(),
            )
        })
        .bench_values(|(array, mut ctx, index)| array.execute_scalar(index, &mut ctx).unwrap());
}

#[divan::bench(name = "float_quant_nonzero_secondary_prior_default_compress_f64")]
fn bench_float_quant_nonzero_secondary_prior_default_compress_f64(bencher: Bencher) {
    let float_array = setup_nonzero_secondary_array();

    with_byte_counter(bencher, NUM_VALUES * 8)
        .with_inputs(|| &float_array)
        .bench_refs(|array| encode_prior_default(array));
}

#[divan::bench(name = "float_quant_nonzero_secondary_prior_default_decompress_f64")]
fn bench_float_quant_nonzero_secondary_prior_default_decompress_f64(bencher: Bencher) {
    let encoded = encode_prior_default(&setup_nonzero_secondary_array());

    with_byte_counter(bencher, NUM_VALUES * 8)
        .with_inputs(|| (&encoded, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| canonicalize((**array).clone(), ctx));
}

#[divan::bench(name = "float_quant_nonzero_secondary_prior_default_scalar_at_f64")]
fn bench_float_quant_nonzero_secondary_prior_default_scalar_at_f64(bencher: Bencher) {
    let encoded = encode_prior_default(&setup_nonzero_secondary_array());
    let next_index = AtomicUsize::new(0);

    bencher
        .with_inputs(|| {
            (
                &encoded,
                SESSION.create_execution_ctx(),
                next_index.fetch_add(2_654_435_761, Ordering::Relaxed) % encoded.len(),
            )
        })
        .bench_values(|(array, mut ctx, index)| array.execute_scalar(index, &mut ctx).unwrap());
}

#[divan::bench(name = "float_quant_prior_default_scalar_at_f64")]
fn bench_float_quant_prior_default_scalar_at_f64(bencher: Bencher) {
    let encoded = encode_prior_default(&setup_widened_f32_array());
    let next_index = AtomicUsize::new(0);

    bencher
        .with_inputs(|| {
            (
                &encoded,
                SESSION.create_execution_ctx(),
                next_index.fetch_add(2_654_435_761, Ordering::Relaxed) % encoded.len(),
            )
        })
        .bench_values(|(array, mut ctx, index)| array.execute_scalar(index, &mut ctx).unwrap());
}

#[divan::bench(name = "pcodec_compress_f64")]
fn bench_pcodec_compress_f64(bencher: Bencher) {
    let (_, _, float_array) = setup_primitive_arrays(PCO_NUM_VALUES);

    with_byte_counter(bencher, PCO_NUM_VALUES * 8)
        .with_inputs(|| (&float_array, SESSION.create_execution_ctx()))
        .bench_refs(|(a, ctx)| Pco::from_primitive(a.as_view(), 3, 0, ctx).unwrap());
}

#[divan::bench(name = "pcodec_decompress_f64")]
fn bench_pcodec_decompress_f64(bencher: Bencher) {
    let (_, _, float_array) = setup_primitive_arrays(PCO_NUM_VALUES);
    let compressed = Pco::from_primitive(
        float_array.as_view(),
        3,
        0,
        &mut SESSION.create_execution_ctx(),
    )
    .unwrap();

    with_byte_counter(bencher, PCO_NUM_VALUES * 8)
        .with_inputs(|| (&compressed, SESSION.create_execution_ctx()))
        .bench_refs(|(a, ctx)| canonicalize((**a).clone(), ctx));
}

#[divan::bench(name = "pcodec_compress_widened_f32_f64")]
fn bench_pcodec_compress_widened_f32_f64(bencher: Bencher) {
    let float_array = setup_widened_f32_array();

    with_byte_counter(bencher, NUM_VALUES * 8)
        .with_inputs(|| (&float_array, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| {
            Pco::from_primitive(
                array.as_view(),
                PCO_COMPRESSION_LEVEL,
                PCO_VALUES_PER_PAGE,
                ctx,
            )
            .unwrap()
        });
}

#[divan::bench(name = "pcodec_decompress_widened_f32_f64")]
fn bench_pcodec_decompress_widened_f32_f64(bencher: Bencher) {
    let float_array = setup_widened_f32_array();
    let compressed = Pco::from_primitive(
        float_array.as_view(),
        PCO_COMPRESSION_LEVEL,
        PCO_VALUES_PER_PAGE,
        &mut SESSION.create_execution_ctx(),
    )
    .unwrap();

    with_byte_counter(bencher, NUM_VALUES * 8)
        .with_inputs(|| (&compressed, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| canonicalize((**array).clone(), ctx));
}

#[divan::bench(name = "pcodec_compress_random_walk_f64")]
fn bench_pcodec_compress_random_walk_f64(bencher: Bencher) {
    let float_array = setup_random_walk_array();

    with_byte_counter(bencher, NUM_VALUES * 8)
        .with_inputs(|| (&float_array, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| {
            Pco::from_primitive(
                array.as_view(),
                PCO_COMPRESSION_LEVEL,
                PCO_VALUES_PER_PAGE,
                ctx,
            )
            .unwrap()
        });
}

#[divan::bench(name = "pcodec_decompress_random_walk_f64")]
fn bench_pcodec_decompress_random_walk_f64(bencher: Bencher) {
    let float_array = setup_random_walk_array();
    let compressed = Pco::from_primitive(
        float_array.as_view(),
        PCO_COMPRESSION_LEVEL,
        PCO_VALUES_PER_PAGE,
        &mut SESSION.create_execution_ctx(),
    )
    .unwrap();

    with_byte_counter(bencher, NUM_VALUES * 8)
        .with_inputs(|| (&compressed, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| canonicalize((**array).clone(), ctx));
}

#[cfg(feature = "zstd")]
#[divan::bench(name = "zstd_compress_u32")]
fn bench_zstd_compress_u32(bencher: Bencher) {
    let (uint_array, ..) = setup_primitive_arrays(ZSTD_NUM_VALUES);
    let array = uint_array.into_array();

    with_byte_counter(bencher, ZSTD_NUM_VALUES * 4)
        .with_inputs(|| (array.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(a, mut ctx)| ZstdData::from_array(a, 3, 8192, &mut ctx).unwrap());
}

#[cfg(feature = "zstd")]
#[divan::bench(name = "zstd_decompress_u32")]
fn bench_zstd_decompress_u32(bencher: Bencher) {
    let (uint_array, ..) = setup_primitive_arrays(ZSTD_NUM_VALUES);
    let dtype = uint_array.dtype().clone();
    let validity = uint_array.validity().unwrap();
    let compressed = Zstd::try_new(
        dtype,
        ZstdData::from_array(
            uint_array.into_array(),
            3,
            8192,
            &mut SESSION.create_execution_ctx(),
        )
        .unwrap(),
        validity,
    )
    .unwrap()
    .into_array();

    with_byte_counter(bencher, ZSTD_NUM_VALUES * 4)
        .with_inputs(|| (&compressed, SESSION.create_execution_ctx()))
        .bench_refs(|(a, ctx)| canonicalize((**a).clone(), ctx));
}

// String compression benchmarks
#[divan::bench(name = "dict_compress_string")]
fn bench_dict_compress_string(bencher: Bencher) {
    let varbinview_arr =
        VarBinViewArray::from_iter_str(gen_varbin_words(STRING_NUM_VALUES, STRING_UNIQUENESS));
    let nbytes = varbinview_arr.nbytes();
    let array = varbinview_arr.into_array();

    with_byte_counter(bencher, nbytes)
        .with_inputs(|| (&array, SESSION.create_execution_ctx()))
        .bench_refs(|(a, ctx)| dict_encode(a, ctx).unwrap());
}

#[divan::bench(name = "dict_decompress_string")]
fn bench_dict_decompress_string(bencher: Bencher) {
    let varbinview_arr =
        VarBinViewArray::from_iter_str(gen_varbin_words(STRING_NUM_VALUES, STRING_UNIQUENESS));
    let dict = dict_encode(
        &varbinview_arr.clone().into_array(),
        &mut SESSION.create_execution_ctx(),
    )
    .unwrap();
    let nbytes = varbinview_arr.into_array().nbytes();

    with_byte_counter(bencher, nbytes)
        .with_inputs(|| (&dict, SESSION.create_execution_ctx()))
        .bench_refs(|(a, ctx)| canonicalize((**a).clone(), ctx));
}

#[divan::bench(name = "fsst_compress_string")]
fn bench_fsst_compress_string(bencher: Bencher) {
    let varbinview_arr =
        VarBinViewArray::from_iter_str(gen_varbin_words(STRING_NUM_VALUES, STRING_UNIQUENESS))
            .into_array();
    let fsst_compressor =
        fsst_train_compressor(&varbinview_arr, &mut SESSION.create_execution_ctx()).unwrap();
    let nbytes = varbinview_arr.nbytes();

    with_byte_counter(bencher, nbytes)
        .with_inputs(|| (&varbinview_arr, SESSION.create_execution_ctx()))
        .bench_refs(|(a, ctx)| fsst_compress(a, &fsst_compressor, ctx).unwrap());
}

#[divan::bench(name = "fsst_decompress_string")]
fn bench_fsst_decompress_string(bencher: Bencher) {
    let varbinview_arr =
        VarBinViewArray::from_iter_str(gen_varbin_words(STRING_NUM_VALUES, STRING_UNIQUENESS))
            .into_array();
    let mut ctx = SESSION.create_execution_ctx();
    let fsst_compressor = fsst_train_compressor(&varbinview_arr, &mut ctx).unwrap();
    let fsst_array = fsst_compress(&varbinview_arr, &fsst_compressor, &mut ctx)
        .unwrap()
        .into_array();
    let nbytes = varbinview_arr.nbytes();

    with_byte_counter(bencher, nbytes)
        .with_inputs(|| (&fsst_array, SESSION.create_execution_ctx()))
        .bench_refs(|(a, ctx)| canonicalize(a.clone(), ctx));
}

#[cfg(feature = "zstd")]
#[divan::bench(name = "zstd_compress_string")]
fn bench_zstd_compress_string(bencher: Bencher) {
    let varbinview_arr = VarBinViewArray::from_iter_str(gen_varbin_words(
        ZSTD_STRING_NUM_VALUES,
        ZSTD_STRING_UNIQUENESS,
    ))
    .into_array();
    let nbytes = varbinview_arr.nbytes();

    with_byte_counter(bencher, nbytes)
        .with_inputs(|| (&varbinview_arr, SESSION.create_execution_ctx()))
        .bench_refs(|(a, ctx)| ZstdData::from_array(a.clone(), 3, 8192, ctx).unwrap());
}

#[cfg(feature = "zstd")]
#[divan::bench(name = "zstd_decompress_string")]
fn bench_zstd_decompress_string(bencher: Bencher) {
    let varbinview_arr = VarBinViewArray::from_iter_str(gen_varbin_words(
        ZSTD_STRING_NUM_VALUES,
        ZSTD_STRING_UNIQUENESS,
    ))
    .into_array();
    let dtype = varbinview_arr.dtype().clone();
    let validity = varbinview_arr.validity().unwrap();
    let compressed = Zstd::try_new(
        dtype,
        ZstdData::from_array(
            varbinview_arr.clone().into_array(),
            3,
            8192,
            &mut SESSION.create_execution_ctx(),
        )
        .unwrap(),
        validity,
    )
    .unwrap()
    .into_array();
    let nbytes = varbinview_arr.nbytes();

    with_byte_counter(bencher, nbytes)
        .with_inputs(|| (&compressed, SESSION.create_execution_ctx()))
        .bench_refs(|(a, ctx)| canonicalize(a.clone(), ctx));
}
