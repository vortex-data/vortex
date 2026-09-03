// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

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
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::builders::dict::dict_encode;
use vortex::array::builtins::ArrayBuiltins;
use vortex::array::dtype::Nullability;
use vortex::dtype::PType;
use vortex::encodings::alp::RDEncoder;
use vortex::encodings::alp::RDEncoderExt;
use vortex::encodings::alp::alp_encode;
use vortex::encodings::fastlanes::Delta;
use vortex::encodings::fastlanes::DeltaData;
use vortex::encodings::fastlanes::FoR;
use vortex::encodings::fastlanes::delta_compress;
use vortex::encodings::fsst::fsst_compress;
use vortex::encodings::fsst::fsst_train_compressor;
use vortex::encodings::pco::Pco;
use vortex::encodings::runend::RunEnd;
use vortex::encodings::sequence::sequence_encode;
use vortex::encodings::zigzag::zigzag_encode;
use vortex::encodings::zstd::Zstd;
use vortex::encodings::zstd::ZstdData;
use vortex_array::VortexSessionExecute;
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
