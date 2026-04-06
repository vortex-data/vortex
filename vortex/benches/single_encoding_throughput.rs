// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]
#![allow(clippy::cast_possible_truncation)]
#![allow(unexpected_cfgs)]

use std::sync::LazyLock;

use divan::Bencher;
#[cfg(not(codspeed))]
use divan::counter::BytesCount;
use mimalloc::MiMalloc;
use rand::RngExt;
use rand::SeedableRng;
use rand::prelude::IndexedRandom;
use rand::rngs::StdRng;
use vortex::array::IntoArray;
use vortex::array::ToCanonical;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::builders::dict::dict_encode;
use vortex::array::builtins::ArrayBuiltins;
use vortex::array::dtype::Nullability;
use vortex::array::session::ArraySession;
use vortex::dtype::PType;
use vortex::encodings::alp::RDEncoder;
use vortex::encodings::alp::alp_encode;
use vortex::encodings::fastlanes::DeltaData;
use vortex::encodings::fastlanes::FoR;
use vortex::encodings::fastlanes::delta_compress;
use vortex::encodings::fsst::fsst_compress;
use vortex::encodings::fsst::fsst_train_compressor;
use vortex::encodings::pco::Pco;
use vortex::encodings::runend::RunEnd;
use vortex::encodings::sequence::sequence_encode;
use vortex::encodings::zigzag::zigzag_encode;
use vortex::encodings::zstd::ZstdData;
use vortex_sequence::Sequence;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

fn main() {
    divan::main();
}

const NUM_VALUES: u64 = 100_000;

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

// Setup functions
fn setup_primitive_arrays() -> (PrimitiveArray, PrimitiveArray, PrimitiveArray) {
    let mut rng = StdRng::seed_from_u64(0);
    let uint_array =
        PrimitiveArray::from_iter((0..NUM_VALUES).map(|_| rng.random_range(42u32..256)));
    let int_array = uint_array
        .clone()
        .into_array()
        .cast(PType::I32.into())
        .unwrap()
        .to_primitive();
    let float_array = uint_array
        .clone()
        .into_array()
        .cast(PType::F64.into())
        .unwrap()
        .to_primitive();
    (uint_array, int_array, float_array)
}

#[allow(clippy::cast_possible_truncation)]
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

    let (uint_array, ..) = setup_primitive_arrays();
    let bit_width = 8;

    with_byte_counter(bencher, NUM_VALUES * 4)
        .with_inputs(|| uint_array.clone())
        .bench_values(|a| unsafe { bitpack_encode_unchecked(a, bit_width).unwrap() });
}

#[divan::bench(name = "bitpacked_decompress_u32")]
fn bench_bitpacked_decompress_u32(bencher: Bencher) {
    use vortex::encodings::fastlanes::bitpack_compress::bitpack_encode;

    let (uint_array, ..) = setup_primitive_arrays();
    let bit_width = 8;
    let compressed = bitpack_encode(&uint_array, bit_width, None)
        .unwrap()
        .into_array();

    with_byte_counter(bencher, NUM_VALUES * 4)
        .with_inputs(|| &compressed)
        .bench_refs(|a| a.to_canonical());
}

#[divan::bench(name = "runend_compress_u32")]
fn bench_runend_compress_u32(bencher: Bencher) {
    let (uint_array, ..) = setup_primitive_arrays();

    with_byte_counter(bencher, NUM_VALUES * 4)
        .with_inputs(|| uint_array.clone())
        .bench_values(|a| RunEnd::encode(a.into_array()).unwrap());
}

#[divan::bench(name = "runend_decompress_u32")]
fn bench_runend_decompress_u32(bencher: Bencher) {
    let (uint_array, ..) = setup_primitive_arrays();
    let compressed = RunEnd::encode(uint_array.into_array()).unwrap();

    with_byte_counter(bencher, NUM_VALUES * 4)
        .with_inputs(|| &compressed)
        .bench_refs(|a| a.to_canonical());
}

#[divan::bench(name = "delta_compress_u32")]
fn bench_delta_compress_u32(bencher: Bencher) {
    let (uint_array, ..) = setup_primitive_arrays();

    with_byte_counter(bencher, NUM_VALUES * 4)
        .with_inputs(|| &uint_array)
        .bench_refs(|a| {
            let (bases, deltas) = delta_compress(a, &mut SESSION.create_execution_ctx()).unwrap();
            DeltaData::try_new(bases.into_array(), deltas.into_array(), 0, a.len()).unwrap()
        });
}

#[divan::bench(name = "delta_decompress_u32")]
fn bench_delta_decompress_u32(bencher: Bencher) {
    let (uint_array, ..) = setup_primitive_arrays();
    let (bases, deltas) = delta_compress(&uint_array, &mut SESSION.create_execution_ctx()).unwrap();
    let compressed =
        DeltaData::try_new(bases.into_array(), deltas.into_array(), 0, uint_array.len())
            .unwrap()
            .into_array();

    with_byte_counter(bencher, NUM_VALUES * 4)
        .with_inputs(|| &compressed)
        .bench_refs(|a| a.to_canonical());
}

#[divan::bench(name = "for_compress_i32")]
fn bench_for_compress_i32(bencher: Bencher) {
    let (_, int_array, _) = setup_primitive_arrays();

    with_byte_counter(bencher, NUM_VALUES * 4)
        .with_inputs(|| int_array.clone())
        .bench_values(|a| FoR::encode(a).unwrap());
}

#[divan::bench(name = "for_decompress_i32")]
fn bench_for_decompress_i32(bencher: Bencher) {
    let (_, int_array, _) = setup_primitive_arrays();
    let compressed = FoR::encode(int_array).unwrap();

    with_byte_counter(bencher, NUM_VALUES * 4)
        .with_inputs(|| &compressed)
        .bench_refs(|a| a.to_canonical());
}

#[divan::bench(name = "dict_compress_u32")]
fn bench_dict_compress_u32(bencher: Bencher) {
    let (uint_array, ..) = setup_primitive_arrays();

    with_byte_counter(bencher, NUM_VALUES * 4)
        .with_inputs(|| &uint_array)
        .bench_refs(|a| dict_encode(&a.clone().into_array()).unwrap());
}

#[divan::bench(name = "dict_decompress_u32")]
fn bench_dict_decompress_u32(bencher: Bencher) {
    let (uint_array, ..) = setup_primitive_arrays();
    let compressed = dict_encode(&uint_array.into_array()).unwrap();

    with_byte_counter(bencher, NUM_VALUES * 4)
        .with_inputs(|| &compressed)
        .bench_refs(|a| a.to_canonical());
}

#[divan::bench(name = "zigzag_compress_i32")]
fn bench_zigzag_compress_i32(bencher: Bencher) {
    let (_, int_array, _) = setup_primitive_arrays();

    with_byte_counter(bencher, NUM_VALUES * 4)
        .with_inputs(|| int_array.clone())
        .bench_values(|a| zigzag_encode(a).unwrap());
}

#[divan::bench(name = "zigzag_decompress_i32")]
fn bench_zigzag_decompress_i32(bencher: Bencher) {
    let (_, int_array, _) = setup_primitive_arrays();
    let compressed = zigzag_encode(int_array).unwrap().into_array();

    with_byte_counter(bencher, NUM_VALUES * 4)
        .with_inputs(|| &compressed)
        .bench_refs(|a| a.to_canonical());
}

#[expect(clippy::cast_possible_truncation)]
#[divan::bench(name = "sequence_compress_u32")]
fn bench_sequence_compress_u32(bencher: Bencher) {
    let seq_array = PrimitiveArray::from_iter(0..NUM_VALUES as u32);

    with_byte_counter(bencher, NUM_VALUES * 4)
        .with_inputs(|| seq_array.clone())
        .bench_values(|a| sequence_encode(&a).unwrap().unwrap());
}

#[expect(clippy::cast_possible_truncation)]
#[divan::bench(name = "sequence_decompress_u32")]
fn bench_sequence_decompress_u32(bencher: Bencher) {
    let compressed = Sequence::try_new_typed(0, 1, Nullability::NonNullable, NUM_VALUES as usize)
        .unwrap()
        .into_array();

    with_byte_counter(bencher, NUM_VALUES * 4)
        .with_inputs(|| &compressed)
        .bench_refs(|a| a.to_canonical());
}

#[divan::bench(name = "alp_compress_f64")]
fn bench_alp_compress_f64(bencher: Bencher) {
    let (_, _, float_array) = setup_primitive_arrays();

    with_byte_counter(bencher, NUM_VALUES * 8)
        .with_inputs(|| &float_array)
        .bench_refs(|a| alp_encode(a, None).unwrap());
}

#[divan::bench(name = "alp_decompress_f64")]
fn bench_alp_decompress_f64(bencher: Bencher) {
    let (_, _, float_array) = setup_primitive_arrays();
    let compressed = alp_encode(&float_array, None).unwrap();

    with_byte_counter(bencher, NUM_VALUES * 8)
        .with_inputs(|| &compressed)
        .bench_refs(|a| a.to_canonical());
}

#[divan::bench(name = "alp_rd_compress_f64")]
fn bench_alp_rd_compress_f64(bencher: Bencher) {
    let (_, _, float_array) = setup_primitive_arrays();

    with_byte_counter(bencher, NUM_VALUES * 8)
        .with_inputs(|| &float_array)
        .bench_refs(|a| {
            let encoder = RDEncoder::new(a.as_slice::<f64>());
            encoder.encode(a)
        });
}

#[divan::bench(name = "alp_rd_decompress_f64")]
fn bench_alp_rd_decompress_f64(bencher: Bencher) {
    let (_, _, float_array) = setup_primitive_arrays();
    let encoder = RDEncoder::new(float_array.as_slice::<f64>());
    let compressed = encoder.encode(&float_array);

    with_byte_counter(bencher, NUM_VALUES * 8)
        .with_inputs(|| &compressed)
        .bench_refs(|a| a.to_canonical());
}

#[divan::bench(name = "pcodec_compress_f64")]
fn bench_pcodec_compress_f64(bencher: Bencher) {
    let (_, _, float_array) = setup_primitive_arrays();

    with_byte_counter(bencher, NUM_VALUES * 8)
        .with_inputs(|| &float_array)
        .bench_refs(|a| Pco::from_primitive(a, 3, 0).unwrap());
}

#[divan::bench(name = "pcodec_decompress_f64")]
fn bench_pcodec_decompress_f64(bencher: Bencher) {
    let (_, _, float_array) = setup_primitive_arrays();
    let compressed = Pco::from_primitive(&float_array, 3, 0).unwrap();

    with_byte_counter(bencher, NUM_VALUES * 8)
        .with_inputs(|| &compressed)
        .bench_refs(|a| a.to_canonical());
}

#[cfg(feature = "zstd")]
#[divan::bench(name = "zstd_compress_u32")]
fn bench_zstd_compress_u32(bencher: Bencher) {
    let (uint_array, ..) = setup_primitive_arrays();
    let array = uint_array.into_array();

    with_byte_counter(bencher, NUM_VALUES * 4)
        .with_inputs(|| array.clone())
        .bench_values(|a| ZstdData::from_array(a, 3, 8192).unwrap());
}

#[cfg(feature = "zstd")]
#[divan::bench(name = "zstd_decompress_u32")]
fn bench_zstd_decompress_u32(bencher: Bencher) {
    let (uint_array, ..) = setup_primitive_arrays();
    let compressed = ZstdData::from_array(uint_array.into_array(), 3, 8192)
        .unwrap()
        .into_array();

    with_byte_counter(bencher, NUM_VALUES * 4)
        .with_inputs(|| &compressed)
        .bench_refs(|a| a.to_canonical());
}

// String compression benchmarks
#[divan::bench(name = "dict_compress_string")]
fn bench_dict_compress_string(bencher: Bencher) {
    let varbinview_arr =
        VarBinViewArray::from_iter_str(gen_varbin_words(NUM_VALUES as usize, 0.00005));
    let nbytes = varbinview_arr.nbytes() as u64;

    with_byte_counter(bencher, nbytes)
        .with_inputs(|| &varbinview_arr)
        .bench_refs(|a| dict_encode(&a.clone().into_array()).unwrap());
}

#[divan::bench(name = "dict_decompress_string")]
fn bench_dict_decompress_string(bencher: Bencher) {
    let varbinview_arr =
        VarBinViewArray::from_iter_str(gen_varbin_words(NUM_VALUES as usize, 0.00005));
    let dict = dict_encode(&varbinview_arr.clone().into_array()).unwrap();
    let nbytes = varbinview_arr.into_array().nbytes() as u64;

    with_byte_counter(bencher, nbytes)
        .with_inputs(|| &dict)
        .bench_refs(|a| a.to_canonical());
}

#[divan::bench(name = "fsst_compress_string")]
fn bench_fsst_compress_string(bencher: Bencher) {
    let varbinview_arr =
        VarBinViewArray::from_iter_str(gen_varbin_words(NUM_VALUES as usize, 0.00005));
    let fsst_compressor = fsst_train_compressor(&varbinview_arr);
    let nbytes = varbinview_arr.nbytes() as u64;

    with_byte_counter(bencher, nbytes)
        .with_inputs(|| &varbinview_arr)
        .bench_refs(|a| fsst_compress(*a, a.len(), a.dtype(), &fsst_compressor));
}

#[divan::bench(name = "fsst_decompress_string")]
fn bench_fsst_decompress_string(bencher: Bencher) {
    let varbinview_arr =
        VarBinViewArray::from_iter_str(gen_varbin_words(NUM_VALUES as usize, 0.00005));
    let fsst_compressor = fsst_train_compressor(&varbinview_arr);
    let fsst_array = fsst_compress(
        &varbinview_arr,
        varbinview_arr.len(),
        varbinview_arr.dtype(),
        &fsst_compressor,
    );
    let nbytes = varbinview_arr.into_array().nbytes() as u64;

    with_byte_counter(bencher, nbytes)
        .with_inputs(|| &fsst_array)
        .bench_refs(|a| a.to_canonical());
}

#[cfg(feature = "zstd")]
#[divan::bench(name = "zstd_compress_string")]
fn bench_zstd_compress_string(bencher: Bencher) {
    let varbinview_arr =
        VarBinViewArray::from_iter_str(gen_varbin_words(NUM_VALUES as usize, 0.00005));
    let nbytes = varbinview_arr.nbytes() as u64;
    let array = varbinview_arr.into_array();

    with_byte_counter(bencher, nbytes)
        .with_inputs(|| array.clone())
        .bench_values(|a| ZstdData::from_array(a, 3, 8192).unwrap());
}

#[cfg(feature = "zstd")]
#[divan::bench(name = "zstd_decompress_string")]
fn bench_zstd_decompress_string(bencher: Bencher) {
    let varbinview_arr =
        VarBinViewArray::from_iter_str(gen_varbin_words(NUM_VALUES as usize, 0.00005));
    let compressed = ZstdData::from_array(varbinview_arr.clone().into_array(), 3, 8192)
        .unwrap()
        .into_array();
    let nbytes = varbinview_arr.into_array().nbytes() as u64;

    with_byte_counter(bencher, nbytes)
        .with_inputs(|| &compressed)
        .bench_refs(|a| a.to_canonical());
}
