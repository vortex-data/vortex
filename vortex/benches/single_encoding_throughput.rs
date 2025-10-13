// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]
#![allow(unexpected_cfgs)]

use divan::Bencher;
#[cfg(not(codspeed))]
use divan::counter::BytesCount;
use mimalloc::MiMalloc;
use rand::prelude::IndexedRandom;
use rand::{Rng, SeedableRng};
use vortex::arrays::{PrimitiveArray, VarBinViewArray};
use vortex::compute::cast;
use vortex::dtype::PType;
use vortex::encodings::alp::{RDEncoder, alp_encode};
use vortex::encodings::dict::builders::dict_encode;
use vortex::encodings::fastlanes::{DeltaArray, FoRArray, delta_compress};
use vortex::encodings::fsst::{fsst_compress, fsst_train_compressor};
use vortex::encodings::pco::PcoArray;
use vortex::encodings::runend::RunEndArray;
use vortex::encodings::zigzag::zigzag_encode;
use vortex::encodings::zstd::ZstdArray;
use vortex::{IntoArray, ToCanonical};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

const NUM_VALUES: u64 = 1_000_000;

// Helper macro to conditionally add counter based on codspeed cfg
macro_rules! with_counter {
    ($bencher:expr, $bytes:expr) => {{
        #[cfg(not(codspeed))]
        let bencher = $bencher.counter(BytesCount::new($bytes));
        #[cfg(codspeed)]
        let bencher = {
            let _ = $bytes; // Consume the bytes value to avoid unused variable warning
            $bencher
        };
        bencher
    }};
}

// Setup functions
fn setup_primitive_arrays() -> (PrimitiveArray, PrimitiveArray, PrimitiveArray) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    let uint_array =
        PrimitiveArray::from_iter((0..NUM_VALUES).map(|_| rng.random_range(42u32..256)));
    let int_array = cast(uint_array.as_ref(), PType::I32.into())
        .unwrap()
        .to_primitive();
    let float_array = cast(uint_array.as_ref(), PType::F64.into())
        .unwrap()
        .to_primitive();
    (uint_array, int_array, float_array)
}

#[allow(clippy::cast_possible_truncation)]
fn gen_varbin_words(len: usize, uniqueness: f64) -> Vec<String> {
    let mut rng = rand::rng();
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
    use vortex::encodings::fastlanes::bitpack_encode_unchecked;

    let (uint_array, ..) = setup_primitive_arrays();
    let bit_width = 8;

    with_counter!(bencher, NUM_VALUES * 4)
        .with_inputs(|| uint_array.clone())
        .bench_values(|a| unsafe { bitpack_encode_unchecked(a, bit_width).unwrap() });
}

#[divan::bench(name = "bitpacked_decompress_u32")]
fn bench_bitpacked_decompress_u32(bencher: Bencher) {
    use vortex::encodings::fastlanes::bitpack_encode;

    let (uint_array, ..) = setup_primitive_arrays();
    let bit_width = 8;
    let compressed = bitpack_encode(&uint_array, bit_width, None).unwrap();

    with_counter!(bencher, NUM_VALUES * 4)
        .with_inputs(|| compressed.clone())
        .bench_values(|a| a.to_canonical());
}

#[divan::bench(name = "runend_compress_u32")]
fn bench_runend_compress_u32(bencher: Bencher) {
    let (uint_array, ..) = setup_primitive_arrays();

    with_counter!(bencher, NUM_VALUES * 4)
        .with_inputs(|| uint_array.clone())
        .bench_values(|a| RunEndArray::encode(a.into_array()).unwrap());
}

#[divan::bench(name = "runend_decompress_u32")]
fn bench_runend_decompress_u32(bencher: Bencher) {
    let (uint_array, ..) = setup_primitive_arrays();
    let compressed = RunEndArray::encode(uint_array.into_array()).unwrap();

    with_counter!(bencher, NUM_VALUES * 4)
        .with_inputs(|| compressed.clone())
        .bench_values(|a| a.to_canonical());
}

#[divan::bench(name = "delta_compress_u32")]
fn bench_delta_compress_u32(bencher: Bencher) {
    let (uint_array, ..) = setup_primitive_arrays();

    with_counter!(bencher, NUM_VALUES * 4)
        .with_inputs(|| uint_array.clone())
        .bench_values(|a| {
            let (bases, deltas) = delta_compress(&a).unwrap();
            DeltaArray::try_from_delta_compress_parts(bases.into_array(), deltas.into_array())
                .unwrap()
        });
}

#[divan::bench(name = "delta_decompress_u32")]
fn bench_delta_decompress_u32(bencher: Bencher) {
    let (uint_array, ..) = setup_primitive_arrays();
    let (bases, deltas) = delta_compress(&uint_array).unwrap();
    let compressed =
        DeltaArray::try_from_delta_compress_parts(bases.into_array(), deltas.into_array()).unwrap();

    with_counter!(bencher, NUM_VALUES * 4)
        .with_inputs(|| compressed.clone())
        .bench_values(|a| a.to_canonical());
}

#[divan::bench(name = "for_compress_i32")]
fn bench_for_compress_i32(bencher: Bencher) {
    let (_, int_array, _) = setup_primitive_arrays();

    with_counter!(bencher, NUM_VALUES * 4)
        .with_inputs(|| int_array.clone())
        .bench_values(|a| FoRArray::encode(a).unwrap());
}

#[divan::bench(name = "for_decompress_i32")]
fn bench_for_decompress_i32(bencher: Bencher) {
    let (_, int_array, _) = setup_primitive_arrays();
    let compressed = FoRArray::encode(int_array).unwrap();

    with_counter!(bencher, NUM_VALUES * 4)
        .with_inputs(|| compressed.clone())
        .bench_values(|a| a.to_canonical());
}

#[divan::bench(name = "dict_compress_u32")]
fn bench_dict_compress_u32(bencher: Bencher) {
    let (uint_array, ..) = setup_primitive_arrays();

    with_counter!(bencher, NUM_VALUES * 4)
        .with_inputs(|| uint_array.clone())
        .bench_values(|a| dict_encode(a.as_ref()).unwrap());
}

#[divan::bench(name = "dict_decompress_u32")]
fn bench_dict_decompress_u32(bencher: Bencher) {
    let (uint_array, ..) = setup_primitive_arrays();
    let compressed = dict_encode(uint_array.as_ref()).unwrap();

    with_counter!(bencher, NUM_VALUES * 4)
        .with_inputs(|| compressed.clone())
        .bench_values(|a| a.to_canonical());
}

#[divan::bench(name = "zigzag_compress_i32")]
fn bench_zigzag_compress_i32(bencher: Bencher) {
    let (_, int_array, _) = setup_primitive_arrays();

    with_counter!(bencher, NUM_VALUES * 4)
        .with_inputs(|| int_array.clone())
        .bench_values(|a| zigzag_encode(a).unwrap());
}

#[divan::bench(name = "zigzag_decompress_i32")]
fn bench_zigzag_decompress_i32(bencher: Bencher) {
    let (_, int_array, _) = setup_primitive_arrays();
    let compressed = zigzag_encode(int_array).unwrap();

    with_counter!(bencher, NUM_VALUES * 4)
        .with_inputs(|| compressed.clone())
        .bench_values(|a| a.to_canonical());
}

#[divan::bench(name = "alp_compress_f64")]
fn bench_alp_compress_f64(bencher: Bencher) {
    let (_, _, float_array) = setup_primitive_arrays();

    with_counter!(bencher, NUM_VALUES * 8)
        .with_inputs(|| float_array.clone())
        .bench_values(|a| alp_encode(&a, None).unwrap());
}

#[divan::bench(name = "alp_decompress_f64")]
fn bench_alp_decompress_f64(bencher: Bencher) {
    let (_, _, float_array) = setup_primitive_arrays();
    let compressed = alp_encode(&float_array, None).unwrap();

    with_counter!(bencher, NUM_VALUES * 8)
        .with_inputs(|| compressed.clone())
        .bench_values(|a| a.to_canonical());
}

#[divan::bench(name = "alp_rd_compress_f64")]
fn bench_alp_rd_compress_f64(bencher: Bencher) {
    let (_, _, float_array) = setup_primitive_arrays();

    with_counter!(bencher, NUM_VALUES * 8)
        .with_inputs(|| float_array.clone())
        .bench_values(|a| {
            let encoder = RDEncoder::new(a.as_slice::<f64>());
            encoder.encode(&a)
        });
}

#[divan::bench(name = "alp_rd_decompress_f64")]
fn bench_alp_rd_decompress_f64(bencher: Bencher) {
    let (_, _, float_array) = setup_primitive_arrays();
    let encoder = RDEncoder::new(float_array.as_slice::<f64>());
    let compressed = encoder.encode(&float_array);

    with_counter!(bencher, NUM_VALUES * 8)
        .with_inputs(|| compressed.clone())
        .bench_values(|a| a.to_canonical());
}

#[divan::bench(name = "pcodec_compress_f64")]
fn bench_pcodec_compress_f64(bencher: Bencher) {
    let (_, _, float_array) = setup_primitive_arrays();

    with_counter!(bencher, NUM_VALUES * 8)
        .with_inputs(|| float_array.clone())
        .bench_values(|a| PcoArray::from_primitive(&a, 3, 0).unwrap());
}

#[divan::bench(name = "pcodec_decompress_f64")]
fn bench_pcodec_decompress_f64(bencher: Bencher) {
    let (_, _, float_array) = setup_primitive_arrays();
    let compressed = PcoArray::from_primitive(&float_array, 3, 0).unwrap();

    with_counter!(bencher, NUM_VALUES * 8)
        .with_inputs(|| compressed.clone())
        .bench_values(|a| a.to_canonical());
}

#[cfg(feature = "zstd")]
#[divan::bench(name = "zstd_compress_u32")]
fn bench_zstd_compress_u32(bencher: Bencher) {
    let (uint_array, ..) = setup_primitive_arrays();

    with_counter!(bencher, NUM_VALUES * 4)
        .with_inputs(|| uint_array.clone())
        .bench_values(|a| ZstdArray::from_array(a.into_array(), 3, 8192).unwrap());
}

#[cfg(feature = "zstd")]
#[divan::bench(name = "zstd_decompress_u32")]
fn bench_zstd_decompress_u32(bencher: Bencher) {
    let (uint_array, ..) = setup_primitive_arrays();
    let compressed = ZstdArray::from_array(uint_array.into_array(), 3, 8192).unwrap();

    with_counter!(bencher, NUM_VALUES * 4)
        .with_inputs(|| compressed.clone())
        .bench_values(|a| a.to_canonical());
}

// String compression benchmarks
#[divan::bench(name = "dict_compress_string")]
fn bench_dict_compress_string(bencher: Bencher) {
    let varbinview_arr = VarBinViewArray::from_iter_str(gen_varbin_words(1_000_000, 0.00005));
    let nbytes = varbinview_arr.nbytes() as u64;

    with_counter!(bencher, nbytes)
        .with_inputs(|| varbinview_arr.clone())
        .bench_values(|a| dict_encode(a.as_ref()).unwrap());
}

#[divan::bench(name = "dict_decompress_string")]
fn bench_dict_decompress_string(bencher: Bencher) {
    let varbinview_arr = VarBinViewArray::from_iter_str(gen_varbin_words(1_000_000, 0.00005));
    let dict = dict_encode(varbinview_arr.as_ref()).unwrap();
    let nbytes = varbinview_arr.into_array().nbytes() as u64;

    with_counter!(bencher, nbytes)
        .with_inputs(|| dict.clone())
        .bench_values(|a| a.to_canonical());
}

#[divan::bench(name = "fsst_compress_string")]
fn bench_fsst_compress_string(bencher: Bencher) {
    let varbinview_arr = VarBinViewArray::from_iter_str(gen_varbin_words(1_000_000, 0.00005));
    let fsst_compressor = fsst_train_compressor(&varbinview_arr.clone().into_array()).unwrap();
    let nbytes = varbinview_arr.nbytes() as u64;

    with_counter!(bencher, nbytes)
        .with_inputs(|| varbinview_arr.clone())
        .bench_values(|a| fsst_compress(&a.into_array(), &fsst_compressor).unwrap());
}

#[divan::bench(name = "fsst_decompress_string")]
fn bench_fsst_decompress_string(bencher: Bencher) {
    let varbinview_arr = VarBinViewArray::from_iter_str(gen_varbin_words(1_000_000, 0.00005));
    let fsst_compressor = fsst_train_compressor(&varbinview_arr.clone().into_array()).unwrap();
    let fsst_array = fsst_compress(&varbinview_arr.clone().into_array(), &fsst_compressor).unwrap();
    let nbytes = varbinview_arr.into_array().nbytes() as u64;

    with_counter!(bencher, nbytes)
        .with_inputs(|| fsst_array.clone())
        .bench_values(|a| a.to_canonical());
}

#[cfg(feature = "zstd")]
#[divan::bench(name = "zstd_compress_string")]
fn bench_zstd_compress_string(bencher: Bencher) {
    let varbinview_arr = VarBinViewArray::from_iter_str(gen_varbin_words(1_000_000, 0.00005));
    let nbytes = varbinview_arr.nbytes() as u64;

    with_counter!(bencher, nbytes)
        .with_inputs(|| varbinview_arr.clone())
        .bench_values(|a| ZstdArray::from_array(a.into_array(), 3, 8192).unwrap());
}

#[cfg(feature = "zstd")]
#[divan::bench(name = "zstd_decompress_string")]
fn bench_zstd_decompress_string(bencher: Bencher) {
    let varbinview_arr = VarBinViewArray::from_iter_str(gen_varbin_words(1_000_000, 0.00005));
    let compressed = ZstdArray::from_array(varbinview_arr.clone().into_array(), 3, 8192).unwrap();
    let nbytes = varbinview_arr.into_array().nbytes() as u64;

    with_counter!(bencher, nbytes)
        .with_inputs(|| compressed.clone())
        .bench_values(|a| a.to_canonical());
}
