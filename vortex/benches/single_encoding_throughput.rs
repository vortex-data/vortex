// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]

use divan::{Bencher, counter::BytesCount};
use mimalloc::MiMalloc;
use rand::{Rng, SeedableRng};
use vortex::arrays::{PrimitiveArray, VarBinViewArray};
use vortex::compute::cast;
use vortex::encodings::alp::{RDEncoder, alp_encode};
use vortex::encodings::dict::builders::dict_encode;
use vortex::encodings::fastlanes::{DeltaArray, delta_compress};
use vortex::encodings::fsst::{fsst_compress, fsst_train_compressor};
use vortex::encodings::pco::PcoArray;
use vortex::encodings::runend::RunEndArray;
use vortex::encodings::zigzag::zigzag_encode;
use vortex::encodings::zstd::ZstdArray;
use vortex::validity::Validity;
use vortex::{IntoArray, ToCanonical};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

const NUM_VALUES: u64 = 1_000_000;

#[divan::bench_group]
mod primitive_decompression {
    use super::*;
    use vortex::dtype::PType;

    fn setup_arrays() -> (PrimitiveArray, PrimitiveArray, PrimitiveArray) {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let uint_array =
            PrimitiveArray::from_iter((0..NUM_VALUES).map(|_| rng.random_range(0u32..256)));
        let int_array = cast(uint_array.as_ref(), PType::I32.into())
            .unwrap()
            .to_primitive()
            .unwrap();
        let float_array = cast(uint_array.as_ref(), PType::F32.into())
            .unwrap()
            .to_primitive()
            .unwrap();
        (uint_array, int_array, float_array)
    }

    #[divan::bench(name = "bitpacked_compress")]
    fn bench_bitpacked_compress(bencher: Bencher) {
        use vortex::encodings::fastlanes::bitpack_encode_unchecked;

        let (uint_array, _, _) = setup_arrays();
        let bit_width = 8;

        bencher
            .counter(BytesCount::new(NUM_VALUES * 4))
            .with_inputs(|| uint_array.clone())
            .bench_values(|a| unsafe { bitpack_encode_unchecked(a, bit_width).unwrap() });
    }

    #[divan::bench(name = "bitpacked_decompress")]
    fn bench_bitpacked_decompress(bencher: Bencher) {
        use vortex::encodings::fastlanes::bitpack_encode;

        let (uint_array, _, _) = setup_arrays();
        let bit_width = 8;
        let compressed = bitpack_encode(&uint_array, bit_width, None).unwrap();

        bencher
            .counter(BytesCount::new(NUM_VALUES * 4))
            .with_inputs(|| compressed.clone())
            .bench_values(|a| a.to_canonical().unwrap());
    }

    #[divan::bench(name = "runend_compress")]
    fn bench_runend_compress(bencher: Bencher) {
        let (uint_array, _, _) = setup_arrays();

        bencher
            .counter(BytesCount::new(NUM_VALUES * 4))
            .with_inputs(|| uint_array.clone())
            .bench_values(|a| RunEndArray::encode(a.into_array()).unwrap());
    }

    #[divan::bench(name = "runend_decompress")]
    fn bench_runend_decompress(bencher: Bencher) {
        let (uint_array, _, _) = setup_arrays();
        let compressed = RunEndArray::encode(uint_array.into_array()).unwrap();

        bencher
            .counter(BytesCount::new(NUM_VALUES * 4))
            .with_inputs(|| compressed.clone())
            .bench_values(|a| a.to_canonical().unwrap());
    }

    #[divan::bench(name = "delta_compress")]
    fn bench_delta_compress(bencher: Bencher) {
        let (uint_array, _, _) = setup_arrays();

        bencher
            .counter(BytesCount::new(NUM_VALUES * 4))
            .with_inputs(|| uint_array.clone())
            .bench_values(|a| {
                let (bases, deltas) = delta_compress(&a).unwrap();
                DeltaArray::try_from_delta_compress_parts(
                    bases.into_array(),
                    deltas.into_array(),
                    Validity::NonNullable,
                )
                .unwrap()
            });
    }

    #[divan::bench(name = "delta_decompress")]
    fn bench_delta_decompress(bencher: Bencher) {
        let (uint_array, _, _) = setup_arrays();
        let (bases, deltas) = delta_compress(&uint_array).unwrap();
        let compressed = DeltaArray::try_from_delta_compress_parts(
            bases.into_array(),
            deltas.into_array(),
            Validity::NonNullable,
        )
        .unwrap();

        bencher
            .counter(BytesCount::new(NUM_VALUES * 4))
            .with_inputs(|| compressed.clone())
            .bench_values(|a| a.to_canonical().unwrap());
    }

    #[divan::bench(name = "dict_compress")]
    fn bench_dict_compress(bencher: Bencher) {
        let (uint_array, _, _) = setup_arrays();

        bencher
            .counter(BytesCount::new(NUM_VALUES * 4))
            .with_inputs(|| uint_array.clone())
            .bench_values(|a| dict_encode(a.as_ref()).unwrap());
    }

    #[divan::bench(name = "dict_decompress")]
    fn bench_dict_decompress(bencher: Bencher) {
        let (uint_array, _, _) = setup_arrays();
        let compressed = dict_encode(uint_array.as_ref()).unwrap();

        bencher
            .counter(BytesCount::new(NUM_VALUES * 4))
            .with_inputs(|| compressed.clone())
            .bench_values(|a| a.to_canonical().unwrap());
    }

    #[divan::bench(name = "zigzag_compress")]
    fn bench_zigzag_compress(bencher: Bencher) {
        let (_, int_array, _) = setup_arrays();

        bencher
            .counter(BytesCount::new(NUM_VALUES * 4))
            .with_inputs(|| int_array.clone())
            .bench_values(|a| zigzag_encode(a).unwrap());
    }

    #[divan::bench(name = "zigzag_decompress")]
    fn bench_zigzag_decompress(bencher: Bencher) {
        let (_, int_array, _) = setup_arrays();
        let compressed = zigzag_encode(int_array).unwrap();

        bencher
            .counter(BytesCount::new(NUM_VALUES * 4))
            .with_inputs(|| compressed.clone())
            .bench_values(|a| a.to_canonical().unwrap());
    }

    #[divan::bench(name = "alp_compress")]
    fn bench_alp_compress(bencher: Bencher) {
        let (_, _, float_array) = setup_arrays();

        bencher
            .counter(BytesCount::new(NUM_VALUES * 4))
            .with_inputs(|| float_array.clone())
            .bench_values(|a| alp_encode(&a, None).unwrap());
    }

    #[divan::bench(name = "alp_decompress")]
    fn bench_alp_decompress(bencher: Bencher) {
        let (_, _, float_array) = setup_arrays();
        let compressed = alp_encode(&float_array, None).unwrap();

        bencher
            .counter(BytesCount::new(NUM_VALUES * 4))
            .with_inputs(|| compressed.clone())
            .bench_values(|a| a.to_canonical().unwrap());
    }

    #[divan::bench(name = "alp_rd_compress")]
    fn bench_alp_rd_compress(bencher: Bencher) {
        let (_, _, float_array) = setup_arrays();

        bencher
            .counter(BytesCount::new(NUM_VALUES * 4))
            .with_inputs(|| float_array.clone())
            .bench_values(|a| {
                let encoder = RDEncoder::new(a.as_slice::<f32>());
                encoder.encode(&a)
            });
    }

    #[divan::bench(name = "alp_rd_decompress")]
    fn bench_alp_rd_decompress(bencher: Bencher) {
        let (_, _, float_array) = setup_arrays();
        let encoder = RDEncoder::new(float_array.as_slice::<f32>());
        let compressed = encoder.encode(&float_array);

        bencher
            .counter(BytesCount::new(NUM_VALUES * 4))
            .with_inputs(|| compressed.clone())
            .bench_values(|a| a.to_canonical().unwrap());
    }

    #[divan::bench(name = "pcodec_compress")]
    fn bench_pcodec_compress(bencher: Bencher) {
        let (_, _, float_array) = setup_arrays();

        bencher
            .counter(BytesCount::new(NUM_VALUES * 4))
            .with_inputs(|| float_array.clone())
            .bench_values(|a| PcoArray::from_primitive(&a, 3, 0).unwrap());
    }

    #[divan::bench(name = "pcodec_decompress")]
    fn bench_pcodec_decompress(bencher: Bencher) {
        let (_, _, float_array) = setup_arrays();
        let compressed = PcoArray::from_primitive(&float_array, 3, 0).unwrap();

        bencher
            .counter(BytesCount::new(NUM_VALUES * 4))
            .with_inputs(|| compressed.clone())
            .bench_values(|a| a.to_canonical().unwrap());
    }

    #[divan::bench(name = "zstd_compress")]
    fn bench_zstd_compress(bencher: Bencher) {
        let (uint_array, _, _) = setup_arrays();

        bencher
            .counter(BytesCount::new(NUM_VALUES * 4))
            .with_inputs(|| uint_array.clone())
            .bench_values(|a| ZstdArray::from_array(a.into_array(), 3, 8192).unwrap());
    }

    #[divan::bench(name = "zstd_decompress")]
    fn bench_zstd_decompress(bencher: Bencher) {
        let (uint_array, _, _) = setup_arrays();
        let compressed = ZstdArray::from_array(uint_array.into_array(), 3, 8192).unwrap();

        bencher
            .counter(BytesCount::new(NUM_VALUES * 4))
            .with_inputs(|| compressed.clone())
            .bench_values(|a| a.to_canonical().unwrap());
    }
}

#[divan::bench_group]
mod string_decompression {
    use super::*;
    use rand::prelude::IndexedRandom;

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

    #[divan::bench(name = "dict_decode_varbinview")]
    fn bench_dict_decode_varbinview(bencher: Bencher) {
        let varbinview_arr = VarBinViewArray::from_iter_str(gen_varbin_words(1_000_000, 0.00005));
        let dict = dict_encode(varbinview_arr.as_ref()).unwrap();
        let nbytes = varbinview_arr.into_array().nbytes() as u64;

        bencher
            .counter(BytesCount::new(nbytes))
            .with_inputs(|| dict.clone())
            .bench_values(|a| a.to_canonical().unwrap());
    }

    #[divan::bench(name = "fsst_decompress_varbinview")]
    fn bench_fsst_decompress_varbinview(bencher: Bencher) {
        let varbinview_arr = VarBinViewArray::from_iter_str(gen_varbin_words(1_000_000, 0.00005));
        let fsst_compressor = fsst_train_compressor(&varbinview_arr.clone().into_array()).unwrap();
        let fsst_array =
            fsst_compress(&varbinview_arr.clone().into_array(), &fsst_compressor).unwrap();
        let nbytes = varbinview_arr.into_array().nbytes() as u64;

        bencher
            .counter(BytesCount::new(nbytes))
            .with_inputs(|| fsst_array.clone())
            .bench_values(|a| a.to_canonical().unwrap());
    }
}
