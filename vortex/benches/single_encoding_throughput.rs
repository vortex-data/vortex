// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]

use divan::Bencher;
use mimalloc::MiMalloc;
use rand::{Rng, SeedableRng};
use vortex::arrays::{PrimitiveArray, VarBinViewArray};
use vortex::compressor::BtrBlocksCompressor;
use vortex::compute::cast;
use vortex::encodings::alp::{alp_encode, RDEncoder};
use vortex::encodings::dict::builders::dict_encode;
use vortex::encodings::fastlanes::{bitpack_to_best_bit_width, delta_compress, DeltaArray, FoRArray};
use vortex::encodings::fsst::{fsst_compress, fsst_train_compressor};
use vortex::encodings::runend::RunEndArray;
use vortex::encodings::zigzag::zigzag_encode;
use vortex::validity::Validity;
use vortex::{Array, IntoArray, ToCanonical};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

const NUM_VALUES: u64 = u16::MAX as u64;

#[divan::bench_group]
mod primitive_decompression {
    use super::*;
    use vortex::dtype::PType;

    fn setup_arrays() -> (PrimitiveArray, PrimitiveArray, PrimitiveArray) {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let uint_array = PrimitiveArray::from_iter((0..NUM_VALUES).map(|_| rng.random_range(0u32..256)));
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
        let (uint_array, _, _) = setup_arrays();
        
        bencher
            .counter(NUM_VALUES * 4)
            .bench_local(|| {
                bitpack_to_best_bit_width(&uint_array).unwrap()
            });
    }

    #[divan::bench(name = "bitpacked_decompress")]
    fn bench_bitpacked_decompress(bencher: Bencher) {
        let (uint_array, _, _) = setup_arrays();
        let compressed = bitpack_to_best_bit_width(&uint_array).unwrap();
        
        bencher
            .counter(NUM_VALUES * 4)
            .bench_local(move || {
                compressed.clone().to_canonical().unwrap()
            });
    }

    #[divan::bench(name = "runend_compress")]
    fn bench_runend_compress(bencher: Bencher) {
        let (uint_array, _, _) = setup_arrays();
        
        bencher
            .counter(NUM_VALUES * 4)
            .bench_local(|| {
                RunEndArray::encode(uint_array.clone().into_array()).unwrap()
            });
    }

    #[divan::bench(name = "runend_decompress")]
    fn bench_runend_decompress(bencher: Bencher) {
        let (uint_array, _, _) = setup_arrays();
        let compressed = RunEndArray::encode(uint_array.clone().into_array()).unwrap();
        
        bencher
            .counter(NUM_VALUES * 4)
            .bench_local(move || {
                compressed.clone().to_canonical().unwrap()
            });
    }

    #[divan::bench(name = "delta_compress")]
    fn bench_delta_compress(bencher: Bencher) {
        let (uint_array, _, _) = setup_arrays();
        
        bencher
            .counter(NUM_VALUES * 4)
            .bench_local(|| {
                let (bases, deltas) = delta_compress(&uint_array).unwrap();
                DeltaArray::try_from_delta_compress_parts(
                    bases.into_array(),
                    deltas.into_array(),
                    Validity::NonNullable
                ).unwrap()
            });
    }

    #[divan::bench(name = "delta_decompress")]
    fn bench_delta_decompress(bencher: Bencher) {
        let (uint_array, _, _) = setup_arrays();
        let (bases, deltas) = delta_compress(&uint_array).unwrap();
        let compressed = DeltaArray::try_from_delta_compress_parts(
            bases.into_array(),
            deltas.into_array(),
            Validity::NonNullable
        ).unwrap();
        
        bencher
            .counter(NUM_VALUES * 4)
            .bench_local(move || {
                compressed.clone().to_canonical().unwrap()
            });
    }

    #[divan::bench(name = "dict_compress")]
    fn bench_dict_compress(bencher: Bencher) {
        let (uint_array, _, _) = setup_arrays();
        
        bencher
            .counter(NUM_VALUES * 4)
            .bench_local(|| {
                dict_encode(uint_array.as_ref()).unwrap()
            });
    }

    #[divan::bench(name = "dict_decompress")]
    fn bench_dict_decompress(bencher: Bencher) {
        let (uint_array, _, _) = setup_arrays();
        let compressed = dict_encode(uint_array.as_ref()).unwrap();
        
        bencher
            .counter(NUM_VALUES * 4)
            .bench_local(move || {
                compressed.clone().to_canonical().unwrap()
            });
    }

    #[divan::bench(name = "frame_of_reference_compress")]
    fn bench_for_compress(bencher: Bencher) {
        let (_, int_array, _) = setup_arrays();
        
        bencher
            .counter(NUM_VALUES * 4)
            .bench_local(|| {
                FoRArray::encode(int_array.clone()).unwrap()
            });
    }

    #[divan::bench(name = "frame_of_reference_decompress")]
    fn bench_for_decompress(bencher: Bencher) {
        let (_, int_array, _) = setup_arrays();
        let compressed = FoRArray::encode(int_array.clone()).unwrap();
        
        bencher
            .counter(NUM_VALUES * 4)
            .bench_local(move || {
                compressed.clone().to_canonical().unwrap()
            });
    }

    #[divan::bench(name = "zigzag_compress")]
    fn bench_zigzag_compress(bencher: Bencher) {
        let (_, int_array, _) = setup_arrays();
        
        bencher
            .counter(NUM_VALUES * 4)
            .bench_local(|| {
                zigzag_encode(int_array.clone()).unwrap()
            });
    }

    #[divan::bench(name = "zigzag_decompress")]
    fn bench_zigzag_decompress(bencher: Bencher) {
        let (_, int_array, _) = setup_arrays();
        let compressed = zigzag_encode(int_array.clone()).unwrap();
        
        bencher
            .counter(NUM_VALUES * 4)
            .bench_local(move || {
                compressed.clone().to_canonical().unwrap()
            });
    }

    #[divan::bench(name = "alp_compress")]
    fn bench_alp_compress(bencher: Bencher) {
        let (_, _, float_array) = setup_arrays();
        
        bencher
            .counter(NUM_VALUES * 4)
            .bench_local(|| {
                alp_encode(&float_array, None).unwrap()
            });
    }

    #[divan::bench(name = "alp_decompress")]
    fn bench_alp_decompress(bencher: Bencher) {
        let (_, _, float_array) = setup_arrays();
        let compressed = alp_encode(&float_array, None).unwrap();
        
        bencher
            .counter(NUM_VALUES * 4)
            .bench_local(move || {
                compressed.clone().to_canonical().unwrap()
            });
    }

    #[divan::bench(name = "alp_rd_compress")]
    fn bench_alp_rd_compress(bencher: Bencher) {
        let (_, _, float_array) = setup_arrays();
        
        bencher
            .counter(NUM_VALUES * 4)
            .bench_local(|| {
                let encoder = RDEncoder::new(float_array.as_slice::<f32>());
                encoder.encode(&float_array)
            });
    }

    #[divan::bench(name = "alp_rd_decompress")]
    fn bench_alp_rd_decompress(bencher: Bencher) {
        let (_, _, float_array) = setup_arrays();
        let encoder = RDEncoder::new(float_array.as_slice::<f32>());
        let compressed = encoder.encode(&float_array);
        
        bencher
            .counter(NUM_VALUES * 4)
            .bench_local(move || {
                compressed.clone().to_canonical().unwrap()
            });
    }

    #[divan::bench(name = "btrblocks_compress")]
    fn bench_btrblocks_compress(bencher: Bencher) {
        let (uint_array, _, _) = setup_arrays();
        
        bencher
            .counter(NUM_VALUES * 4)
            .bench_local(|| {
                BtrBlocksCompressor.compress(uint_array.as_ref()).unwrap()
            });
    }

    #[divan::bench(name = "btrblocks_decompress")]
    fn bench_btrblocks_decompress(bencher: Bencher) {
        let (uint_array, _, _) = setup_arrays();
        let compressed = BtrBlocksCompressor.compress(uint_array.as_ref()).unwrap();
        
        bencher
            .counter(NUM_VALUES * 4)
            .bench_local(move || {
                compressed.clone().to_canonical().unwrap()
            });
    }
}

#[divan::bench_group]
mod string_decompression {
    use super::*;
    use rand::prelude::IndexedRandom;

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
        let nbytes = varbinview_arr.clone().into_array().nbytes() as u64;
        
        bencher
            .counter(nbytes)
            .bench_local(move || {
                dict.clone().to_canonical().unwrap()
            });
    }

    #[divan::bench(name = "fsst_decompress_varbinview")]
    fn bench_fsst_decompress_varbinview(bencher: Bencher) {
        let varbinview_arr = VarBinViewArray::from_iter_str(gen_varbin_words(1_000_000, 0.00005));
        let fsst_compressor = fsst_train_compressor(&varbinview_arr.clone().into_array()).unwrap();
        let fsst_array = fsst_compress(&varbinview_arr.clone().into_array(), &fsst_compressor).unwrap();
        let nbytes = varbinview_arr.clone().into_array().nbytes() as u64;
        
        bencher
            .counter(nbytes)
            .bench_local(move || {
                fsst_array.clone().to_canonical().unwrap()
            });
    }
}