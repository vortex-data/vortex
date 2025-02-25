#![allow(clippy::unwrap_used, clippy::cast_possible_truncation)]
#![allow(unexpected_cfgs)]

use divan::Bencher;
use rand::{Rng, SeedableRng};
use vortex_array::aliases::hash_set::HashSet;
use vortex_array::compute::try_cast;
use vortex_array::{ArrayRef, IntoArray};
use vortex_buffer::Buffer;
use vortex_dtype::PType;
use vortex_error::vortex_panic;
use vortex_sampling_compressor::SamplingCompressor;
use vortex_sampling_compressor::compressors::CompressorRef;
use vortex_sampling_compressor::compressors::alp::ALPCompressor;
use vortex_sampling_compressor::compressors::alp_rd::ALPRDCompressor;
use vortex_sampling_compressor::compressors::bitpacked::{
    BITPACK_NO_PATCHES, BITPACK_WITH_PATCHES,
};
use vortex_sampling_compressor::compressors::delta::DeltaCompressor;
use vortex_sampling_compressor::compressors::dict::DictCompressor;
use vortex_sampling_compressor::compressors::r#for::FoRCompressor;
use vortex_sampling_compressor::compressors::runend::DEFAULT_RUN_END_COMPRESSOR;
use vortex_sampling_compressor::compressors::zigzag::ZigZagCompressor;

fn main() {
    divan::main();
}

const BENCH_ARGS: &[(CompressorRef, PType)] = &[
    (&BITPACK_NO_PATCHES, PType::U32),
    (&BITPACK_WITH_PATCHES, PType::U32),
    (&DEFAULT_RUN_END_COMPRESSOR, PType::U32),
    (&DeltaCompressor, PType::U32),
    (&DictCompressor, PType::U32),
    (&FoRCompressor, PType::I32),
    (&ZigZagCompressor, PType::I32),
    (&ALPCompressor, PType::F32),
    (&ALPRDCompressor, PType::F32),
];

#[divan::bench(args = BENCH_ARGS)]
fn compress(bencher: Bencher, (compressor, array_type): (CompressorRef, PType)) {
    let ctx = SamplingCompressor::new(HashSet::new());
    let array = fixture(array_type);

    bencher
        .with_inputs(|| array.to_array())
        .bench_values(|array| {
            compressor
                .compress(&array, None, ctx.including(compressor))
                .unwrap()
        })
}

#[divan::bench(args = BENCH_ARGS)]
fn decompress(bencher: Bencher, (compressor, ptype): (CompressorRef, PType)) {
    let ctx = SamplingCompressor::new(HashSet::new());

    let compressed = compressor
        .compress(&fixture(ptype), None, ctx.including(compressor))
        .unwrap()
        .into_array();

    bencher
        .with_inputs(|| compressed.clone())
        .bench_values(|compressed| compressed.to_canonical().unwrap())
}

fn fixture(ptype: PType) -> ArrayRef {
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    let uint_array =
        Buffer::from_iter((0..u16::MAX as u64).map(|_| rng.random_range(0u32..256))).into_array();
    let int_array = try_cast(&uint_array, PType::I32.into()).unwrap();
    let float_array = try_cast(&uint_array, PType::F32.into()).unwrap();

    match ptype {
        PType::F32 => float_array,
        PType::I32 => int_array,
        PType::U32 => uint_array,
        _ => vortex_panic!("error: invalid ptype for benchmark: {}", ptype),
    }
}

#[cfg(not(codspeed))]
mod varbinview {
    use rand::distr::Alphanumeric;
    use rand::prelude::IndexedRandom;
    use vortex_array::Array;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::nbytes::NBytes;
    use vortex_dict::builders::dict_encode;
    use vortex_fsst::{fsst_compress, fsst_train_compressor};

    use super::*;

    #[divan::bench]
    fn dict_decode_varbinview(bencher: Bencher) {
        let varbinview_arr = VarBinViewArray::from_iter_str(gen_varbin_words(1_000_000, 0.00005));
        let dict = dict_encode(&varbinview_arr).unwrap();

        bencher
            .with_inputs(|| dict.clone())
            .counter(divan::counter::BytesCount::new(
                varbinview_arr.nbytes() as u64
            ))
            .bench_values(|dict| dict.to_canonical().unwrap())
    }

    #[divan::bench]
    fn fsst_decompress_varbinview(bencher: Bencher) {
        let varbinview_arr = VarBinViewArray::from_iter_str(gen_varbin_words(1_000_000, 0.00005));
        let fsst_compressor = fsst_train_compressor(&varbinview_arr.clone().into_array()).unwrap();
        let fsst_array =
            fsst_compress(&varbinview_arr.clone().into_array(), &fsst_compressor).unwrap();

        bencher
            .with_inputs(|| fsst_array.to_array())
            .counter(divan::counter::BytesCount::new(
                varbinview_arr.into_array().nbytes(),
            ))
            .bench_values(|fsst_array| fsst_array.to_canonical().unwrap())
    }

    fn gen_varbin_words(len: usize, uniqueness: f64) -> Vec<String> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let uniq_cnt = (len as f64 * uniqueness) as usize;
        let dict: Vec<String> = (0..uniq_cnt)
            .map(|_| {
                (&mut rng)
                    .sample_iter(&Alphanumeric)
                    .take(8)
                    .map(char::from)
                    .collect()
            })
            .collect();

        (0..len)
            .map(|_| dict.choose(&mut rng).unwrap().clone())
            .collect()
    }
}
