use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use itertools::Itertools as _;
use mimalloc::MiMalloc;
use rand::{Rng, SeedableRng as _};
use vortex::aliases::hash_set::HashSet;
use vortex::array::PrimitiveArray;
use vortex::compute::unary::try_cast;
use vortex::validity::Validity;
use vortex::{IntoArray as _, IntoCanonical};
use vortex_dtype::PType;
use vortex_sampling_compressor::compressors::alp::ALPCompressor;
use vortex_sampling_compressor::compressors::alp_rd::ALPRDCompressor;
use vortex_sampling_compressor::compressors::bitpacked::{
    BITPACK_NO_PATCHES, BITPACK_WITH_PATCHES,
};
use vortex_sampling_compressor::compressors::delta::DeltaCompressor;
use vortex_sampling_compressor::compressors::dict::DictCompressor;
use vortex_sampling_compressor::compressors::r#for::FoRCompressor;
use vortex_sampling_compressor::compressors::roaring_int::RoaringIntCompressor;
use vortex_sampling_compressor::compressors::runend::DEFAULT_RUN_END_COMPRESSOR;
use vortex_sampling_compressor::compressors::zigzag::ZigZagCompressor;
use vortex_sampling_compressor::compressors::CompressorRef;
use vortex_sampling_compressor::SamplingCompressor;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn primitive(c: &mut Criterion) {
    let mut group = c.benchmark_group("primitive-decompression");
    let num_values = u16::MAX as u64;
    group.throughput(Throughput::Bytes(num_values * 4));

    let mut rng = rand::rngs::StdRng::seed_from_u64(0);

    let uint_array = PrimitiveArray::from_vec(
        (0..num_values)
            .map(|_| rng.gen_range(0u32..256))
            .collect_vec(),
        Validity::NonNullable,
    )
    .into_array();
    let int_array = try_cast(uint_array.clone(), PType::I32.into()).unwrap();
    let index_array = PrimitiveArray::from_vec(
        (0..num_values).map(|i| (i * 2) as u32 + 42).collect_vec(),
        Validity::NonNullable,
    )
    .into_array();
    let float_array = try_cast(uint_array.clone(), PType::F32.into()).unwrap();

    let compressors_names_and_arrays = [
        (
            &BITPACK_NO_PATCHES as CompressorRef,
            "bitpacked_no_patches",
            &uint_array,
        ),
        (&BITPACK_WITH_PATCHES, "bitpacked_with_patches", &uint_array),
        (&DEFAULT_RUN_END_COMPRESSOR, "runend", &uint_array),
        (&DeltaCompressor, "delta", &uint_array),
        (&DictCompressor, "dict", &uint_array),
        (&RoaringIntCompressor, "roaring_int", &index_array),
        (&FoRCompressor, "frame_of_reference", &int_array),
        (&ZigZagCompressor, "zigzag", &int_array),
        (&ALPCompressor, "alp", &float_array),
        (&ALPRDCompressor, "alp_rd", &float_array),
    ];

    let ctx = SamplingCompressor::new(HashSet::new());
    for (compressor, name, array) in compressors_names_and_arrays {
        group.bench_function(format!("{} compress", name), |b| {
            b.iter(|| {
                black_box(
                    compressor
                        .compress(array, None, ctx.including(compressor))
                        .unwrap(),
                );
            })
        });

        let compressed = compressor
            .compress(array, None, ctx.including(compressor))
            .unwrap()
            .into_array();
        group.bench_function(format!("{} decompress", name), |b| {
            b.iter_batched(
                || compressed.clone(),
                |compressed| {
                    black_box(compressed.into_canonical().unwrap());
                },
                BatchSize::SmallInput,
            )
        });
    }
}

criterion_group!(benches, primitive);
criterion_main!(benches);
