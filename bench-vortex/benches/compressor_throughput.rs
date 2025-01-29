use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use mimalloc::MiMalloc;
use rand::distributions::Alphanumeric;
use rand::seq::SliceRandom as _;
use rand::{thread_rng, Rng, SeedableRng as _};
use vortex::aliases::hash_set::HashSet;
use vortex::array::VarBinViewArray;
use vortex::buffer::Buffer;
use vortex::compute::try_cast;
use vortex::dtype::PType;
use vortex::encodings::dict::dict_encode;
use vortex::encodings::fsst::{fsst_compress, fsst_train_compressor};
use vortex::sampling_compressor::compressors::alp::ALPCompressor;
use vortex::sampling_compressor::compressors::alp_rd::ALPRDCompressor;
use vortex::sampling_compressor::compressors::bitpacked::{
    BITPACK_NO_PATCHES, BITPACK_WITH_PATCHES,
};
use vortex::sampling_compressor::compressors::delta::DeltaCompressor;
use vortex::sampling_compressor::compressors::dict::DictCompressor;
use vortex::sampling_compressor::compressors::r#for::FoRCompressor;
use vortex::sampling_compressor::compressors::runend::DEFAULT_RUN_END_COMPRESSOR;
use vortex::sampling_compressor::compressors::zigzag::ZigZagCompressor;
use vortex::sampling_compressor::compressors::CompressorRef;
use vortex::sampling_compressor::SamplingCompressor;
use vortex::{IntoArrayData as _, IntoCanonical};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn primitive(c: &mut Criterion) {
    let mut group = c.benchmark_group("primitive-decompression");
    let num_values = u16::MAX as u64;
    group.throughput(Throughput::Bytes(num_values * 4));

    let mut rng = rand::rngs::StdRng::seed_from_u64(0);

    let uint_array =
        Buffer::from_iter((0..num_values).map(|_| rng.gen_range(0u32..256))).into_array();
    let int_array = try_cast(uint_array.clone(), PType::I32.into()).unwrap();
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

fn strings(c: &mut Criterion) {
    let mut group = c.benchmark_group("string-decompression");
    let num_values = u16::MAX as u64;
    group.throughput(Throughput::Bytes(num_values * 8));

    let varbinview_arr = VarBinViewArray::from_iter_str(gen_varbin_words(1_000_000, 0.00005));
    let dict = dict_encode(varbinview_arr.as_ref()).unwrap();
    group.throughput(Throughput::Bytes(
        varbinview_arr.clone().into_array().nbytes() as u64,
    ));
    group.bench_function("dict_decode_varbinview", |b| {
        b.iter(|| black_box(dict.clone().into_canonical().unwrap()));
    });

    let fsst_compressor = fsst_train_compressor(&varbinview_arr.clone().into_array()).unwrap();
    let fsst_array = fsst_compress(&varbinview_arr.clone().into_array(), &fsst_compressor).unwrap();
    group.bench_function("fsst_decompress_varbinview", |b| {
        b.iter(|| black_box(fsst_array.clone().into_canonical().unwrap()));
    });
}

fn gen_varbin_words(len: usize, uniqueness: f64) -> Vec<String> {
    let mut rng = thread_rng();
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

criterion_group!(benches, primitive, strings);
criterion_main!(benches);
