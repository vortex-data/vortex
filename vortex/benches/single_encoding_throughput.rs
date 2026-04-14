// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]

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
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::builders::dict::dict_encode;
use vortex::array::builtins::ArrayBuiltins;
use vortex::array::dtype::Nullability;
use vortex::array::session::ArraySession;
use vortex::dtype::PType;
use vortex::encodings::alp::RDEncoder;
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
        .with_inputs(|| (&uint_array, SESSION.create_execution_ctx()))
        .bench_refs(|(a, ctx)| {
            let (_bases, _deltas) = delta_compress(a, ctx).unwrap();
            DeltaData::try_new(0).unwrap()
        });
}

#[divan::bench(name = "delta_decompress_u32")]
fn bench_delta_decompress_u32(bencher: Bencher) {
    let (uint_array, ..) = setup_primitive_arrays();
    let (bases, deltas) = delta_compress(&uint_array, &mut SESSION.create_execution_ctx()).unwrap();
    let compressed = Delta::try_new(bases.into_array(), deltas.into_array(), 0, uint_array.len())
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
        .bench_values(|a| zigzag_encode(a.as_view()).unwrap());
}

#[divan::bench(name = "zigzag_decompress_i32")]
fn bench_zigzag_decompress_i32(bencher: Bencher) {
    let (_, int_array, _) = setup_primitive_arrays();
    let compressed = zigzag_encode(int_array.as_view()).unwrap().into_array();

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
        .bench_values(|a| sequence_encode(a.as_view()).unwrap().unwrap());
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
        .bench_refs(|a| alp_encode(a.as_view(), None).unwrap());
}

#[divan::bench(name = "alp_decompress_f64")]
fn bench_alp_decompress_f64(bencher: Bencher) {
    let (_, _, float_array) = setup_primitive_arrays();
    let compressed = alp_encode(float_array.as_view(), None).unwrap();

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
            encoder.encode(a.as_view())
        });
}

#[divan::bench(name = "alp_rd_decompress_f64")]
fn bench_alp_rd_decompress_f64(bencher: Bencher) {
    let (_, _, float_array) = setup_primitive_arrays();
    let encoder = RDEncoder::new(float_array.as_slice::<f64>());
    let compressed = encoder.encode(float_array.as_view());

    with_byte_counter(bencher, NUM_VALUES * 8)
        .with_inputs(|| &compressed)
        .bench_refs(|a| a.to_canonical());
}

#[divan::bench(name = "pcodec_compress_f64")]
fn bench_pcodec_compress_f64(bencher: Bencher) {
    let (_, _, float_array) = setup_primitive_arrays();

    with_byte_counter(bencher, NUM_VALUES * 8)
        .with_inputs(|| &float_array)
        .bench_refs(|a| Pco::from_primitive(a.as_view(), 3, 0).unwrap());
}

#[divan::bench(name = "pcodec_decompress_f64")]
fn bench_pcodec_decompress_f64(bencher: Bencher) {
    let (_, _, float_array) = setup_primitive_arrays();
    let compressed = Pco::from_primitive(float_array.as_view(), 3, 0).unwrap();

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
    let dtype = uint_array.dtype().clone();
    let validity = uint_array.validity().unwrap();
    let compressed = Zstd::try_new(
        dtype,
        ZstdData::from_array(uint_array.into_array(), 3, 8192).unwrap(),
        validity,
    )
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
    let dtype = varbinview_arr.dtype().clone();
    let validity = varbinview_arr.validity().unwrap();
    let compressed = Zstd::try_new(
        dtype,
        ZstdData::from_array(varbinview_arr.clone().into_array(), 3, 8192).unwrap(),
        validity,
    )
    .unwrap()
    .into_array();
    let nbytes = varbinview_arr.into_array().nbytes() as u64;

    with_byte_counter(bencher, nbytes)
        .with_inputs(|| &compressed)
        .bench_refs(|a| a.to_canonical());
}

// TurboQuant vector quantization benchmarks.
#[cfg(feature = "unstable_encodings")]
mod turboquant_benches {
    use divan::Bencher;
    use paste::paste;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use vortex::array::IntoArray;
    use vortex::array::arrays::Extension;
    use vortex::array::arrays::ExtensionArray;
    use vortex::array::arrays::FixedSizeListArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::arrays::scalar_fn::ScalarFnArrayExt;
    use vortex::array::dtype::extension::ExtDType;
    use vortex::array::extension::EmptyMetadata;
    use vortex::array::validity::Validity;
    use vortex_array::VortexSessionExecute;
    use vortex_buffer::BufferMut;
    use vortex_tensor::encodings::turboquant::TurboQuantConfig;
    use vortex_tensor::encodings::turboquant::turboquant_encode_unchecked;
    use vortex_tensor::scalar_fns::l2_denorm::normalize_as_l2_denorm;
    use vortex_tensor::vector::Vector;

    use super::SESSION;
    use super::with_byte_counter;

    const NUM_VECTORS: usize = 1_000;

    /// Generate `num_vectors` random f32 Vector extension arrays of the given dimension
    /// using i.i.d. standard normal components. This is a conservative test distribution:
    /// real neural network embeddings typically have structure (clustered, anisotropic)
    /// that the SRHT exploits for better quantization, so Gaussian i.i.d. is a
    /// worst-case baseline for TurboQuant.
    fn setup_vector_ext(dim: usize) -> ExtensionArray {
        let mut rng = StdRng::seed_from_u64(42);
        let normal = rand_distr::Normal::new(0.0f32, 1.0).unwrap();

        let mut buf = BufferMut::<f32>::with_capacity(NUM_VECTORS * dim);
        for _ in 0..(NUM_VECTORS * dim) {
            buf.push(rand_distr::Distribution::sample(&normal, &mut rng));
        }

        let elements = PrimitiveArray::new::<f32>(buf.freeze(), Validity::NonNullable);
        let fsl = FixedSizeListArray::try_new(
            elements.into_array(),
            dim as u32,
            Validity::NonNullable,
            NUM_VECTORS,
        )
        .unwrap();
        let ext_dtype = ExtDType::<Vector>::try_new(EmptyMetadata, fsl.dtype().clone())
            .unwrap()
            .erased();
        ExtensionArray::new(ext_dtype, fsl.into_array())
    }

    fn turboquant_config(bit_width: u8) -> TurboQuantConfig {
        TurboQuantConfig {
            bit_width,
            seed: Some(123),
            num_rounds: 3,
        }
    }

    fn setup_normalized_vector_ext(dim: usize) -> ExtensionArray {
        let ext = setup_vector_ext(dim);
        let mut ctx = SESSION.create_execution_ctx();
        let normalized = normalize_as_l2_denorm(ext.into_array(), &mut ctx)
            .unwrap()
            .child_at(0)
            .clone();
        normalized.execute::<ExtensionArray>(&mut ctx).unwrap()
    }

    macro_rules! turboquant_bench {
        (compress, $dim:literal, $bits:literal, $name:ident) => {
            paste! {
                #[divan::bench(name = concat!("turboquant_compress_dim", stringify!($dim), "_", stringify!($bits), "bit"))]
                fn $name(bencher: Bencher) {
                    let normalized_ext = setup_normalized_vector_ext($dim);
                    let config = turboquant_config($bits);
                    with_byte_counter(bencher, (NUM_VECTORS * $dim * 4) as u64)
                        .with_inputs(|| (normalized_ext.clone(), SESSION.create_execution_ctx()))
                        .bench_refs(|(a, ctx)| {
                            let normalized = a
                                .as_ref()
                                .as_opt::<Extension>()
                                .expect("normalized benchmark input should be an Extension array");
                            // SAFETY: Benchmark inputs are normalized once up front so the timed
                            // region measures only TurboQuant encoding.
                            unsafe { turboquant_encode_unchecked(normalized, &config, ctx) }
                                .unwrap()
                        });
                }
            }
        };
        (decompress, $dim:literal, $bits:literal, $name:ident) => {
            paste! {
                #[divan::bench(name = concat!("turboquant_decompress_dim", stringify!($dim), "_", stringify!($bits), "bit"))]
                fn $name(bencher: Bencher) {
                    let normalized_ext = setup_normalized_vector_ext($dim);
                    let config = turboquant_config($bits);
                    let mut ctx = SESSION.create_execution_ctx();
                    let compressed = unsafe {
                        turboquant_encode_unchecked(normalized_ext.as_view(), &config, &mut ctx)
                    }
                    .unwrap();
                    with_byte_counter(bencher, (NUM_VECTORS * $dim * 4) as u64)
                        .with_inputs(|| (&compressed, SESSION.create_execution_ctx()))
                        .bench_refs(|(a, ctx)| {
                            (*a).clone()
                                .into_array()
                                .execute::<ExtensionArray>(ctx)
                                .unwrap()
                        });
                }
            }
        };
    }

    turboquant_bench!(compress, 128, 4, bench_tq_compress_128_4);
    turboquant_bench!(decompress, 128, 4, bench_tq_decompress_128_4);
    turboquant_bench!(compress, 768, 4, bench_tq_compress_768_4);
    turboquant_bench!(decompress, 768, 4, bench_tq_decompress_768_4);
    turboquant_bench!(compress, 1024, 2, bench_tq_compress_1024_2);
    turboquant_bench!(decompress, 1024, 2, bench_tq_decompress_1024_2);
    turboquant_bench!(compress, 1024, 4, bench_tq_compress_1024_4);
    turboquant_bench!(decompress, 1024, 4, bench_tq_decompress_1024_4);
    turboquant_bench!(compress, 1024, 8, bench_tq_compress_1024_8);
    turboquant_bench!(decompress, 1024, 8, bench_tq_decompress_1024_8);
}
