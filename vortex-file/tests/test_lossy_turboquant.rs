// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! End-to-end test for the `Lossy<Vector<f32>>` -> TurboQuant -> file-roundtrip path.
//!
//! Asserts that:
//!
//! 1. Wrapping a `Vector<f32, dim>` column in `Lossy<Vector<f32, dim>>` triggers TurboQuant under
//!    the default `BtrBlocksCompressorBuilder`. The compressed inner array under the Lossy wrapper
//!    must contain a `ScalarFnArray(L2Denorm, ...)` produced by [`TurboQuantScheme`].
//! 2. The same column WITHOUT the `Lossy` wrapper is NOT compressed by TurboQuant under the
//!    default builder, because the compressor gates lossy schemes on dtype-level Lossy consent.
//! 3. Round-tripping the `Lossy<Vector<f32, dim>>` column through `vortex-file` write/read
//!    preserves the dtype.
//! 4. The decompressed values are within a relative-MSE bound of the input.
//!
//! Gated behind `unstable_encodings` so the test can register the tensor scalar-fn array plugins
//! and pull the L2Denorm / SorfTransform array IDs into the writer's allow-list.

#![cfg(feature = "unstable_encodings")]
#![expect(clippy::tests_outside_test_module)]

use std::sync::LazyLock;

use futures::TryStreamExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::Distribution;
use rand_distr::Normal;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::ScalarFn;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex_array::arrays::scalar_fn::ExactScalarFn;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::extension::AnyLossy;
use vortex_array::extension::EmptyMetadata;
use vortex_array::extension::Lossy;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::session::ScalarFnSession;
use vortex_array::session::ArraySession;
use vortex_array::validity::Validity;
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_btrblocks::BtrBlocksCompressorBuilder;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_file::ALLOWED_ENCODINGS;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::WriteOptionsSessionExt;
use vortex_file::WriteStrategyBuilder;
use vortex_io::session::RuntimeSession;
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;
use vortex_tensor::scalar_fns::l2_denorm::L2Denorm;
use vortex_tensor::scalar_fns::sorf_transform::SorfTransform;
use vortex_tensor::vector::Vector;

const NUM_ROWS: usize = 200;
const DIM: u32 = 128;
const SEED: u64 = 1234;
/// Slack on the per-vector normalized MSE for default (8-bit) TurboQuant on 128-d Gaussian
/// vectors. The paper's theoretical bound at 8 bits is ~4e-5; we use 0.05 to give plenty of
/// margin against the SRHT-vs-Haar gap and across-platform floating-point variation.
const RELATIVE_ERROR_BOUND: f32 = 0.05;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    // Register the tensor scalar-fn array plugins so the L2Denorm/SorfTransform encodings
    // emitted by TurboQuant can be serialized into the file and deserialized on read.
    //
    // SAFETY: single-threaded test setup before any other thread exists.
    unsafe { std::env::set_var(vortex_tensor::SCALAR_FN_ARRAY_TENSOR_PLUGIN_ENV, "1") };

    let session = VortexSession::empty()
        .with::<ArraySession>()
        .with::<LayoutSession>()
        .with::<ScalarFnSession>()
        .with::<RuntimeSession>();

    vortex_file::register_default_encodings(&session);

    session
});

/// Build a non-nullable `Vector<f32, dim>` extension array of `num_rows` Gaussian vectors.
fn gaussian_vector_array(num_rows: usize, dim: u32, seed: u64) -> VortexResult<ArrayRef> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0f32, 1.0).map_err(|e| vortex_err!("{e}"))?;

    let mut buf = BufferMut::<f32>::with_capacity(num_rows * dim as usize);
    for _ in 0..(num_rows * dim as usize) {
        buf.push(normal.sample(&mut rng));
    }
    let elements = PrimitiveArray::new::<f32>(buf.freeze(), Validity::NonNullable).into_array();
    let fsl = FixedSizeListArray::try_new(elements, dim, Validity::NonNullable, num_rows)?;
    let ext_dtype = ExtDType::<Vector>::try_new(EmptyMetadata, fsl.dtype().clone())?.erased();
    Ok(ExtensionArray::new(ext_dtype, fsl.into_array()).into_array())
}

/// Wrap an arbitrary `Vector<f32, dim>` extension array in a `Lossy<Vector<f32, dim>>` extension
/// array.
fn wrap_lossy(inner: ArrayRef) -> VortexResult<ArrayRef> {
    let lossy_dtype = Lossy::new(inner.dtype().clone())?.erased();
    Ok(ExtensionArray::new(lossy_dtype, inner).into_array())
}

/// Returns true iff the array recursively contains a `ScalarFnArray` whose function is
/// [`L2Denorm`]. TurboQuant always wraps its compressed output in `ScalarFnArray(L2Denorm, ...)`,
/// so this is the structural signal that TurboQuant fired.
fn contains_l2_denorm(array: &ArrayRef) -> bool {
    if array.as_opt::<ScalarFn>().is_some() && array.as_opt::<ExactScalarFn<L2Denorm>>().is_some() {
        return true;
    }

    for slot in array.slots() {
        if let Some(child) = slot.as_ref()
            && contains_l2_denorm(child)
        {
            return true;
        }
    }
    false
}

/// Per-vector normalized MSE: `sum_row ||x - x_hat||^2 / ||x||^2 / num_rows`.
fn per_vector_normalized_mse(
    original: &[f32],
    reconstructed: &[f32],
    dim: usize,
    num_rows: usize,
) -> f32 {
    let mut total = 0.0f32;
    for row in 0..num_rows {
        let orig = &original[row * dim..(row + 1) * dim];
        let recon = &reconstructed[row * dim..(row + 1) * dim];
        let norm_sq: f32 = orig.iter().map(|&v| v * v).sum();
        if norm_sq < 1e-10 {
            continue;
        }
        let err_sq: f32 = orig
            .iter()
            .zip(recon.iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum();
        total += err_sq / norm_sq;
    }
    total / num_rows as f32
}

/// Pull the flat f32 elements out of a `Vector<f32, _>` or `Lossy<Vector<f32, _>>` extension
/// array.
fn flatten_vector_elements(
    array: ArrayRef,
    ctx: &mut vortex_array::ExecutionCtx,
) -> VortexResult<Vec<f32>> {
    let ext: ExtensionArray = array.execute(ctx)?;
    let inner_ext: ExtensionArray = if let Some(inner) = ext.peel_lossy() {
        inner.clone().execute(ctx)?
    } else {
        ext
    };
    let fsl: FixedSizeListArray = inner_ext.storage_array().clone().execute(ctx)?;
    let elements: PrimitiveArray = fsl.elements().clone().execute(ctx)?;
    Ok(elements.as_slice::<f32>().to_vec())
}

/// In-memory compression: assert TurboQuant fires on `Lossy<Vector<f32>>` but NOT on bare
/// `Vector<f32>` under the default `BtrBlocksCompressorBuilder`.
#[test]
fn turboquant_fires_only_when_lossy() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();

    let plain_vector = gaussian_vector_array(NUM_ROWS, DIM, SEED)?;
    let lossy_vector = wrap_lossy(plain_vector.clone())?;

    let compressor = BtrBlocksCompressorBuilder::default().build();

    let compressed_lossy = compressor.compress(&lossy_vector, &mut ctx)?;
    // The outer wrapper must remain `Lossy<...>`.
    let lossy_ext: ExtensionArray = compressed_lossy.clone().execute(&mut ctx)?;
    assert!(
        lossy_ext.ext_dtype().is::<AnyLossy>(),
        "expected outer Lossy wrapper after compression, got dtype {}",
        compressed_lossy.dtype()
    );
    let inner = lossy_ext.storage_array();
    assert!(
        contains_l2_denorm(inner),
        "expected ScalarFnArray(L2Denorm, ...) under the Lossy wrapper, got dtype {} encoding \
         {:?}",
        inner.dtype(),
        inner.encoding_id(),
    );

    // Plain Vector<f32, dim>: TurboQuant must NOT fire because the column is not marked Lossy.
    let compressed_plain = compressor.compress(&plain_vector, &mut ctx)?;
    assert!(
        !contains_l2_denorm(&compressed_plain),
        "expected no TurboQuant on bare Vector<f32, dim>, got dtype {} encoding {:?}",
        compressed_plain.dtype(),
        compressed_plain.encoding_id(),
    );

    Ok(())
}

/// File round-trip: write a `Lossy<Vector<f32, DIM>>` column to a Vortex file, read it back, and
/// verify (a) the dtype is preserved as `Lossy<Vector<f32, DIM>>`, and (b) the decompressed
/// values are within a relative-MSE bound of the original input.
#[tokio::test]
async fn lossy_vector_roundtrip_through_file() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();

    let original_vector = gaussian_vector_array(NUM_ROWS, DIM, SEED)?;
    let original_flat = flatten_vector_elements(original_vector.clone(), &mut ctx)?;

    let lossy_vector = wrap_lossy(original_vector)?;
    let original_dtype = lossy_vector.dtype().clone();
    let struct_array = StructArray::from_fields(&[("emb", lossy_vector)])
        .vortex_expect("from_fields on a single-field struct")
        .into_array();

    // TurboQuant emits ScalarFnArray(L2Denorm, ...) and ScalarFnArray(SorfTransform, ...). The
    // default file allow-list does not include those, so we extend it here so the writer accepts
    // them during normalization.
    let mut allowed = ALLOWED_ENCODINGS.clone();
    allowed.insert(L2Denorm.id());
    allowed.insert(SorfTransform.id());

    let strategy = WriteStrategyBuilder::default()
        .with_compressor(BtrBlocksCompressor::default())
        .with_allow_encodings(allowed)
        .build();

    let mut bytes = Vec::new();
    SESSION
        .write_options()
        .with_strategy(strategy)
        .write(&mut bytes, struct_array.clone().to_array_stream())
        .await
        .vortex_expect("write should succeed");

    let bytes = ByteBuffer::from(bytes);
    let vxf = SESSION
        .open_options()
        .open_buffer(bytes)
        .vortex_expect("open should succeed");

    let chunks: Vec<ArrayRef> = vxf
        .scan()
        .vortex_expect("scan")
        .into_array_stream()
        .vortex_expect("into_array_stream")
        .try_collect()
        .await
        .vortex_expect("collect chunks");

    let read_struct: StructArray = ChunkedArray::try_new(chunks, struct_array.dtype().clone())?
        .into_array()
        .execute(&mut ctx)?;

    let read_emb = read_struct.unmasked_field_by_name("emb")?.clone();
    assert_eq!(
        read_emb.dtype(),
        &original_dtype,
        "expected Lossy<Vector<f32, dim>> dtype to round-trip exactly",
    );

    let DType::Extension(read_ext) = read_emb.dtype() else {
        panic!(
            "expected Extension after roundtrip, got {}",
            read_emb.dtype()
        );
    };
    assert!(
        read_ext.is::<Lossy>(),
        "expected Lossy outer wrapper after roundtrip, got {}",
        read_emb.dtype()
    );

    let read_flat = flatten_vector_elements(read_emb, &mut ctx)?;
    assert_eq!(read_flat.len(), original_flat.len());

    let mse = per_vector_normalized_mse(&original_flat, &read_flat, DIM as usize, NUM_ROWS);
    assert!(
        mse < RELATIVE_ERROR_BOUND,
        "per-vector normalized MSE {mse} exceeds bound {RELATIVE_ERROR_BOUND}",
    );

    Ok(())
}
