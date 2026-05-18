// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Unit tests for the [`SorfTransform`] scalar function.

#![allow(clippy::cast_possible_truncation)]

use std::sync::Arc;

use vortex_array::ArrayPlugin;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::dict::DictArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex_array::arrays::scalar_fn::plugin::ScalarFnArrayPlugin;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::extension::EmptyMetadata;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use super::SorfOptions;
use super::SorfTransform;
use super::rotation::SorfMatrix;
use crate::encodings::turboquant::centroids::compute_centroid_boundaries;
use crate::encodings::turboquant::centroids::compute_or_get_centroids;
use crate::encodings::turboquant::centroids::find_nearest_centroid;
use crate::tests::SESSION;
use crate::types::vector::AnyVector;
use crate::types::vector::Vector;

/// Build a unit-normalized input vector array and forward-transform + quantize it, returning
/// `(input_f32, Vector<padded_dim>(FSL(Dict(codes, centroids))), padded_dim)`.
///
/// This mimics what the TurboQuant compression pipeline does, but directly, so we can test
/// `SorfTransform` in isolation.
fn forward_rotate_and_quantize(
    dim: usize,
    num_rows: usize,
    seed: u64,
    num_rounds: usize,
    bit_width: u8,
) -> VortexResult<(Vec<f32>, ExtensionArray, usize)> {
    // Build simple unit-normalized input vectors.
    let mut input_f32 = vec![0.0f32; num_rows * dim];
    for row in 0..num_rows {
        let mut norm_sq = 0.0f32;
        for i in 0..dim {
            let val = ((row * dim + i) as f32 + 1.0) * 0.01;
            input_f32[row * dim + i] = val;
            norm_sq += val * val;
        }
        let norm = norm_sq.sqrt();
        for i in 0..dim {
            input_f32[row * dim + i] /= norm;
        }
    }

    let padded_dim = dim.next_power_of_two();
    let rotation = SorfMatrix::try_new_padded(padded_dim, num_rounds, seed)?;
    let centroids = compute_or_get_centroids(padded_dim as u32, bit_width)?;
    let boundaries = compute_centroid_boundaries(&centroids);

    let mut all_indices = BufferMut::<u8>::with_capacity(num_rows * padded_dim);
    let mut padded = vec![0.0f32; padded_dim];
    let mut rotated = vec![0.0f32; padded_dim];

    for row in 0..num_rows {
        padded[..dim].copy_from_slice(&input_f32[row * dim..(row + 1) * dim]);
        padded[dim..].fill(0.0);
        rotation.rotate(&padded, &mut rotated);
        for j in 0..padded_dim {
            all_indices.push(find_nearest_centroid(rotated[j], &boundaries));
        }
    }

    let codes = PrimitiveArray::new::<u8>(all_indices.freeze(), Validity::NonNullable);
    let mut centroids_buf = BufferMut::<f32>::with_capacity(centroids.len());
    centroids_buf.extend_from_slice(&centroids);
    let centroids_arr = PrimitiveArray::new::<f32>(centroids_buf.freeze(), Validity::NonNullable);
    let dict = DictArray::try_new(codes.into_array(), centroids_arr.into_array())?;
    let fsl = FixedSizeListArray::try_new(
        dict.into_array(),
        padded_dim as u32,
        Validity::NonNullable,
        num_rows,
    )?;
    let padded_vector = wrap_as_vector(fsl, Validity::NonNullable)?;

    Ok((input_f32, padded_vector, padded_dim))
}

/// Wrap an FSL in a Vector extension, optionally re-tagging its validity. This is used by tests
/// that need to adjust top-level nullability of a padded vector child.
fn wrap_as_vector(fsl: FixedSizeListArray, validity: Validity) -> VortexResult<ExtensionArray> {
    let list_size = fsl.list_size();
    let num_rows = fsl.len();
    let elements = fsl.elements().clone();
    let fsl = FixedSizeListArray::try_new(elements, list_size, validity, num_rows)?;
    let ext_dtype = ExtDType::<Vector>::try_new(EmptyMetadata, fsl.dtype().clone())?.erased();
    Ok(ExtensionArray::new(ext_dtype, fsl.into_array()))
}

/// Helper to build `SorfOptions` with common defaults.
fn default_options(dim: u32, seed: u64) -> SorfOptions {
    SorfOptions {
        seed,
        num_rounds: 3,
        dimensions: dim,
        element_ptype: PType::F32,
    }
}

/// Execute a `SorfTransform` array and return the decoded flat f32 elements.
fn execute_sorf(
    options: &SorfOptions,
    child: ExtensionArray,
    num_rows: usize,
) -> VortexResult<Vec<f32>> {
    let sorf = SorfTransform::try_new_array(options, child.into_array(), num_rows)?;
    let mut ctx = SESSION.create_execution_ctx();
    let result: ExtensionArray = sorf.into_array().execute(&mut ctx)?;
    let result_fsl: FixedSizeListArray = result.storage_array().clone().execute(&mut ctx)?;
    let result_prim: PrimitiveArray = result_fsl.elements().clone().execute(&mut ctx)?;
    Ok(result_prim.as_slice::<f32>().to_vec())
}

/// Build an empty `Vector<padded_dim>` extension array wrapping an empty FSL.
fn empty_padded_vector(padded_dim: u32, validity: Validity) -> VortexResult<ExtensionArray> {
    let elements = PrimitiveArray::empty::<f32>(Nullability::NonNullable);
    let fsl = FixedSizeListArray::try_new(elements.into_array(), padded_dim, validity, 0)?;
    let ext_dtype = ExtDType::<Vector>::try_new(EmptyMetadata, fsl.dtype().clone())?.erased();
    Ok(ExtensionArray::new(ext_dtype, fsl.into_array()))
}

#[test]
fn roundtrip_recovery() -> VortexResult<()> {
    let dim = 128;
    let num_rows = 10;
    let seed = 42u64;
    let (input_f32, padded_vector, _) = forward_rotate_and_quantize(dim, num_rows, seed, 3, 8)?;
    let options = default_options(dim as u32, seed);
    let result = execute_sorf(&options, padded_vector, num_rows)?;

    assert_eq!(result.len(), num_rows * dim);

    // At 8-bit quantization, the reconstruction should be very close to the input.
    for row in 0..num_rows {
        let orig = &input_f32[row * dim..(row + 1) * dim];
        let recon = &result[row * dim..(row + 1) * dim];
        let err_sq: f32 = orig
            .iter()
            .zip(recon)
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum();
        let norm_sq: f32 = orig.iter().map(|&v| v * v).sum();
        assert!(
            err_sq / norm_sq < 1e-3,
            "row {row} MSE too high: {:.6}",
            err_sq / norm_sq
        );
    }
    Ok(())
}

#[test]
fn empty_array_non_nullable() -> VortexResult<()> {
    let dim = 128u32;
    let padded_dim = dim.next_power_of_two();
    let options = default_options(dim, 42);

    // Build an empty Vector<padded_dim> child.
    let child = empty_padded_vector(padded_dim, Validity::NonNullable)?;

    let sorf = SorfTransform::try_new_array(&options, child.into_array(), 0)?;
    let mut ctx = SESSION.create_execution_ctx();
    let result: ExtensionArray = sorf.into_array().execute(&mut ctx)?;

    assert_eq!(result.len(), 0);

    // Output should be non-nullable.
    let result_fsl: FixedSizeListArray = result.storage_array().clone().execute(&mut ctx)?;
    assert!(!result_fsl.dtype().is_nullable());

    Ok(())
}

#[test]
fn empty_array_nullable() -> VortexResult<()> {
    let dim = 128u32;
    let padded_dim = dim.next_power_of_two();
    let options = default_options(dim, 42);

    // Build an empty but nullable Vector<padded_dim> child.
    let child = empty_padded_vector(padded_dim, Validity::from(Nullability::Nullable))?;

    let sorf = SorfTransform::try_new_array(&options, child.into_array(), 0)?;
    let mut ctx = SESSION.create_execution_ctx();
    let result: ExtensionArray = sorf.into_array().execute(&mut ctx)?;

    assert_eq!(result.len(), 0);

    // Output should be nullable (matching the child).
    let result_fsl: FixedSizeListArray = result.storage_array().clone().execute(&mut ctx)?;
    assert!(result_fsl.dtype().is_nullable());

    Ok(())
}

#[test]
fn nullable_validity_propagation() -> VortexResult<()> {
    let dim = 128;
    let num_rows = 4;
    let seed = 42u64;
    let (_, non_nullable_vector, padded_dim) =
        forward_rotate_and_quantize(dim, num_rows, seed, 3, 8)?;

    // Re-wrap the underlying FSL with a validity mask: rows 0 and 2 are valid, rows 1 and 3
    // are null.
    let validity = Validity::from_iter([true, false, true, false]);
    let fsl_non_nullable: FixedSizeListArray = non_nullable_vector
        .storage_array()
        .clone()
        .execute(&mut SESSION.create_execution_ctx())?;
    let fsl_nullable = FixedSizeListArray::try_new(
        fsl_non_nullable.elements().clone(),
        padded_dim as u32,
        validity.clone(),
        num_rows,
    )?;
    let nullable_vector = wrap_as_vector(fsl_nullable, validity.clone())?;

    let options = default_options(dim as u32, seed);
    let sorf = SorfTransform::try_new_array(&options, nullable_vector.into_array(), num_rows)?;
    let mut ctx = SESSION.create_execution_ctx();
    let result: ExtensionArray = sorf.into_array().execute(&mut ctx)?;
    let result_fsl: FixedSizeListArray = result.storage_array().clone().execute(&mut ctx)?;

    // The output FSL validity should match the input.
    let output_validity = result_fsl.validity()?;
    for row in 0..num_rows {
        assert_eq!(
            output_validity.is_valid(row)?,
            validity.is_valid(row)?,
            "validity mismatch at row {row}"
        );
    }

    Ok(())
}

#[test]
fn dimension_truncation() -> VortexResult<()> {
    // Use a non-power-of-2 dimension (padded 200 -> 256).
    let dim = 200;
    let num_rows = 3;
    let seed = 42u64;
    let (_, padded_vector, padded_dim) = forward_rotate_and_quantize(dim, num_rows, seed, 3, 8)?;

    assert_eq!(padded_dim, 256, "200 should pad to 256");

    let options = default_options(dim as u32, seed);
    let result = execute_sorf(&options, padded_vector, num_rows)?;

    // Output should have original dimension, not padded.
    assert_eq!(result.len(), num_rows * dim);

    Ok(())
}

#[test]
fn return_dtype_is_vector_extension() -> VortexResult<()> {
    let dim = 128u32;
    let padded_dim = dim.next_power_of_two();
    let options = default_options(dim, 42);

    // Input must be a Vector<padded_dim> extension dtype.
    let child_elem_dtype = DType::Primitive(PType::F32, Nullability::NonNullable);
    let child_storage_dtype = DType::FixedSizeList(
        Arc::new(child_elem_dtype),
        padded_dim,
        Nullability::NonNullable,
    );
    let child_ext_dtype = ExtDType::<Vector>::try_new(EmptyMetadata, child_storage_dtype)?.erased();
    let child_dtype = DType::Extension(child_ext_dtype);

    use vortex_array::scalar_fn::ScalarFnVTable;
    let return_dtype = SorfTransform.return_dtype(&options, &[child_dtype])?;

    // Should be a Vector extension type.
    let ext = return_dtype
        .as_extension_opt()
        .expect("return dtype should be an extension type");
    assert!(ext.metadata_opt::<AnyVector>().is_some());

    // Inner FSL should have the original (unpadded) dimension.
    let DType::FixedSizeList(_, inner_dim, _) = ext.storage_dtype() else {
        panic!("expected storage dtype to be FSL");
    };
    assert_eq!(*inner_dim, dim);

    Ok(())
}

#[test]
fn rejects_zero_rounds_at_construction() {
    let options = SorfOptions {
        seed: 42,
        num_rounds: 0,
        dimensions: 128,
        element_ptype: PType::F32,
    };
    let elements = PrimitiveArray::from_iter([0.0f32; 128]).into_array();
    let child = FixedSizeListArray::try_new(elements, 128, Validity::NonNullable, 1)
        .expect("test child should be valid");

    let err = SorfTransform::try_new_array(&options, child.into_array(), 1)
        .expect_err("zero rounds should be rejected at construction time");
    assert!(err.to_string().contains("num_rounds"));
}

#[test]
fn rejects_non_float_output_ptype_at_construction() {
    let options = SorfOptions {
        seed: 42,
        num_rounds: 3,
        dimensions: 128,
        element_ptype: PType::U8,
    };
    let elements = PrimitiveArray::from_iter([0.0f32; 128]).into_array();
    let child = FixedSizeListArray::try_new(elements, 128, Validity::NonNullable, 1)
        .expect("test child should be valid");

    let err = SorfTransform::try_new_array(&options, child.into_array(), 1)
        .expect_err("non-float output ptypes should be rejected at construction time");
    assert!(err.to_string().contains("element_ptype"));
}

#[test]
fn rejects_non_vector_extension_child_at_construction() {
    let options = default_options(128, 42);
    // A bare FSL child (not wrapped in a Vector extension) should be rejected.
    let elements = PrimitiveArray::from_iter([0.0f32; 128]).into_array();
    let child = FixedSizeListArray::try_new(elements, 128, Validity::NonNullable, 1)
        .expect("test child should be valid");

    let err = SorfTransform::try_new_array(&options, child.into_array(), 1)
        .expect_err("non-Vector-extension children should be rejected at construction time");
    assert!(err.to_string().contains("Vector extension"));
}

#[test]
fn rejects_wrong_padded_dimension_at_construction() {
    // Options say dimension=128 so padded_dim should be 128. Pass a Vector<256> instead.
    let options = default_options(128, 42);
    let elements = PrimitiveArray::from_iter([0.0f32; 256]).into_array();
    let fsl = FixedSizeListArray::try_new(elements, 256, Validity::NonNullable, 1)
        .expect("test child should be valid");
    let child = wrap_as_vector(fsl, Validity::NonNullable).expect("wrap should succeed");

    let err = SorfTransform::try_new_array(&options, child.into_array(), 1)
        .expect_err("mismatched padded dimension should be rejected at construction time");
    assert!(err.to_string().contains("dimension"));
}

#[test]
fn rejects_non_f32_child_storage_at_construction() {
    // Options are valid and target f32 output. Pass a Vector<128> whose storage is f16 instead
    // of f32 -- SorfTransform's f32-only input constraint should reject this.
    let options = default_options(128, 42);
    let elements = PrimitiveArray::from_iter([half::f16::from_f32(0.0); 128]).into_array();
    let fsl = FixedSizeListArray::try_new(elements, 128, Validity::NonNullable, 1)
        .expect("test child should be valid");
    let child = wrap_as_vector(fsl, Validity::NonNullable).expect("wrap should succeed");

    let err = SorfTransform::try_new_array(&options, child.into_array(), 1)
        .expect_err("non-f32 Vector storage should be rejected at construction time");
    assert!(err.to_string().contains("f32"));
}

#[test]
fn f16_output_type() -> VortexResult<()> {
    let dim = 128;
    let num_rows = 3;
    let seed = 42u64;
    let (_, padded_vector, _) = forward_rotate_and_quantize(dim, num_rows, seed, 3, 8)?;

    let options = SorfOptions {
        seed,
        num_rounds: 3,
        dimensions: dim as u32,
        element_ptype: PType::F16,
    };
    let sorf = SorfTransform::try_new_array(&options, padded_vector.into_array(), num_rows)?;
    let mut ctx = SESSION.create_execution_ctx();
    let result: ExtensionArray = sorf.into_array().execute(&mut ctx)?;
    let result_fsl: FixedSizeListArray = result.storage_array().clone().execute(&mut ctx)?;
    let result_prim: PrimitiveArray = result_fsl.elements().clone().execute(&mut ctx)?;

    assert_eq!(result_prim.ptype(), PType::F16);
    assert_eq!(result_prim.as_slice::<half::f16>().len(), num_rows * dim);

    Ok(())
}

#[test]
fn f64_output_type() -> VortexResult<()> {
    let dim = 128;
    let num_rows = 3;
    let seed = 42u64;
    let (_, padded_vector, _) = forward_rotate_and_quantize(dim, num_rows, seed, 3, 8)?;

    let options = SorfOptions {
        seed,
        num_rounds: 3,
        dimensions: dim as u32,
        element_ptype: PType::F64,
    };
    let sorf = SorfTransform::try_new_array(&options, padded_vector.into_array(), num_rows)?;
    let mut ctx = SESSION.create_execution_ctx();
    let result: ExtensionArray = sorf.into_array().execute(&mut ctx)?;
    let result_fsl: FixedSizeListArray = result.storage_array().clone().execute(&mut ctx)?;
    let result_prim: PrimitiveArray = result_fsl.elements().clone().execute(&mut ctx)?;

    assert_eq!(result_prim.ptype(), PType::F64);
    assert_eq!(result_prim.as_slice::<f64>().len(), num_rows * dim);

    Ok(())
}

/// Build a trivial `Vector<FSL<f32, padded_dim, validity>>` child populated with zeroes. The values
/// are irrelevant for the serde round-trip test; only the dtype shape matters.
fn trivial_padded_vector(padded_dim: u32, num_rows: usize, validity: Validity) -> ArrayRef {
    let elements = PrimitiveArray::new(
        Buffer::<f32>::zeroed(num_rows * padded_dim as usize),
        Validity::NonNullable,
    );
    let fsl = FixedSizeListArray::try_new(elements.into_array(), padded_dim, validity, num_rows)
        .vortex_expect("fsl must build");
    let ext_dtype = ExtDType::<Vector>::try_new(EmptyMetadata, fsl.dtype().clone())
        .vortex_expect("ext dtype must build")
        .erased();
    ExtensionArray::new(ext_dtype, fsl.into_array()).into_array()
}

#[rstest::rstest]
// Non-power-of-two dimension to exercise `padded_dim = dim.next_power_of_two()`.
#[case::power_of_two_dim(128, Validity::NonNullable)]
#[case::non_power_of_two_dim(100, Validity::NonNullable)]
// Nullable top-level Vector to verify child nullability is reconstructed from the parent output.
#[case::nullable_child(100, Validity::AllValid)]
fn serde_round_trip(#[case] dimensions: u32, #[case] validity: Validity) -> VortexResult<()> {
    let padded_dim = dimensions.next_power_of_two();
    let num_rows = 4;
    let options = SorfOptions {
        seed: 42,
        num_rounds: 3,
        dimensions,
        element_ptype: PType::F32,
    };
    let child = trivial_padded_vector(padded_dim, num_rows, validity);
    let original = SorfTransform::try_new_array(&options, child.clone(), num_rows)?.into_array();

    let plugin = ScalarFnArrayPlugin::new(SorfTransform);
    let metadata = plugin
        .serialize(&original, &SESSION)?
        .expect("SorfTransform serialize must produce metadata");

    let children = vec![child];
    let recovered = plugin.deserialize(
        original.dtype(),
        original.len(),
        &metadata,
        &[],
        &children,
        &SESSION,
    )?;

    assert_eq!(recovered.dtype(), original.dtype());
    assert_eq!(recovered.len(), original.len());
    assert_eq!(recovered.encoding_id(), original.encoding_id());
    Ok(())
}
