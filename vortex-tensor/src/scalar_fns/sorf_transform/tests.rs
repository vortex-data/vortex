// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Unit tests for the [`SorfTransform`] scalar function.

#![allow(clippy::cast_possible_truncation)]

use std::fmt;
use std::sync::Arc;
use std::sync::LazyLock;

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::arrays::dict::DictArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::Expression;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ScalarFn;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::session::ArraySession;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use super::SorfOptions;
use super::SorfTransform;
use super::rotation::SorfMatrix;
use crate::encodings::turboquant::centroids::compute_centroid_boundaries;
use crate::encodings::turboquant::centroids::find_nearest_centroid;
use crate::encodings::turboquant::centroids::get_centroids;

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

/// Build a unit-normalized input vector array and forward-transform + quantize it, returning
/// `(input_f32, FSL(Dict(codes, centroids)), padded_dim)`.
///
/// This mimics what the TurboQuant compression pipeline does, but directly, so we can test
/// `SorfTransform` in isolation.
fn forward_rotate_and_quantize(
    dim: usize,
    num_rows: usize,
    seed: u64,
    num_rounds: usize,
    bit_width: u8,
) -> VortexResult<(Vec<f32>, FixedSizeListArray, usize)> {
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

    let rotation = SorfMatrix::try_new(seed, dim, num_rounds)?;
    let padded_dim = rotation.padded_dim();
    let centroids = get_centroids(padded_dim as u32, bit_width)?;
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

    Ok((input_f32, fsl, padded_dim))
}

/// Helper to build `SorfOptions` with common defaults.
fn default_options(dim: u32, seed: u64) -> SorfOptions {
    SorfOptions {
        seed,
        num_rounds: 3,
        dimension: dim,
        element_ptype: PType::F32,
    }
}

/// Execute a `SorfTransform` array and return the decoded flat f32 elements.
fn execute_sorf(
    options: &SorfOptions,
    child: FixedSizeListArray,
    num_rows: usize,
) -> VortexResult<Vec<f32>> {
    let sorf = SorfTransform::try_new_array(options, child.into_array(), num_rows)?;
    let mut ctx = SESSION.create_execution_ctx();
    let result: ExtensionArray = sorf.into_array().execute(&mut ctx)?;
    let result_fsl: FixedSizeListArray = result.storage_array().clone().execute(&mut ctx)?;
    let result_prim: PrimitiveArray = result_fsl.elements().clone().execute(&mut ctx)?;
    Ok(result_prim.as_slice::<f32>().to_vec())
}

#[derive(Clone)]
struct NonFloatMaterializedChild;

impl ScalarFnVTable for NonFloatMaterializedChild {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::from("vortex.tensor.test.non_float_materialized_child")
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(0)
    }

    fn child_name(&self, _options: &Self::Options, _child_idx: usize) -> ChildName {
        unreachable!("NonFloatMaterializedChild has no children")
    }

    fn fmt_sql(
        &self,
        _options: &Self::Options,
        _expr: &Expression,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        write!(f, "non_float_materialized_child()")
    }

    fn return_dtype(&self, _options: &Self::Options, _arg_dtypes: &[DType]) -> VortexResult<DType> {
        Ok(DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::F32, Nullability::NonNullable)),
            128,
            Nullability::NonNullable,
        ))
    }

    fn execute(
        &self,
        _options: &Self::Options,
        _args: &dyn ExecutionArgs,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let elements = PrimitiveArray::from_iter([1u8; 128]).into_array();
        let fsl = FixedSizeListArray::try_new(elements, 128, Validity::NonNullable, 1)?;
        Ok(fsl.into_array())
    }
}

#[test]
fn roundtrip_recovery() -> VortexResult<()> {
    let dim = 128;
    let num_rows = 10;
    let seed = 42u64;
    let (input_f32, fsl, _) = forward_rotate_and_quantize(dim, num_rows, seed, 3, 8)?;
    let options = default_options(dim as u32, seed);
    let result = execute_sorf(&options, fsl, num_rows)?;

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

    // Build an empty FSL child.
    let elements = PrimitiveArray::empty::<f32>(Nullability::NonNullable);
    let fsl =
        FixedSizeListArray::try_new(elements.into_array(), padded_dim, Validity::NonNullable, 0)?;

    let sorf = SorfTransform::try_new_array(&options, fsl.into_array(), 0)?;
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

    // Build an empty but nullable FSL child.
    let elements = PrimitiveArray::empty::<f32>(Nullability::NonNullable);
    let fsl = FixedSizeListArray::try_new(
        elements.into_array(),
        padded_dim,
        Validity::from(Nullability::Nullable),
        0,
    )?;

    let sorf = SorfTransform::try_new_array(&options, fsl.into_array(), 0)?;
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
    let (_, fsl_non_nullable, padded_dim) = forward_rotate_and_quantize(dim, num_rows, seed, 3, 8)?;

    // Re-wrap the FSL with a validity mask: rows 0 and 2 are valid, rows 1 and 3 are null.
    let validity = Validity::from_iter([true, false, true, false]);
    let fsl_nullable = FixedSizeListArray::try_new(
        fsl_non_nullable.elements().clone(),
        padded_dim as u32,
        validity.clone(),
        num_rows,
    )?;

    let options = default_options(dim as u32, seed);
    let sorf = SorfTransform::try_new_array(&options, fsl_nullable.into_array(), num_rows)?;
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
    let (_, fsl, padded_dim) = forward_rotate_and_quantize(dim, num_rows, seed, 3, 8)?;

    assert_eq!(padded_dim, 256, "200 should pad to 256");

    let options = default_options(dim as u32, seed);
    let result = execute_sorf(&options, fsl, num_rows)?;

    // Output should have original dimension, not padded.
    assert_eq!(result.len(), num_rows * dim);

    Ok(())
}

#[test]
fn return_dtype_is_vector_extension() -> VortexResult<()> {
    let dim = 128u32;
    let padded_dim = dim.next_power_of_two();
    let options = default_options(dim, 42);

    let child_elem_dtype = DType::Primitive(PType::F32, Nullability::NonNullable);
    let child_dtype = DType::FixedSizeList(
        Arc::new(child_elem_dtype),
        padded_dim,
        Nullability::NonNullable,
    );

    use vortex_array::scalar_fn::ScalarFnVTable;
    let return_dtype = SorfTransform.return_dtype(&options, &[child_dtype])?;

    // Should be a Vector extension type.
    let ext = return_dtype
        .as_extension_opt()
        .expect("return dtype should be an extension type");
    assert!(ext.metadata_opt::<crate::vector::AnyVector>().is_some());

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
        dimension: 128,
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
        dimension: 128,
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
fn rejects_non_float_child_dtype_at_construction() {
    let options = default_options(128, 42);
    let elements = PrimitiveArray::from_iter([1u8; 128]).into_array();
    let child = FixedSizeListArray::try_new(elements, 128, Validity::NonNullable, 1)
        .expect("test child should be valid");

    let err = SorfTransform::try_new_array(&options, child.into_array(), 1)
        .expect_err("non-float child dtypes should be rejected at construction time");
    assert!(err.to_string().contains("logical float"));
}

#[test]
fn execute_errors_when_child_materializes_to_non_float_elements() -> VortexResult<()> {
    let child = ScalarFnArray::try_new(
        ScalarFn::new(NonFloatMaterializedChild, EmptyOptions).erased(),
        vec![],
        1,
    )?
    .into_array();
    let sorf = SorfTransform::try_new_array(&default_options(128, 42), child, 1)?;
    let mut ctx = SESSION.create_execution_ctx();

    let err = sorf
        .into_array()
        .execute::<ExtensionArray>(&mut ctx)
        .expect_err("runtime child materialization mismatch should error");
    let message = err.to_string();
    assert!(
        message.contains("float") || message.contains("U8") || message.contains("u8"),
        "unexpected runtime error: {message}",
    );
    Ok(())
}

#[test]
fn f16_output_type() -> VortexResult<()> {
    let dim = 128;
    let num_rows = 3;
    let seed = 42u64;
    let (_, fsl, _) = forward_rotate_and_quantize(dim, num_rows, seed, 3, 8)?;

    let options = SorfOptions {
        seed,
        num_rounds: 3,
        dimension: dim as u32,
        element_ptype: PType::F16,
    };
    let sorf = SorfTransform::try_new_array(&options, fsl.into_array(), num_rows)?;
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
    let (_, fsl, _) = forward_rotate_and_quantize(dim, num_rows, seed, 3, 8)?;

    let options = SorfOptions {
        seed,
        num_rounds: 3,
        dimension: dim as u32,
        element_ptype: PType::F64,
    };
    let sorf = SorfTransform::try_new_array(&options, fsl.into_array(), num_rows)?;
    let mut ctx = SESSION.create_execution_ctx();
    let result: ExtensionArray = sorf.into_array().execute(&mut ctx)?;
    let result_fsl: FixedSizeListArray = result.storage_array().clone().execute(&mut ctx)?;
    let result_prim: PrimitiveArray = result_fsl.elements().clone().execute(&mut ctx)?;

    assert_eq!(result_prim.ptype(), PType::F64);
    assert_eq!(result_prim.as_slice::<f64>().len(), num_rows * dim);

    Ok(())
}
