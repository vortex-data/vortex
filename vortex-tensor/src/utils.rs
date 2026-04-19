// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::arrays::scalar_fn::ExactScalarFn;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::matcher::AnyTensor;
use crate::matcher::TensorMatch;
use crate::scalar_fns::l2_denorm::L2Denorm;

/// Validates that `input_dtype` is a float-valued tensor-like extension dtype.
pub fn validate_tensor_float_input(input_dtype: &DType) -> VortexResult<TensorMatch<'_>> {
    let ext = input_dtype
        .as_extension_opt()
        .ok_or_else(|| vortex_err!("expected an extension type, got {input_dtype}"))?;

    let tensor_match = ext
        .metadata_opt::<AnyTensor>()
        .ok_or_else(|| vortex_err!("expected an `AnyTensor`, got {input_dtype}"))?;

    let ptype = tensor_match.element_ptype();
    vortex_ensure!(
        ptype.is_float(),
        "expected a float element dtype, got {ptype}",
    );

    Ok(tensor_match)
}

/// Validates that two arguments of a binary tensor-like operator share the same float tensor
/// dtype (ignoring top-level nullability), returning the shared [`TensorMatch`].
///
/// `op_name` is interpolated into the shape-mismatch error message so callers get a
/// self-identifying diagnostic (e.g. "InnerProduct requires both inputs ...").
pub fn validate_binary_tensor_float_inputs<'a>(
    op_name: &str,
    lhs: &'a DType,
    rhs: &DType,
) -> VortexResult<TensorMatch<'a>> {
    vortex_ensure!(
        lhs.eq_ignore_nullability(rhs),
        "{op_name} requires both inputs to have the same dtype, got {lhs} and {rhs}"
    );
    validate_tensor_float_input(lhs)
}

/// Cast a float [`PrimitiveArray`] to a `Buffer<f32>`.
///
/// Several operations in this crate (SORF transform, TurboQuant quantization) work exclusively
/// in f32. This function handles the cast from any float ptype:
///
/// - f16: losslessly widened to f32.
/// - f32: zero-copy buffer extraction.
/// - f64: truncated to f32 precision. Values outside f32 range become +/- infinity. This is
///   acceptable because callers of this function operate in f32 and document this constraint.
pub fn cast_to_f32(prim: PrimitiveArray) -> VortexResult<Buffer<f32>> {
    match prim.ptype() {
        PType::F16 => Ok(prim
            .as_slice::<half::f16>()
            .iter()
            .map(|&v| f32::from(v))
            .collect()),
        PType::F32 => Ok(prim.into_buffer()),
        PType::F64 => Ok(prim
            .as_slice::<f64>()
            .iter()
            .map(|&v| {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "f64 values outside f32 range become infinity, which is acceptable \
                              because callers operate in f32 and document this constraint"
                )]
                let v = v as f32;
                v
            })
            .collect()),
        other => vortex_bail!("expected float elements, got {other:?}"),
    }
}

/// The flat primitive elements of a tensor storage array, with typed row access.
///
/// This struct hides the stride detail that arises from the [`ConstantArray`] optimization: a
/// constant-backed input materializes only a single row that every index reads (`is_constant =
/// true`), while a full array stores one row per index.
pub struct FlatElements {
    elems: PrimitiveArray,
    list_size: usize,
    is_constant: bool,
}

impl FlatElements {
    /// Returns the [`PType`] of the underlying elements.
    #[must_use]
    pub fn ptype(&self) -> PType {
        self.elems.ptype()
    }

    /// Returns the `i`-th row as a typed slice of length `list_size`.
    ///
    /// When the source was a constant-backed storage, all indices resolve to the single stored
    /// row.
    #[must_use]
    pub fn row<T: NativePType>(&self, i: usize) -> &[T] {
        let row_idx = if self.is_constant { 0 } else { i };
        let slice = self.elems.as_slice::<T>();
        &slice[row_idx * self.list_size..][..self.list_size]
    }
}

/// Extracts the flat primitive elements from a tensor storage array (FixedSizeList).
///
/// When the input is a [`ConstantArray`] (e.g., a literal query vector), only a single row is
/// materialized to avoid expanding it to the full column length. Callers that have already
/// confirmed the storage is constant-backed should prefer [`extract_constant_flat_row`].
pub fn extract_flat_elements(
    storage: &ArrayRef,
    list_size: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<FlatElements> {
    // Constant-backed storage: materialize just the single stored row so canonicalization does
    // not expand the array to the full column length.
    let (source, is_constant) = if let Some(constant) = storage.as_opt::<Constant>() {
        let single = ConstantArray::new(constant.scalar().clone(), 1).into_array();
        (single, true)
    } else {
        (storage.clone(), false)
    };

    let fsl: FixedSizeListArray = source.execute(ctx)?;
    let elems: PrimitiveArray = fsl.elements().clone().execute(ctx)?;
    vortex_ensure!(
        !elems.nullability().is_nullable(),
        "tensor storage elements must be non-nullable, got {}",
        elems.dtype(),
    );
    Ok(FlatElements {
        elems,
        list_size,
        is_constant,
    })
}

/// The single stored row of a constant-backed tensor storage array.
///
/// Contrast with [`FlatElements`], which exposes arbitrary row indices: a `FlatRow` statically
/// encodes "there is exactly one row available," so call sites that have gated on a constant input
/// read the row via [`Self::as_slice`] instead of `row(0)`.
pub struct FlatRow {
    elems: PrimitiveArray,
}

impl FlatRow {
    /// Returns the [`PType`] of the underlying elements.
    #[must_use]
    pub fn ptype(&self) -> PType {
        self.elems.ptype()
    }

    /// Returns the stored row as a typed slice. Its length equals the storage scalar's
    /// fixed-size-list size.
    #[must_use]
    pub fn as_slice<T: NativePType>(&self) -> &[T] {
        self.elems.as_slice::<T>()
    }
}

/// Extracts the single stored row from a [`Constant`]-backed tensor storage array.
///
/// The caller must have confirmed that `storage` is a [`Constant`] encoding whose scalar is a
/// non-null fixed-size list. This is the fast path for constant query vectors: exactly one row is
/// materialized regardless of the column length.
///
/// # Panics
///
/// Panics if `storage` is not a [`Constant`] encoding.
pub fn extract_constant_flat_row(
    storage: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<FlatRow> {
    let constant = storage
        .as_opt::<Constant>()
        .vortex_expect("extract_constant_flat_row requires Constant-backed storage");
    let single = ConstantArray::new(constant.scalar().clone(), 1).into_array();
    let fsl: FixedSizeListArray = single.execute(ctx)?;
    let elems: PrimitiveArray = fsl.elements().clone().execute(ctx)?;
    vortex_ensure!(
        !elems.nullability().is_nullable(),
        "tensor storage elements must be non-nullable, got {}",
        elems.dtype(),
    );
    Ok(FlatRow { elems })
}

/// Extracts the `(normalized, norms)` children from an [`L2Denorm`] scalar function array.
///
/// [`L2Denorm`]: crate::scalar_fns::l2_denorm::L2Denorm
pub fn extract_l2_denorm_children(array: &ArrayRef) -> (ArrayRef, ArrayRef) {
    let sfn = array
        .as_opt::<ExactScalarFn<L2Denorm>>()
        .vortex_expect("expected ScalarFnArray wrapping L2Denorm");
    (
        sfn.nth_child(0)
            .vortex_expect("L2Denorm missing normalized array"),
        sfn.nth_child(1).vortex_expect("L2Denorm missing norms"),
    )
}

#[cfg(test)]
pub mod test_helpers {
    use vortex_array::ArrayRef;
    use vortex_array::ExecutionCtx;
    use vortex_array::IntoArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::ExtensionArray;
    use vortex_array::arrays::FixedSizeListArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::NativePType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::extension::ExtDType;
    use vortex_array::scalar::PValue;
    use vortex_array::scalar::Scalar;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;

    use crate::fixed_shape::FixedShapeTensor;
    use crate::fixed_shape::FixedShapeTensorMetadata;
    use crate::scalar_fns::l2_denorm::L2Denorm;
    use crate::vector::Vector;

    /// Builds a `FixedSizeList<T, list_size>` storage array from flat `elements`. The row count is
    /// inferred from `elements.len() / list_size`.
    fn flat_fsl<T: NativePType>(elements: &[T], list_size: u32) -> ArrayRef {
        let row_count = elements.len() / list_size as usize;
        let elems: ArrayRef = Buffer::copy_from(elements).into_array();
        FixedSizeListArray::new(elems, list_size, Validity::NonNullable, row_count).into_array()
    }

    /// Builds an FSL-valued [`Scalar`] from `elements` for use as a constant query.
    fn fsl_scalar<T: NativePType + Into<PValue>>(elements: &[T]) -> Scalar {
        let element_dtype = DType::Primitive(T::PTYPE, Nullability::NonNullable);
        let children: Vec<Scalar> = elements
            .iter()
            .map(|&v| Scalar::primitive(v, Nullability::NonNullable))
            .collect();
        Scalar::fixed_size_list(element_dtype, children, Nullability::NonNullable)
    }

    /// Builds a [`FixedShapeTensor`] extension array from flat `elements` and a logical shape.
    ///
    /// The number of rows is inferred from the total element count divided by the product of the
    /// shape dimensions. For 0-dimensional tensors (scalar), each element is one row.
    pub fn tensor_array<T: NativePType>(shape: &[usize], elements: &[T]) -> VortexResult<ArrayRef> {
        let list_size: u32 = shape.iter().product::<usize>().max(1).try_into().unwrap();
        let storage = flat_fsl(elements, list_size);
        let metadata = FixedShapeTensorMetadata::new(shape.to_vec());
        let ext_dtype =
            ExtDType::<FixedShapeTensor>::try_new(metadata, storage.dtype().clone())?.erased();
        Ok(ExtensionArray::new(ext_dtype, storage).into_array())
    }

    /// Builds a [`Vector`] extension array from flat `elements` and a vector dimension size.
    pub fn vector_array<T: NativePType>(dim: u32, elements: &[T]) -> VortexResult<ArrayRef> {
        Vector::try_new_vector_array(flat_fsl(elements, dim))
    }

    /// Builds a [`FixedShapeTensor`] extension array whose storage is a [`ConstantArray`],
    /// representing a single query tensor broadcast to `len` rows.
    pub fn constant_tensor_array<T: NativePType + Into<PValue>>(
        shape: &[usize],
        elements: &[T],
        len: usize,
    ) -> VortexResult<ArrayRef> {
        let storage = ConstantArray::new(fsl_scalar(elements), len).into_array();
        let metadata = FixedShapeTensorMetadata::new(shape.to_vec());
        let ext_dtype =
            ExtDType::<FixedShapeTensor>::try_new(metadata, storage.dtype().clone())?.erased();
        Ok(ExtensionArray::new(ext_dtype, storage).into_array())
    }

    /// Builds a [`ConstantArray`] whose scalar is itself a [`Vector`] extension scalar, broadcast
    /// to `len` rows. This is the shape produced by an `lit(vector_scalar)` literal expression —
    /// the constant lives at the extension level rather than inside the FSL storage, in contrast
    /// to [`Vector::constant_array`].
    pub fn literal_vector_array<T: NativePType + Into<PValue>>(
        elements: &[T],
        len: usize,
    ) -> ArrayRef {
        use vortex_array::extension::EmptyMetadata;
        let ext_scalar = Scalar::extension::<Vector>(EmptyMetadata, fsl_scalar(elements));
        ConstantArray::new(ext_scalar, len).into_array()
    }

    /// Creates an [`L2Denorm`] scalar function array from pre-normalized tensor elements and
    /// matching norms. The caller must ensure every row of `normalized_elements` is unit-norm or
    /// zero.
    pub fn l2_denorm_array<T: NativePType>(
        shape: &[usize],
        normalized_elements: &[T],
        norms: &[T],
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let len = norms.len();
        let normalized = tensor_array(shape, normalized_elements)?;
        let norms =
            PrimitiveArray::new(Buffer::copy_from(norms), Validity::NonNullable).into_array();
        Ok(L2Denorm::try_new_array(normalized, norms, len, ctx)?.into_array())
    }

    /// Asserts that each element in `actual` is within `1e-10` of the corresponding `expected`
    /// value, with support for NaN (NaN == NaN is considered equal).
    #[track_caller]
    pub fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "length mismatch: got {} elements, expected {}",
            actual.len(),
            expected.len()
        );

        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            if a.is_nan() && e.is_nan() {
                continue;
            }
            assert!(
                (a - e).abs() < 1e-10,
                "element {i}: got {a}, expected {e} (diff = {})",
                (a - e).abs()
            );
        }
    }
}
