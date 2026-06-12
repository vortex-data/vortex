// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use half::f16;
use prost::Message;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::ScalarFn;
use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::arrays::scalar_fn::ExactScalarFn;
use vortex_array::arrays::scalar_fn::ScalarFnArrayExt;
use vortex_array::arrays::scalar_fn::ScalarFnArrayView;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_array::dtype::proto::dtype as pb;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::matcher::AnyTensor;
use crate::matcher::TensorMatch;
use crate::scalar_fns::l2_denorm::L2Denorm;

/// Safety factor for unit-norm tolerance. Applied as a constant multiplier on the probabilistic
/// `√d · ε` bound so that legitimate round-off noise clears the check with headroom.
pub(crate) const SAFETY_FACTOR: usize = 10;

/// Returns the acceptable unit-norm drift for the given element precision and dimension count.
///
/// Uses the `c · √d · ε` bound where ε is machine epsilon and d is the vector dimension. Under
/// IEEE 754 round-to-nearest the probabilistic (RMS-case) forward error for computing ‖x‖₂ grows
/// as `O(√d · ε)` rather than the worst-case `O(d · ε)` from the classical Wilkinson bound,
/// assuming near-independent rounding errors across the d-term summation.
///
/// Reference: Croci, Fasi, Higham, Mary, Mikaitis (2022). "Stochastic rounding: implementation,
/// error analysis and applications." Royal Society Open Science, 9: 211631, §6.1 "Probabilistic
/// error analysis." https://doi.org/10.1098/rsos.211631
pub fn unit_norm_tolerance(element_ptype: PType, dimensions: usize) -> f64 {
    let machine_epsilon: f64 = match element_ptype {
        PType::F64 => f64::EPSILON,
        PType::F32 => f32::EPSILON as f64,
        PType::F16 => f16::EPSILON.to_f64_const(),
        _ => unreachable!("unit_norm_tolerance requires a float ptype, got {element_ptype:?}"),
    };

    let dimensions_root = (dimensions as f64).sqrt();

    SAFETY_FACTOR as f64 * machine_epsilon * dimensions_root
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
pub fn validate_binary_tensor_float_inputs<'a>(
    lhs: &'a DType,
    rhs: &DType,
) -> VortexResult<TensorMatch<'a>> {
    vortex_ensure!(
        lhs.eq_ignore_nullability(rhs),
        "binary tensor expression expects inputs to have the same dtype, got {lhs} and {rhs}"
    );
    validate_tensor_float_input(lhs)
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

/// Metadata for a serialized binary tensor-op array (shared by [`InnerProduct`] and
/// [`CosineSimilarity`]). Both operands share the same extension dtype up to nullability
/// (enforced by their `return_dtype` checks), but their individual nullabilities are lost in the
/// parent's unioned output, so both are persisted.
///
/// [`CosineSimilarity`]: crate::scalar_fns::cosine_similarity::CosineSimilarity
#[derive(Clone, prost::Message)]
pub(crate) struct BinaryTensorOpMetadata {
    #[prost(message, optional, tag = "1")]
    pub(crate) lhs_dtype: Option<pb::DType>,
    #[prost(message, optional, tag = "2")]
    pub(crate) rhs_dtype: Option<pb::DType>,
}

impl BinaryTensorOpMetadata {
    /// Encodes the two children of `view` into a [`BinaryTensorOpMetadata`] byte blob.
    pub(crate) fn encode_from_view<V: ScalarFnVTable>(
        view: &ScalarFnArrayView<V>,
    ) -> VortexResult<Vec<u8>> {
        let scalar_fn_array = view.as_::<ScalarFn>();
        let lhs_dtype = Some(scalar_fn_array.child_at(0).dtype().try_into()?);
        let rhs_dtype = Some(scalar_fn_array.child_at(1).dtype().try_into()?);
        Ok(Self {
            lhs_dtype,
            rhs_dtype,
        }
        .encode_to_vec())
    }

    /// Decodes `metadata` and fetches both children from `children` using the decoded dtypes,
    /// validating that `lhs` and `rhs` are compatible tensor operands.
    pub(crate) fn decode_children(
        metadata: &[u8],
        len: usize,
        children: &dyn vortex_array::serde::ArrayChildren,
        session: &VortexSession,
    ) -> VortexResult<Vec<ArrayRef>> {
        let metadata = Self::decode(metadata)
            .map_err(|e| vortex_err!("Failed to decode BinaryTensorOpMetadata: {e}"))?;
        let lhs_pb = metadata
            .lhs_dtype
            .as_ref()
            .ok_or_else(|| vortex_err!("metadata missing lhs_dtype"))?;
        let rhs_pb = metadata
            .rhs_dtype
            .as_ref()
            .ok_or_else(|| vortex_err!("metadata missing rhs_dtype"))?;

        let lhs_dtype = DType::from_proto(lhs_pb, session)?;
        let rhs_dtype = DType::from_proto(rhs_pb, session)?;
        validate_binary_tensor_float_inputs(&lhs_dtype, &rhs_dtype)?;

        let lhs = children.get(0, &lhs_dtype, len)?;
        let rhs = children.get(1, &rhs_dtype, len)?;
        Ok(vec![lhs, rhs])
    }
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

    use crate::scalar_fns::l2_denorm::L2Denorm;
    use crate::types::fixed_shape_tensor::FixedShapeTensor;
    use crate::types::fixed_shape_tensor::FixedShapeTensorMetadata;
    use crate::types::vector::Vector;

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
        use vortex_array::EmptyMetadata;
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
        let normalized = tensor_array(shape, normalized_elements)?;
        let norms =
            PrimitiveArray::new(Buffer::copy_from(norms), Validity::NonNullable).into_array();
        Ok(L2Denorm::try_new_array(normalized, norms, ctx)?.into_array())
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
