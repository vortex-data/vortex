// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use prost::Message;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::ScalarFn;
use vortex_array::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::arrays::scalar_fn::ScalarFnArrayExt;
use vortex_array::arrays::scalar_fn::ScalarFnArrayView;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::proto::dtype as pb;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::matcher::AnyTensor;
use crate::matcher::TensorMatch;

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

/// The flat primitive elements of a tensor storage array, with typed row access and its physical
/// row layout.
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

    /// Returns the number of elements in each row.
    #[must_use]
    pub fn list_size(&self) -> usize {
        self.list_size
    }

    /// Returns the physical distance between rows, or zero when every row uses one stored value.
    #[must_use]
    pub fn row_stride(&self) -> usize {
        if self.is_constant { 0 } else { self.list_size }
    }

    /// Returns the elements as a typed buffer, performing the ptype check once for the batch.
    pub fn into_buffer<T: NativePType>(self) -> Buffer<T> {
        self.elems.into_buffer::<T>()
    }
}

/// Extracts the flat primitive elements from a tensor storage array (FixedSizeList).
///
/// When the input is a [`ConstantArray`] (e.g., a literal query vector), only a single row is
/// materialized to avoid expanding it to the full column length.
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

/// Metadata for a serialized binary tensor-op array (shared by [`InnerProduct`] and
/// [`CosineSimilarity`]). Both operands share the same extension dtype up to nullability
/// (enforced by their `return_dtype` checks), but their individual nullabilities are lost in the
/// parent's unioned output, so both are persisted.
///
/// [`CosineSimilarity`]: crate::scalar_fns::cosine_similarity::CosineSimilarity
/// [`InnerProduct`]: crate::scalar_fns::inner_product::InnerProduct
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
    use vortex_array::IntoArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::ExtensionArray;
    use vortex_array::arrays::FixedSizeListArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::NativePType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::extension::ExtDType;
    use vortex_array::scalar::PValue;
    use vortex_array::scalar::Scalar;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;

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

    /// Builds `row_count` zero-width vectors over an empty typed element buffer.
    pub fn zero_width_vector_array<T: NativePType>(row_count: usize) -> VortexResult<ArrayRef> {
        let storage = FixedSizeListArray::new(
            Buffer::<T>::empty().into_array(),
            0,
            Validity::NonNullable,
            row_count,
        )
        .into_array();
        Vector::try_new_vector_array(storage)
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
