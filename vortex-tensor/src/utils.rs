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
/// constant input materializes only a single row (stride=0), while a full array uses
/// stride=list_size.
pub struct FlatElements {
    elems: PrimitiveArray,
    stride: usize,
    list_size: usize,
}

impl FlatElements {
    /// Returns the [`PType`] of the underlying elements.
    #[must_use]
    pub fn ptype(&self) -> PType {
        self.elems.ptype()
    }

    /// Returns the `i`-th row as a typed slice of length `list_size`.
    #[must_use]
    pub fn row<T: NativePType>(&self, i: usize) -> &[T] {
        let slice = self.elems.as_slice::<T>();
        &slice[i * self.stride..][..self.list_size]
    }
}

// TODO(connor): Usage of this function is sometimes incorrect / not performant.
// Make sure to fix them.
/// Extracts the flat primitive elements from a tensor storage array (FixedSizeList).
///
/// When the input is a [`ConstantArray`] (e.g., a literal query vector), only a single row is
/// materialized to avoid expanding it to the full column length.
pub fn extract_flat_elements(
    storage: &ArrayRef,
    list_size: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<FlatElements> {
    if let Some(constant) = storage.as_opt::<Constant>() {
        // Rewrite the array as a length 1 array so when we canonicalize, we do not duplicate a huge
        // amount of data.
        let single = ConstantArray::new(constant.scalar().clone(), 1).into_array();
        let fsl: FixedSizeListArray = single.execute(ctx)?;
        let elems: PrimitiveArray = fsl.elements().clone().execute(ctx)?;
        return Ok(FlatElements {
            elems,
            stride: 0,
            list_size,
        });
    }

    // Otherwise we have to fully expand all of the data.
    let fsl: FixedSizeListArray = storage.clone().execute(ctx)?;
    let elems: PrimitiveArray = fsl.elements().clone().execute(ctx)?;
    Ok(FlatElements {
        elems,
        stride: list_size,
        list_size,
    })
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
    use vortex_array::IntoArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::ExtensionArray;
    use vortex_array::arrays::FixedSizeListArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::extension::ExtDType;
    use vortex_array::extension::EmptyMetadata;
    use vortex_array::scalar::Scalar;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;

    use crate::fixed_shape::FixedShapeTensor;
    use crate::fixed_shape::FixedShapeTensorMetadata;
    use crate::vector::Vector;

    /// Builds a [`FixedShapeTensor`] extension array from flat f64 elements and a logical shape.
    ///
    /// The number of rows is inferred from the total element count divided by the product of the
    /// shape dimensions. For 0-dimensional tensors (scalar), each element is one row.
    pub fn tensor_array(shape: &[usize], elements: &[f64]) -> VortexResult<ArrayRef> {
        let list_size: u32 = shape.iter().product::<usize>().max(1).try_into().unwrap();
        let row_count = elements.len() / list_size as usize;

        let elems: ArrayRef = Buffer::copy_from(elements).into_array();
        let fsl = FixedSizeListArray::new(elems, list_size, Validity::NonNullable, row_count);

        let metadata = FixedShapeTensorMetadata::new(shape.to_vec());
        let ext_dtype =
            ExtDType::<FixedShapeTensor>::try_new(metadata, fsl.dtype().clone())?.erased();

        Ok(ExtensionArray::new(ext_dtype, fsl.into_array()).into_array())
    }

    /// Builds a [`Vector`] extension array from flat f64 elements and a vector dimension size.
    pub fn vector_array(dim: u32, elements: &[f64]) -> VortexResult<ArrayRef> {
        let row_count = elements.len() / dim as usize;

        let elems: ArrayRef = Buffer::copy_from(elements).into_array();
        let fsl = FixedSizeListArray::new(elems, dim, Validity::NonNullable, row_count);

        let ext_dtype = ExtDType::<Vector>::try_new(EmptyMetadata, fsl.dtype().clone())?.erased();

        Ok(ExtensionArray::new(ext_dtype, fsl.into_array()).into_array())
    }

    /// Builds a [`FixedShapeTensor`] extension array whose storage is a [`ConstantArray`],
    /// representing a single query tensor broadcast to `len` rows.
    pub fn constant_tensor_array(
        shape: &[usize],
        elements: &[f64],
        len: usize,
    ) -> VortexResult<ArrayRef> {
        let element_dtype = DType::Primitive(PType::F64, Nullability::NonNullable);

        let children: Vec<Scalar> = elements
            .iter()
            .map(|&v| Scalar::primitive(v, Nullability::NonNullable))
            .collect();
        let storage_scalar =
            Scalar::fixed_size_list(element_dtype, children, Nullability::NonNullable);

        let storage = ConstantArray::new(storage_scalar, len).into_array();

        let metadata = FixedShapeTensorMetadata::new(shape.to_vec());
        let ext_dtype =
            ExtDType::<FixedShapeTensor>::try_new(metadata, storage.dtype().clone())?.erased();

        Ok(ExtensionArray::new(ext_dtype, storage).into_array())
    }

    /// Builds a [`Vector`] extension array whose storage is a [`ConstantArray`], representing a
    /// single query vector broadcast to `len` rows.
    pub fn constant_vector_array(elements: &[f64], len: usize) -> VortexResult<ArrayRef> {
        let element_dtype = DType::Primitive(PType::F64, Nullability::NonNullable);

        let children: Vec<Scalar> = elements
            .iter()
            .map(|&v| Scalar::primitive(v, Nullability::NonNullable))
            .collect();
        let storage_scalar =
            Scalar::fixed_size_list(element_dtype, children, Nullability::NonNullable);

        let storage = ConstantArray::new(storage_scalar, len).into_array();

        let ext_dtype =
            ExtDType::<Vector>::try_new(EmptyMetadata, storage.dtype().clone())?.erased();

        Ok(ExtensionArray::new(ext_dtype, storage).into_array())
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
