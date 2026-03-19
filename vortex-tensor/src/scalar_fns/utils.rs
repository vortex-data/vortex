// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::Constant;
use vortex::array::arrays::ConstantArray;
use vortex::array::arrays::Extension;
use vortex::array::arrays::PrimitiveArray;
use vortex::dtype::DType;
use vortex::dtype::NativePType;
use vortex::dtype::PType;
use vortex::dtype::extension::ExtDTypeRef;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_ensure;
use vortex::error::vortex_err;

/// Extracts the list size from a tensor-like extension dtype.
///
/// The storage dtype must be a `FixedSizeList`.
pub fn extension_list_size(ext: &ExtDTypeRef) -> VortexResult<usize> {
    let DType::FixedSizeList(_, list_size, _) = ext.storage_dtype() else {
        vortex_bail!(
            "expected FixedSizeList storage dtype, got {}",
            ext.storage_dtype()
        );
    };

    Ok(*list_size as usize)
}

/// Extracts the float element [`PType`] from a tensor-like extension dtype.
///
/// The storage dtype must be a `FixedSizeList` of non-nullable primitives.
pub fn extension_element_ptype(ext: &ExtDTypeRef) -> VortexResult<PType> {
    let element_dtype = ext
        .storage_dtype()
        .as_fixed_size_list_element_opt()
        .ok_or_else(|| {
            vortex_err!(
                "expected FixedSizeList storage dtype, got {}",
                ext.storage_dtype()
            )
        })?;

    vortex_ensure!(
        !element_dtype.is_nullable(),
        "element dtype must be non-nullable"
    );

    Ok(element_dtype.as_ptype())
}

/// Extracts the storage array from an extension array without canonicalizing.
pub fn extension_storage(array: &ArrayRef) -> VortexResult<ArrayRef> {
    let ext = array
        .as_opt::<Extension>()
        .ok_or_else(|| vortex_err!("scalar_fn input must be an extension array"))?;

    Ok(ext.storage_array().clone())
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

/// Extracts the flat primitive elements from a tensor storage array (FixedSizeList).
///
/// When the input is a [`ConstantArray`] (e.g., a literal query vector), only a single row is
/// materialized to avoid expanding it to the full column length.
pub fn extract_flat_elements(storage: &ArrayRef, list_size: usize) -> VortexResult<FlatElements> {
    if let Some(constant) = storage.as_opt::<Constant>() {
        // Rewrite the array as a length 1 array so when we canonicalize, we do not duplicate a huge
        // amount of data.
        let single = ConstantArray::new(constant.scalar().clone(), 1).into_array();
        let fsl = single.to_canonical()?.into_fixed_size_list();
        let elems = fsl.elements().to_canonical()?.into_primitive();
        return Ok(FlatElements {
            elems,
            stride: 0,
            list_size,
        });
    }

    // Otherwise we have to fully expand all of the data.
    let fsl = storage.to_canonical()?.into_fixed_size_list();
    let elems = fsl.elements().to_canonical()?.into_primitive();
    Ok(FlatElements {
        elems,
        stride: list_size,
        list_size,
    })
}

#[cfg(test)]
pub mod test_helpers {
    use vortex::array::ArrayRef;
    use vortex::array::IntoArray;
    use vortex::array::arrays::ConstantArray;
    use vortex::array::arrays::ExtensionArray;
    use vortex::array::arrays::FixedSizeListArray;
    use vortex::array::validity::Validity;
    use vortex::buffer::Buffer;
    use vortex::dtype::DType;
    use vortex::dtype::Nullability;
    use vortex::dtype::extension::ExtDType;
    use vortex::error::VortexResult;
    use vortex::extension::EmptyMetadata;
    use vortex::scalar::Scalar;

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
        let element_dtype = DType::Primitive(vortex::dtype::PType::F64, Nullability::NonNullable);

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
        let element_dtype = DType::Primitive(vortex::dtype::PType::F64, Nullability::NonNullable);

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
