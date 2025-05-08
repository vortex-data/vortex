use std::sync::Arc;

use vortex_dtype::{DType, FieldName};
use vortex_error::{VortexExpect, VortexResult, vortex_err, vortex_panic};
use vortex_scalar::PValue;

use crate::arrays::chunked::ChunkedArray;
use crate::variants::{
    BinaryArrayTrait, BoolArrayTrait, DecimalArrayTrait, ExtensionArrayTrait, ListArrayTrait,
    NullArrayTrait, PrimitiveArrayTrait, StructArrayTrait, Utf8ArrayTrait,
};
use crate::{Array, ArrayRef, ArrayVariants, ArrayVariantsImpl};

/// Chunked arrays support all DTypes
impl ArrayVariantsImpl for ChunkedArray {
    fn _as_null_typed(&self) -> Option<&dyn NullArrayTrait> {
        Some(self)
    }

    fn _as_bool_typed(&self) -> Option<&dyn BoolArrayTrait> {
        Some(self)
    }

    fn _as_primitive_typed(&self) -> Option<&dyn PrimitiveArrayTrait> {
        Some(self)
    }

    fn _as_decimal_typed(&self) -> Option<&dyn DecimalArrayTrait> {
        Some(self)
    }

    fn _as_utf8_typed(&self) -> Option<&dyn Utf8ArrayTrait> {
        Some(self)
    }

    fn _as_binary_typed(&self) -> Option<&dyn BinaryArrayTrait> {
        Some(self)
    }

    fn _as_struct_typed(&self) -> Option<&dyn StructArrayTrait> {
        Some(self)
    }

    fn _as_list_typed(&self) -> Option<&dyn ListArrayTrait> {
        Some(self)
    }

    fn _as_extension_typed(&self) -> Option<&dyn ExtensionArrayTrait> {
        Some(self)
    }
}

impl NullArrayTrait for ChunkedArray {}

impl BoolArrayTrait for ChunkedArray {}

impl PrimitiveArrayTrait for ChunkedArray {
    fn value_unchecked(&self, idx: usize) -> PValue {
        let (chunk, offset_in_chunk) = self.find_chunk_idx(idx);
        let chunk = self
            .chunks()
            .get(chunk)
            .vortex_expect("Chunk index out of bounds");
        chunk
            .as_primitive_typed()
            .vortex_expect("Chunk was not a PrimitiveArray")
            .value_unchecked(offset_in_chunk)
    }
}

impl DecimalArrayTrait for ChunkedArray {}

impl Utf8ArrayTrait for ChunkedArray {}

impl BinaryArrayTrait for ChunkedArray {}

impl StructArrayTrait for ChunkedArray {
    fn maybe_null_field_by_idx(&self, idx: usize) -> VortexResult<ArrayRef> {
        let mut chunks = Vec::with_capacity(self.nchunks());
        for chunk in self.chunks() {
            chunks.push(
                chunk
                    .as_struct_typed()
                    .ok_or_else(|| vortex_err!("Chunk was not a StructArray"))?
                    .maybe_null_field_by_idx(idx)?,
            );
        }

        let projected_dtype = self
            .dtype()
            .as_struct()
            .ok_or_else(|| vortex_err!("Not a struct dtype"))?
            .field_by_index(idx)?;
        let chunked = ChunkedArray::try_new(chunks, projected_dtype.clone())
            .unwrap_or_else(|err| {
                vortex_panic!(
                    err,
                    "Failed to create new chunked array with dtype {}",
                    projected_dtype
                )
            })
            .into_array();
        Ok(chunked)
    }

    fn project(&self, projection: &[FieldName]) -> VortexResult<ArrayRef> {
        let mut chunks = Vec::with_capacity(self.nchunks());
        for chunk in self.chunks() {
            chunks.push(
                chunk
                    .as_struct_typed()
                    .ok_or_else(|| vortex_err!("Chunk was not a StructArray"))?
                    .project(projection)?,
            );
        }

        let projected_dtype = self
            .dtype()
            .as_struct()
            .ok_or_else(|| vortex_err!("Not a struct dtype"))?
            .project(projection)?;
        Ok(ChunkedArray::new_unchecked(
            chunks,
            DType::Struct(Arc::new(projected_dtype), self.dtype().nullability()),
        )
        .into_array())
    }
}

impl ListArrayTrait for ChunkedArray {}

impl ExtensionArrayTrait for ChunkedArray {
    fn storage_data(&self) -> ArrayRef {
        ChunkedArray::new_unchecked(
            self.chunks()
                .iter()
                .map(|chunk| {
                    chunk
                        .as_extension_typed()
                        .vortex_expect("Expected extension array")
                        .storage_data()
                })
                .collect(),
            self.ext_dtype().storage_dtype().clone(),
        )
        .into_array()
    }
}
