use std::sync::Arc;

use vortex_dtype::{DType, FieldName};
use vortex_error::{vortex_err, vortex_panic, VortexExpect, VortexResult};

use crate::arrays::chunked::ChunkedArray;
use crate::arrays::ChunkedEncoding;
use crate::variants::{
    BinaryArrayTrait, BoolArrayTrait, ExtensionArrayTrait, ListArrayTrait, NullArrayTrait,
    PrimitiveArrayTrait, StructArrayTrait, Utf8ArrayTrait,
};
use crate::vtable::VariantsVTable;
use crate::{Array, IntoArray};

/// Chunked arrays support all DTypes
impl VariantsVTable<ChunkedArray> for ChunkedEncoding {
    fn as_null_array<'a>(&self, array: &'a ChunkedArray) -> Option<&'a dyn NullArrayTrait> {
        Some(array)
    }

    fn as_bool_array<'a>(&self, array: &'a ChunkedArray) -> Option<&'a dyn BoolArrayTrait> {
        Some(array)
    }

    fn as_primitive_array<'a>(
        &self,
        array: &'a ChunkedArray,
    ) -> Option<&'a dyn PrimitiveArrayTrait> {
        Some(array)
    }

    fn as_utf8_array<'a>(&self, array: &'a ChunkedArray) -> Option<&'a dyn Utf8ArrayTrait> {
        Some(array)
    }

    fn as_binary_array<'a>(&self, array: &'a ChunkedArray) -> Option<&'a dyn BinaryArrayTrait> {
        Some(array)
    }

    fn as_struct_array<'a>(&self, array: &'a ChunkedArray) -> Option<&'a dyn StructArrayTrait> {
        Some(array)
    }

    fn as_list_array<'a>(&self, array: &'a ChunkedArray) -> Option<&'a dyn ListArrayTrait> {
        Some(array)
    }

    fn as_extension_array<'a>(
        &self,
        array: &'a ChunkedArray,
    ) -> Option<&'a dyn ExtensionArrayTrait> {
        Some(array)
    }
}

impl NullArrayTrait for ChunkedArray {}

impl BoolArrayTrait for ChunkedArray {}

impl PrimitiveArrayTrait for ChunkedArray {}

impl Utf8ArrayTrait for ChunkedArray {}

impl BinaryArrayTrait for ChunkedArray {}

impl StructArrayTrait for ChunkedArray {
    fn maybe_null_field_by_idx(&self, idx: usize) -> VortexResult<Array> {
        let mut chunks = Vec::with_capacity(self.nchunks());
        for chunk in self.chunks() {
            chunks.push(
                chunk
                    .as_struct_array()
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

    fn project(&self, projection: &[FieldName]) -> VortexResult<Array> {
        let mut chunks = Vec::with_capacity(self.nchunks());
        for chunk in self.chunks() {
            chunks.push(
                chunk
                    .as_struct_array()
                    .ok_or_else(|| vortex_err!("Chunk was not a StructArray"))?
                    .project(projection)?,
            );
        }

        let projected_dtype = self
            .dtype()
            .as_struct()
            .ok_or_else(|| vortex_err!("Not a struct dtype"))?
            .project(projection)?;
        Ok(ChunkedArray::try_new_unchecked(
            chunks,
            DType::Struct(Arc::new(projected_dtype), self.dtype().nullability()),
        )
        .into_array())
    }
}

impl ListArrayTrait for ChunkedArray {}

impl ExtensionArrayTrait for ChunkedArray {
    fn storage_data(&self) -> Array {
        ChunkedArray::from_iter(self.chunks().map(|chunk| {
            chunk
                .as_extension_array()
                .vortex_expect("Expected extension array")
                .storage_data()
        }))
        .into_array()
    }
}
