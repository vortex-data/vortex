use vortex_dtype::field::Field;
use vortex_dtype::DType;
use vortex_error::{vortex_err, vortex_panic, VortexResult};

use crate::array::chunked::ChunkedArray;
use crate::variants::{
    ArrayVariants, BinaryArrayTrait, BoolArrayTrait, ExtensionArrayTrait, ListArrayTrait,
    NullArrayTrait, PrimitiveArrayTrait, StructArrayTrait, Utf8ArrayTrait,
};
use crate::{ArrayDType, ArrayData, IntoArrayData};

/// Chunked arrays support all DTypes
impl ArrayVariants for ChunkedArray {
    fn as_null_array(&self) -> Option<&dyn NullArrayTrait> {
        matches!(self.dtype(), DType::Null).then_some(self)
    }

    fn as_bool_array(&self) -> Option<&dyn BoolArrayTrait> {
        matches!(self.dtype(), DType::Bool(_)).then_some(self)
    }

    fn as_primitive_array(&self) -> Option<&dyn PrimitiveArrayTrait> {
        matches!(self.dtype(), DType::Primitive(..)).then_some(self)
    }

    fn as_utf8_array(&self) -> Option<&dyn Utf8ArrayTrait> {
        matches!(self.dtype(), DType::Utf8(_)).then_some(self)
    }

    fn as_binary_array(&self) -> Option<&dyn BinaryArrayTrait> {
        matches!(self.dtype(), DType::Binary(_)).then_some(self)
    }

    fn as_struct_array(&self) -> Option<&dyn StructArrayTrait> {
        matches!(self.dtype(), DType::Struct(..)).then_some(self)
    }

    fn as_list_array(&self) -> Option<&dyn ListArrayTrait> {
        matches!(self.dtype(), DType::List(..)).then_some(self)
    }

    fn as_extension_array(&self) -> Option<&dyn ExtensionArrayTrait> {
        matches!(self.dtype(), DType::Extension(..)).then_some(self)
    }
}

impl NullArrayTrait for ChunkedArray {}

impl BoolArrayTrait for ChunkedArray {
    fn invert(&self) -> VortexResult<ArrayData> {
        let chunks = self
            .chunks()
            .map(|c| {
                c.with_dyn(|a| {
                    a.as_bool_array()
                        .ok_or_else(|| vortex_err!("Child was not a bool array"))
                        .and_then(|b| b.invert())
                })
            })
            .collect::<VortexResult<Vec<_>>>()?;
        ChunkedArray::try_new(chunks, self.dtype().clone()).map(|a| a.into_array())
    }
}

impl PrimitiveArrayTrait for ChunkedArray {}

impl Utf8ArrayTrait for ChunkedArray {}

impl BinaryArrayTrait for ChunkedArray {}

impl StructArrayTrait for ChunkedArray {
    fn field(&self, idx: usize) -> Option<ArrayData> {
        let mut chunks = Vec::with_capacity(self.nchunks());
        for chunk in self.chunks() {
            chunks.push(chunk.with_dyn(|a| a.as_struct_array().and_then(|s| s.field(idx)))?);
        }

        let projected_dtype = self.dtype().as_struct().and_then(|s| s.dtypes().get(idx))?;
        let chunked = ChunkedArray::try_new(chunks, projected_dtype.clone())
            .unwrap_or_else(|err| {
                vortex_panic!(
                    err,
                    "Failed to create new chunked array with dtype {}",
                    projected_dtype
                )
            })
            .into_array();
        Some(chunked)
    }

    fn project(&self, projection: &[Field]) -> VortexResult<ArrayData> {
        let mut chunks = Vec::with_capacity(self.nchunks());
        for chunk in self.chunks() {
            chunks.push(chunk.with_dyn(|a| {
                a.as_struct_array()
                    .ok_or_else(|| vortex_err!("Chunk was not a StructArray"))?
                    .project(projection)
            })?);
        }

        let projected_dtype = self
            .dtype()
            .as_struct()
            .ok_or_else(|| vortex_err!("Not a struct dtype"))?
            .project(projection)?;
        ChunkedArray::try_new(
            chunks,
            DType::Struct(projected_dtype, self.dtype().nullability()),
        )
        .map(|a| a.into_array())
    }
}

impl ListArrayTrait for ChunkedArray {}

impl ExtensionArrayTrait for ChunkedArray {
    fn storage_data(&self) -> ArrayData {
        ChunkedArray::from_iter(
            self.chunks()
                .map(|chunk| chunk.with_dyn(|a| a.as_extension_array_unchecked().storage_data())),
        )
        .into_array()
    }
}
