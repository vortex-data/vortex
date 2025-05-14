use std::fmt::{Debug, Formatter};

use vortex_array::serde::ArrayChildren;
use vortex_array::vtable::{EncodeVTable, SerdeVTable, VisitorVTable};
use vortex_array::{
    ArrayBufferVisitor, ArrayChildVisitor, Canonical, DeserializeMetadata, SerializeMetadata,
};
use vortex_buffer::ByteBuffer;
use vortex_dtype::{DType, PType};
use vortex_error::{VortexResult, vortex_bail};
use vortex_scalar::{Scalar, ScalarValue};

use super::FoREncoding;
use crate::{FoRArray, FoRVTable};

impl SerdeVTable<FoRVTable> for FoRVTable {
    type Metadata = ScalarValueMetadata;

    fn metadata(array: &FoRArray) -> VortexResult<Option<Self::Metadata>> {
        Ok(Some(ScalarValueMetadata(
            array.reference_scalar().value().clone(),
        )))
    }

    fn build(
        _encoding: &FoREncoding,
        dtype: &DType,
        len: usize,
        metadata: &ScalarValue,
        _buffers: &[ByteBuffer],
        children: &dyn ArrayChildren,
    ) -> VortexResult<FoRArray> {
        if children.len() != 1 {
            vortex_bail!(
                "Expected 1 child for FoR encoding, found {}",
                children.len()
            )
        }

        let ptype = PType::try_from(dtype)?;
        let encoded_dtype = DType::Primitive(ptype.to_unsigned(), dtype.nullability());
        let encoded = children.get(0, &encoded_dtype, len)?;

        let reference = Scalar::new(dtype.clone(), metadata.clone());

        FoRArray::try_new(encoded, reference)
    }
}

impl EncodeVTable<FoRVTable> for FoRVTable {
    fn encode(
        _encoding: &FoREncoding,
        canonical: &Canonical,
        _like: Option<&FoRArray>,
    ) -> VortexResult<Option<FoRArray>> {
        let parray = canonical.clone().into_primitive()?;
        Ok(Some(FoRArray::encode(parray)?))
    }
}

impl VisitorVTable<FoRVTable> for FoRVTable {
    fn visit_buffers(_array: &FoRArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn visit_children(array: &FoRArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("encoded", array.encoded())
    }
}

#[derive(Clone)]
pub struct ScalarValueMetadata(ScalarValue);

impl SerializeMetadata for ScalarValueMetadata {
    fn serialize(self) -> Vec<u8> {
        self.0.to_protobytes()
    }
}

impl DeserializeMetadata for ScalarValueMetadata {
    type Output = ScalarValue;

    fn deserialize(metadata: &[u8]) -> VortexResult<Self::Output> {
        ScalarValue::from_protobytes(metadata)
    }
}

impl Debug for ScalarValueMetadata {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", &self.0)
    }
}
