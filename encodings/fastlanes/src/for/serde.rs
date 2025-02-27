use std::fmt::Formatter;

use vortex_array::serde::ArrayParts;
use vortex_array::vtable::SerdeVTable;
use vortex_array::{
    Array, ArrayChildVisitor, ArrayContext, ArrayRef, ArrayVisitorImpl, DeserializeMetadata,
    SerializeMetadata,
};
use vortex_dtype::{DType, PType};
use vortex_error::{VortexResult, vortex_bail, vortex_err};
use vortex_scalar::{Scalar, ScalarValue};

use crate::{FoRArray, FoREncoding};

impl ArrayVisitorImpl<ScalarValueMetadata> for FoRArray {
    fn _children(&self, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("encoded", self.encoded())
    }

    fn _metadata(&self) -> ScalarValueMetadata {
        ScalarValueMetadata(self.reference_scalar().value().clone())
    }
}

impl SerdeVTable<&FoRArray> for FoREncoding {
    fn decode(
        &self,
        parts: &ArrayParts,
        ctx: &ArrayContext,
        dtype: DType,
        len: usize,
    ) -> VortexResult<ArrayRef> {
        if parts.nchildren() != 1 {
            vortex_bail!(
                "Expected 1 child for FoR encoding, found {}",
                parts.nchildren()
            )
        }

        let ptype = PType::try_from(&dtype)?;
        let encoded_dtype = DType::Primitive(ptype.to_unsigned(), dtype.nullability());
        let encoded = parts.child(0).decode(ctx, encoded_dtype, len)?;

        let reference = ScalarValue::from_flexbytes(
            parts
                .metadata()
                .ok_or_else(|| vortex_err!("Missing FoR metadata"))?,
        )?;
        let reference = Scalar::new(dtype, reference);

        Ok(FoRArray::try_new(encoded, reference)?.into_array())
    }
}

#[derive(Clone, Debug)]
pub struct ScalarValueMetadata(ScalarValue);

impl SerializeMetadata for ScalarValueMetadata {
    fn serialize(&self) -> Option<Vec<u8>> {
        Some(self.0.to_flexbytes())
    }
}

impl DeserializeMetadata for ScalarValueMetadata {
    type Output = ScalarValue;

    fn deserialize(metadata: Option<&[u8]>) -> VortexResult<Self::Output> {
        ScalarValue::from_flexbytes(
            metadata.ok_or_else(|| vortex_err!("Missing ScalarValue metadata"))?,
        )
    }

    fn format(metadata: Option<&[u8]>, f: &mut Formatter<'_>) -> std::fmt::Result {
        Self::deserialize(metadata)
            .map(|value| write!(f, "{}", value))
            .unwrap_or_else(|_| write!(f, "<unknown>"))
    }
}
