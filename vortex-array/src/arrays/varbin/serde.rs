use arrow_schema::DataType;
use vortex_dtype::{DType, Nullability, PType};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};

use super::VarBinEncoding;
use crate::arrays::VarBinArray;
use crate::arrow::{FromArrowArray, IntoArrowArray};
use crate::serde::ArrayParts;
use crate::validity::Validity;
use crate::vtable::EncodingVTable;
use crate::{
    Array, ArrayBufferVisitor, ArrayChildVisitor, ArrayContext, ArrayRef, ArrayVisitorImpl,
    Canonical, DeserializeMetadata, EncodingId, IntoArray, ProstMetadata,
};

#[derive(Clone, prost::Message)]
pub struct VarBinMetadata {
    #[prost(enumeration = "PType", tag = "1")]
    pub(crate) offsets_ptype: i32,
}

impl EncodingVTable for VarBinEncoding {
    fn id(&self) -> EncodingId {
        EncodingId::new_ref("vortex.varbin")
    }

    fn decode(
        &self,
        parts: &ArrayParts,
        ctx: &ArrayContext,
        dtype: DType,
        len: usize,
    ) -> VortexResult<ArrayRef> {
        let metadata = ProstMetadata::<VarBinMetadata>::deserialize(parts.metadata())?;

        let validity = if parts.nchildren() == 1 {
            Validity::from(dtype.nullability())
        } else if parts.nchildren() == 2 {
            let validity = parts.child(1).decode(ctx, Validity::DTYPE, len)?;
            Validity::Array(validity)
        } else {
            vortex_bail!("Expected 1 or 2 children, got {}", parts.nchildren());
        };

        let offsets = parts.child(0).decode(
            ctx,
            DType::Primitive(metadata.offsets_ptype(), Nullability::NonNullable),
            len + 1,
        )?;

        if parts.nbuffers() != 1 {
            vortex_bail!("Expected 1 buffer, got {}", parts.nbuffers());
        }
        let bytes = parts.buffer(0)?;

        Ok(VarBinArray::try_new(offsets, bytes, dtype, validity)?.into_array())
    }

    fn encode(
        &self,
        input: &Canonical,
        _like: Option<&dyn Array>,
    ) -> VortexResult<Option<ArrayRef>> {
        let arrow_array = input.clone().into_array().into_arrow_preferred()?;
        let array = match arrow_array.data_type() {
            DataType::Utf8View => arrow_cast::cast(arrow_array.as_ref(), &DataType::Utf8)?,
            DataType::BinaryView => arrow_cast::cast(arrow_array.as_ref(), &DataType::Binary)?,
            _ => unreachable!("VarBinArray must have Utf8 or Binary dtype"),
        };
        Ok(Some(ArrayRef::from_arrow(
            array,
            input.as_ref().dtype().nullability().into(),
        )))
    }
}

impl ArrayVisitorImpl<ProstMetadata<VarBinMetadata>> for VarBinArray {
    fn _visit_buffers(&self, visitor: &mut dyn ArrayBufferVisitor) {
        visitor.visit_buffer(self.bytes()); // TODO(ngates): sliced bytes?
    }

    fn _visit_children(&self, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("offsets", self.offsets());
        visitor.visit_validity(self.validity(), self.len());
    }

    fn _metadata(&self) -> ProstMetadata<VarBinMetadata> {
        ProstMetadata(VarBinMetadata {
            offsets_ptype: PType::try_from(self.offsets().dtype())
                .vortex_expect("Must be a valid PType") as i32,
        })
    }
}
