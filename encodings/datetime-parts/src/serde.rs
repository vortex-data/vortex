use vortex_array::arrays::TemporalArray;
use vortex_array::serde::ArrayChildren;
use vortex_array::vtable::{EncodeVTable, SerdeVTable, VisitorVTable};
use vortex_array::{
    Array, ArrayBufferVisitor, ArrayChildVisitor, Canonical, DeserializeMetadata, ProstMetadata,
};
use vortex_buffer::ByteBuffer;
use vortex_dtype::{DType, Nullability, PType};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};

use crate::{DateTimePartsArray, DateTimePartsEncoding, DateTimePartsVTable};

#[derive(Clone, prost::Message)]
#[repr(C)]
pub struct DateTimePartsMetadata {
    // Validity lives in the days array
    // TODO(ngates): we should actually model this with a Tuple array when we have one.
    #[prost(enumeration = "PType", tag = "1")]
    days_ptype: i32,
    #[prost(enumeration = "PType", tag = "2")]
    seconds_ptype: i32,
    #[prost(enumeration = "PType", tag = "3")]
    subseconds_ptype: i32,
}

impl SerdeVTable<DateTimePartsVTable> for DateTimePartsVTable {
    type Metadata = ProstMetadata<DateTimePartsMetadata>;

    fn metadata(array: &DateTimePartsArray) -> VortexResult<Option<Self::Metadata>> {
        Ok(Some(ProstMetadata(DateTimePartsMetadata {
            days_ptype: PType::try_from(array.days().dtype()).vortex_expect("Must be a valid PType")
                as i32,
            seconds_ptype: PType::try_from(array.seconds().dtype())
                .vortex_expect("Must be a valid PType") as i32,
            subseconds_ptype: PType::try_from(array.subseconds().dtype())
                .vortex_expect("Must be a valid PType") as i32,
        })))
    }

    fn build(
        _encoding: &DateTimePartsEncoding,
        dtype: &DType,
        len: usize,
        metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        _buffers: &[ByteBuffer],
        children: &dyn ArrayChildren,
    ) -> VortexResult<DateTimePartsArray> {
        if children.len() != 3 {
            vortex_bail!(
                "Expected 3 children for datetime-parts encoding, found {}",
                children.len()
            )
        }

        let days = children.get(
            0,
            &DType::Primitive(metadata.days_ptype(), dtype.nullability()),
            len,
        )?;
        let seconds = children.get(
            1,
            &DType::Primitive(metadata.seconds_ptype(), Nullability::NonNullable),
            len,
        )?;
        let subseconds = children.get(
            2,
            &DType::Primitive(metadata.subseconds_ptype(), Nullability::NonNullable),
            len,
        )?;

        DateTimePartsArray::try_new(dtype.clone(), days, seconds, subseconds)
    }
}

impl EncodeVTable<DateTimePartsVTable> for DateTimePartsVTable {
    fn encode(
        _encoding: &DateTimePartsEncoding,
        canonical: &Canonical,
        _like: Option<&DateTimePartsArray>,
    ) -> VortexResult<Option<DateTimePartsArray>> {
        let ext_array = canonical.clone().into_extension()?;
        let temporal = TemporalArray::try_from(ext_array)?;

        Ok(Some(DateTimePartsArray::try_from(temporal)?))
    }
}

impl VisitorVTable<DateTimePartsVTable> for DateTimePartsVTable {
    fn visit_buffers(_array: &DateTimePartsArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn visit_children(array: &DateTimePartsArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("days", array.days());
        visitor.visit_child("seconds", array.seconds());
        visitor.visit_child("subseconds", array.subseconds());
    }
}

#[cfg(test)]
mod test {
    use vortex_array::ProstMetadata;
    use vortex_array::test_harness::check_metadata;
    use vortex_dtype::PType;

    use super::*;

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_datetimeparts_metadata() {
        check_metadata(
            "datetimeparts.metadata",
            ProstMetadata(DateTimePartsMetadata {
                days_ptype: PType::I64 as i32,
                seconds_ptype: PType::I64 as i32,
                subseconds_ptype: PType::I64 as i32,
            }),
        );
    }
}
