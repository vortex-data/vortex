use vortex_array::serde::ArrayParts;
use vortex_array::vtable::SerdeVTable;
use vortex_array::{
    Array, ArrayChildVisitor, ArrayContext, ArrayRef, ArrayVisitorImpl, DeserializeMetadata,
    RkyvMetadata,
};
use vortex_dtype::{DType, Nullability, PType};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};

use crate::{DateTimePartsArray, DateTimePartsEncoding};

#[derive(Clone, Debug, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[repr(C)]
pub struct DateTimePartsMetadata {
    // Validity lives in the days array
    // TODO(ngates): we should actually model this with a Tuple array when we have one.
    days_ptype: PType,
    seconds_ptype: PType,
    subseconds_ptype: PType,
}

impl ArrayVisitorImpl<RkyvMetadata<DateTimePartsMetadata>> for DateTimePartsArray {
    fn _children(&self, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("days", self.days());
        visitor.visit_child("seconds", self.seconds());
        visitor.visit_child("subseconds", self.subseconds());
    }

    fn _metadata(&self) -> RkyvMetadata<DateTimePartsMetadata> {
        RkyvMetadata(DateTimePartsMetadata {
            days_ptype: PType::try_from(self.days().dtype()).vortex_expect("Must be a valid PType"),
            seconds_ptype: PType::try_from(self.seconds().dtype())
                .vortex_expect("Must be a valid PType"),
            subseconds_ptype: PType::try_from(self.subseconds().dtype())
                .vortex_expect("Must be a valid PType"),
        })
    }
}

impl SerdeVTable<&DateTimePartsArray> for DateTimePartsEncoding {
    fn decode(
        &self,
        parts: &ArrayParts,
        ctx: &ArrayContext,
        dtype: DType,
        len: usize,
    ) -> VortexResult<ArrayRef> {
        if parts.nchildren() != 3 {
            vortex_bail!(
                "Expected 3 children for datetime-parts encoding, found {}",
                parts.nchildren()
            )
        }

        let metadata = RkyvMetadata::<DateTimePartsMetadata>::deserialize(parts.metadata())?;
        let days = parts.child(0).decode(
            ctx,
            DType::Primitive(metadata.days_ptype, dtype.nullability()),
            len,
        )?;
        let seconds = parts.child(1).decode(
            ctx,
            DType::Primitive(metadata.seconds_ptype, Nullability::NonNullable),
            len,
        )?;
        let subseconds = parts.child(2).decode(
            ctx,
            DType::Primitive(metadata.subseconds_ptype, Nullability::NonNullable),
            len,
        )?;

        Ok(DateTimePartsArray::try_new(dtype, days, seconds, subseconds)?.into_array())
    }
}

#[cfg(test)]
mod test {
    use vortex_array::test_harness::check_metadata;
    use vortex_dtype::PType;

    use super::*;

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_datetimeparts_metadata() {
        check_metadata(
            "datetimeparts.metadata",
            RkyvMetadata(DateTimePartsMetadata {
                days_ptype: PType::I64,
                seconds_ptype: PType::I64,
                subseconds_ptype: PType::I64,
            }),
        );
    }
}
