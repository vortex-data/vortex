use vortex_array::{Array, ArrayChildVisitor, ArrayVisitorImpl, RkyvMetadata};
use vortex_dtype::PType;
use vortex_error::VortexExpect;

use crate::DateTimePartsArray;

#[derive(Clone, Debug, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[repr(C)]
pub struct DateTimePartsMetadata {
    // Validity lives in the days array
    // TODO(ngates): we should actually model this with a Tuple array when we have one.
    pub(crate) days_ptype: PType,
    pub(crate) seconds_ptype: PType,
    pub(crate) subseconds_ptype: PType,
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
