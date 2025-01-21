use std::fmt::{Debug, Display};

use serde::{Deserialize, Serialize};
use vortex_array::array::StructArray;
use vortex_array::compute::try_cast;
use vortex_array::encoding::ids;
use vortex_array::stats::StatsSet;
use vortex_array::validate::ValidateVTable;
use vortex_array::validity::{ArrayValidity, LogicalValidity, Validity, ValidityVTable};
use vortex_array::variants::{ExtensionArrayTrait, VariantsVTable};
use vortex_array::visitor::{ArrayVisitor, VisitorVTable};
use vortex_array::{impl_encoding, ArrayDType, ArrayData, ArrayLen, IntoArrayData};
use vortex_dtype::{DType, PType};
use vortex_error::{vortex_bail, VortexExpect as _, VortexResult, VortexUnwrap};

impl_encoding!("vortex.datetimeparts", ids::DATE_TIME_PARTS, DateTimeParts);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DateTimePartsMetadata {
    // Validity lives in the days array
    // TODO(ngates): we should actually model this with a Tuple array when we have one.
    days_ptype: PType,
    seconds_ptype: PType,
    subseconds_ptype: PType,
}

impl Display for DateTimePartsMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

impl DateTimePartsArray {
    pub fn try_new(
        dtype: DType,
        days: ArrayData,
        seconds: ArrayData,
        subsecond: ArrayData,
    ) -> VortexResult<Self> {
        if !days.dtype().is_int() || (dtype.is_nullable() != days.dtype().is_nullable()) {
            vortex_bail!(
                "Expected integer with nullability {}, got {}",
                dtype.is_nullable(),
                days.dtype()
            );
        }
        if !seconds.dtype().is_int() || seconds.dtype().is_nullable() {
            vortex_bail!(MismatchedTypes: "non-nullable integer", seconds.dtype());
        }
        if !subsecond.dtype().is_int() || subsecond.dtype().is_nullable() {
            vortex_bail!(MismatchedTypes: "non-nullable integer", subsecond.dtype());
        }

        let length = days.len();
        if length != seconds.len() || length != subsecond.len() {
            vortex_bail!(
                "Mismatched lengths {} {} {}",
                days.len(),
                seconds.len(),
                subsecond.len()
            );
        }

        let metadata = DateTimePartsMetadata {
            days_ptype: days.dtype().try_into()?,
            seconds_ptype: seconds.dtype().try_into()?,
            subseconds_ptype: subsecond.dtype().try_into()?,
        };

        Self::try_from_parts(
            dtype,
            length,
            metadata,
            None,
            Some([days, seconds, subsecond].into()),
            StatsSet::default(),
        )
    }

    pub fn days(&self) -> ArrayData {
        self.as_ref()
            .child(
                0,
                &DType::Primitive(self.metadata().days_ptype, self.dtype().nullability()),
                self.len(),
            )
            .vortex_expect("DatetimePartsArray missing days array")
    }

    pub fn seconds(&self) -> ArrayData {
        self.as_ref()
            .child(1, &self.metadata().seconds_ptype.into(), self.len())
            .vortex_expect("DatetimePartsArray missing seconds array")
    }

    pub fn subsecond(&self) -> ArrayData {
        self.as_ref()
            .child(2, &self.metadata().subseconds_ptype.into(), self.len())
            .vortex_expect("DatetimePartsArray missing subsecond array")
    }

    pub fn validity(&self) -> Validity {
        self.days()
            .logical_validity()
            .into_validity(self.dtype().nullability())
    }
}

impl ValidateVTable<DateTimePartsArray> for DateTimePartsEncoding {}

impl VariantsVTable<DateTimePartsArray> for DateTimePartsEncoding {
    fn as_extension_array<'a>(
        &self,
        array: &'a DateTimePartsArray,
    ) -> Option<&'a dyn ExtensionArrayTrait> {
        Some(array)
    }
}

impl ExtensionArrayTrait for DateTimePartsArray {
    fn storage_data(&self) -> ArrayData {
        // FIXME(ngates): this needs to be a tuple array so we can implement Compare
        // we don't want to write validity twice, so we pull it up to the top
        let days = try_cast(self.days(), &self.days().dtype().as_nonnullable()).vortex_unwrap();
        StructArray::try_new(
            vec!["days".into(), "seconds".into(), "subseconds".into()].into(),
            [days, self.seconds(), self.subsecond()].into(),
            self.len(),
            self.validity(),
        )
        .vortex_expect("Failed to create struct array")
        .into_array()
    }
}

impl ValidityVTable<DateTimePartsArray> for DateTimePartsEncoding {
    fn is_valid(&self, array: &DateTimePartsArray, index: usize) -> bool {
        array.validity().is_valid(index)
    }

    fn logical_validity(&self, array: &DateTimePartsArray) -> LogicalValidity {
        array.validity().to_logical(array.len())
    }
}

impl VisitorVTable<DateTimePartsArray> for DateTimePartsEncoding {
    fn accept(
        &self,
        array: &DateTimePartsArray,
        visitor: &mut dyn ArrayVisitor,
    ) -> VortexResult<()> {
        visitor.visit_child("days", &array.days())?;
        visitor.visit_child("seconds", &array.seconds())?;
        visitor.visit_child("subsecond", &array.subsecond())
    }
}

#[cfg(test)]
mod test {
    use vortex_array::test_harness::check_metadata;
    use vortex_dtype::PType;

    use crate::DateTimePartsMetadata;

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_datetimeparts_metadata() {
        check_metadata(
            "datetimeparts.metadata",
            DateTimePartsMetadata {
                days_ptype: PType::I64,
                seconds_ptype: PType::I64,
                subseconds_ptype: PType::I64,
            },
        );
    }
}
