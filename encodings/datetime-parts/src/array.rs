use std::fmt::Debug;

use serde::{Deserialize, Serialize};
use vortex_array::arrays::StructArray;
use vortex_array::compute::try_cast;
use vortex_array::stats::StatsSet;
use vortex_array::validity::Validity;
use vortex_array::variants::ExtensionArrayTrait;
use vortex_array::visitor::ArrayVisitor;
use vortex_array::vtable::{ValidateVTable, ValidityVTable, VariantsVTable, VisitorVTable};
use vortex_array::{encoding_ids, impl_encoding, Array, IntoArray, SerdeMetadata};
use vortex_dtype::{DType, PType};
use vortex_error::{vortex_bail, VortexExpect as _, VortexResult, VortexUnwrap};
use vortex_mask::Mask;

impl_encoding!(
    "vortex.datetimeparts",
    encoding_ids::DATE_TIME_PARTS,
    DateTimeParts,
    SerdeMetadata<DateTimePartsMetadata>
);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DateTimePartsMetadata {
    // Validity lives in the days array
    // TODO(ngates): we should actually model this with a Tuple array when we have one.
    days_ptype: PType,
    seconds_ptype: PType,
    subseconds_ptype: PType,
}

impl DateTimePartsArray {
    pub fn try_new(
        dtype: DType,
        days: Array,
        seconds: Array,
        subseconds: Array,
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
        if !subseconds.dtype().is_int() || subseconds.dtype().is_nullable() {
            vortex_bail!(MismatchedTypes: "non-nullable integer", subseconds.dtype());
        }

        let length = days.len();
        if length != seconds.len() || length != subseconds.len() {
            vortex_bail!(
                "Mismatched lengths {} {} {}",
                days.len(),
                seconds.len(),
                subseconds.len()
            );
        }

        let metadata = DateTimePartsMetadata {
            days_ptype: days.dtype().try_into()?,
            seconds_ptype: seconds.dtype().try_into()?,
            subseconds_ptype: subseconds.dtype().try_into()?,
        };

        Self::try_from_parts(
            dtype,
            length,
            SerdeMetadata(metadata),
            vec![].into(),
            [days, seconds, subseconds].into(),
            StatsSet::default(),
        )
    }

    pub fn days(&self) -> Array {
        self.as_ref()
            .child(
                0,
                &DType::Primitive(self.metadata().days_ptype, self.dtype().nullability()),
                self.len(),
            )
            .vortex_expect("DatetimePartsArray missing days array")
    }

    pub fn seconds(&self) -> Array {
        self.as_ref()
            .child(1, &self.metadata().seconds_ptype.into(), self.len())
            .vortex_expect("DatetimePartsArray missing seconds array")
    }

    pub fn subseconds(&self) -> Array {
        self.as_ref()
            .child(2, &self.metadata().subseconds_ptype.into(), self.len())
            .vortex_expect("DatetimePartsArray missing subseconds array")
    }

    pub fn validity(&self) -> VortexResult<Validity> {
        // FIXME(ngates): this function is weird... can we just use logical validity?
        Ok(Validity::from_mask(
            self.days().validity_mask()?,
            self.dtype().nullability(),
        ))
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
    fn storage_data(&self) -> Array {
        // FIXME(ngates): this needs to be a tuple array so we can implement Compare
        // we don't want to write validity twice, so we pull it up to the top
        let days = try_cast(self.days(), &self.days().dtype().as_nonnullable()).vortex_unwrap();
        StructArray::try_new(
            vec!["days".into(), "seconds".into(), "subseconds".into()].into(),
            [days, self.seconds(), self.subseconds()].into(),
            self.len(),
            self.validity()
                .vortex_expect("Failed to create struct array validity"),
        )
        .vortex_expect("Failed to create struct array")
        .into_array()
    }
}

impl ValidityVTable<DateTimePartsArray> for DateTimePartsEncoding {
    fn is_valid(&self, array: &DateTimePartsArray, index: usize) -> VortexResult<bool> {
        array.days().is_valid(index)
    }

    fn all_valid(&self, array: &DateTimePartsArray) -> VortexResult<bool> {
        array.days().all_valid()
    }

    fn all_invalid(&self, array: &DateTimePartsArray) -> VortexResult<bool> {
        array.days().all_invalid()
    }

    fn validity_mask(&self, array: &DateTimePartsArray) -> VortexResult<Mask> {
        array.days().validity_mask()
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
        visitor.visit_child("subseconds", &array.subseconds())
    }
}

#[cfg(test)]
mod test {
    use vortex_array::test_harness::check_metadata;
    use vortex_array::SerdeMetadata;
    use vortex_dtype::PType;

    use crate::DateTimePartsMetadata;

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_datetimeparts_metadata() {
        check_metadata(
            "datetimeparts.metadata",
            SerdeMetadata(DateTimePartsMetadata {
                days_ptype: PType::I64,
                seconds_ptype: PType::I64,
                subseconds_ptype: PType::I64,
            }),
        );
    }
}
