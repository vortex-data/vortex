use std::fmt::Debug;

use vortex_array::stats::{ArrayStats, StatsSetRef};
use vortex_array::vtable::{
    ArrayVTable, NotSupported, VTable, ValidityChild, ValidityVTableFromChild,
};
use vortex_array::{Array, ArrayRef, EncodingId, EncodingRef, vtable};
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};

vtable!(DateTimeParts);

impl VTable for DateTimePartsVTable {
    type Array = DateTimePartsArray;
    type Encoding = DateTimePartsEncoding;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = Self;
    type SerdeVTable = Self;

    fn id(_encoding: &Self::Encoding) -> EncodingId {
        EncodingId::new_ref("vortex.datetimeparts")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        EncodingRef::new_ref(DateTimePartsEncoding.as_ref())
    }
}

#[derive(Clone, Debug)]
pub struct DateTimePartsArray {
    dtype: DType,
    days: ArrayRef,
    seconds: ArrayRef,
    subseconds: ArrayRef,
    stats_set: ArrayStats,
}

#[derive(Clone, Debug)]
pub struct DateTimePartsEncoding;

impl DateTimePartsArray {
    pub fn try_new(
        dtype: DType,
        days: ArrayRef,
        seconds: ArrayRef,
        subseconds: ArrayRef,
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

        Ok(Self {
            dtype,
            days,
            seconds,
            subseconds,
            stats_set: Default::default(),
        })
    }

    pub fn days(&self) -> &ArrayRef {
        &self.days
    }

    pub fn seconds(&self) -> &ArrayRef {
        &self.seconds
    }

    pub fn subseconds(&self) -> &ArrayRef {
        &self.subseconds
    }
}

impl ArrayVTable<DateTimePartsVTable> for DateTimePartsVTable {
    fn len(array: &DateTimePartsArray) -> usize {
        array.days.len()
    }

    fn dtype(array: &DateTimePartsArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &DateTimePartsArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }
}

impl ValidityChild<DateTimePartsVTable> for DateTimePartsVTable {
    fn validity_child(array: &DateTimePartsArray) -> &dyn Array {
        array.days()
    }
}
