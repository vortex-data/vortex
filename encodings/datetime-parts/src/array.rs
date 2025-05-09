use std::fmt::Debug;

use vortex_array::stats::{ArrayStats, StatsSetRef};
use vortex_array::vtable::VTableRef;
use vortex_array::{
    Array, ArrayImpl, ArrayRef, ArrayStatisticsImpl, ArrayValidityImpl, Encoding, ProstMetadata,
};
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};
use vortex_mask::Mask;

use crate::serde::DateTimePartsMetadata;

#[derive(Clone, Debug)]
pub struct DateTimePartsArray {
    dtype: DType,
    days: ArrayRef,
    seconds: ArrayRef,
    subseconds: ArrayRef,
    stats_set: ArrayStats,
}

#[derive(Debug)]
pub struct DateTimePartsEncoding;
impl Encoding for DateTimePartsEncoding {
    type Array = DateTimePartsArray;
    type Metadata = ProstMetadata<DateTimePartsMetadata>;
}

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

impl ArrayImpl for DateTimePartsArray {
    type Encoding = DateTimePartsEncoding;

    fn _len(&self) -> usize {
        self.days.len()
    }

    fn _dtype(&self) -> &DType {
        &self.dtype
    }

    fn _vtable(&self) -> VTableRef {
        VTableRef::new_ref(&DateTimePartsEncoding)
    }

    fn _with_children(&self, children: &[ArrayRef]) -> VortexResult<Self> {
        Self::try_new(
            self.dtype.clone(),
            children[0].clone(),
            children[1].clone(),
            children[2].clone(),
        )
    }
}

impl ArrayStatisticsImpl for DateTimePartsArray {
    fn _stats_ref(&self) -> StatsSetRef<'_> {
        self.stats_set.to_ref(self)
    }
}

impl ArrayValidityImpl for DateTimePartsArray {
    fn _is_valid(&self, index: usize) -> VortexResult<bool> {
        self.days().is_valid(index)
    }

    fn _all_valid(&self) -> VortexResult<bool> {
        self.days().all_valid()
    }

    fn _all_invalid(&self) -> VortexResult<bool> {
        self.days().all_invalid()
    }

    fn _validity_mask(&self) -> VortexResult<Mask> {
        self.days().validity_mask()
    }
}
