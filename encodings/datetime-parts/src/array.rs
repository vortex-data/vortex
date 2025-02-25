use std::fmt::Debug;
use std::sync::{Arc, RwLock};

use vortex_array::arrays::StructArray;
use vortex_array::compute::try_cast;
use vortex_array::stats::StatsSet;
use vortex_array::validity::Validity;
use vortex_array::variants::ExtensionArrayTrait;
use vortex_array::vtable::VTableRef;
use vortex_array::{
    encoding_ids, Array, ArrayImpl, ArrayRef, ArrayStatisticsImpl, ArrayValidityImpl,
    ArrayVariantsImpl, Encoding, EncodingId, RkyvMetadata,
};
use vortex_dtype::DType;
use vortex_error::{vortex_bail, VortexExpect as _, VortexResult, VortexUnwrap};
use vortex_mask::Mask;

use crate::serde::DateTimePartsMetadata;

#[derive(Clone, Debug)]
pub struct DateTimePartsArray {
    dtype: DType,
    days: ArrayRef,
    seconds: ArrayRef,
    subseconds: ArrayRef,
    stats_set: Arc<RwLock<StatsSet>>,
}

pub struct DateTimePartsEncoding;
impl Encoding for DateTimePartsEncoding {
    const ID: EncodingId = EncodingId::new("vortex.datetimeparts", encoding_ids::DATE_TIME_PARTS);
    type Array = DateTimePartsArray;
    type Metadata = RkyvMetadata<DateTimePartsMetadata>;
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
        VTableRef::from_static(&DateTimePartsEncoding)
    }
}

impl ArrayVariantsImpl for DateTimePartsArray {
    fn _as_extension_typed(&self) -> Option<&dyn ExtensionArrayTrait> {
        Some(self)
    }
}

impl ExtensionArrayTrait for DateTimePartsArray {
    fn storage_data(&self) -> ArrayRef {
        // FIXME(ngates): this needs to be a tuple array so we can implement Compare
        // we don't want to write validity twice, so we pull it up to the top
        let days = try_cast(self.days(), &self.days().dtype().as_nonnullable()).vortex_unwrap();
        StructArray::try_new(
            vec!["days".into(), "seconds".into(), "subseconds".into()].into(),
            [days, self.seconds().clone(), self.subseconds().clone()].into(),
            self.len(),
            Validity::copy_from_array(self).vortex_expect("Failed to copy validity"),
        )
        .vortex_expect("Failed to create struct array")
        .into_array()
    }
}

impl ArrayStatisticsImpl for DateTimePartsArray {
    fn _stats_set(&self) -> &RwLock<StatsSet> {
        &self.stats_set
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
