#[cfg(test)]
mod test;

use std::sync::Arc;

use vortex_datetime_dtype::{TemporalMetadata, TimeUnit, DATE_ID, TIMESTAMP_ID, TIME_ID};
use vortex_dtype::{DType, ExtDType};
use vortex_error::{vortex_panic, VortexError};

use crate::array::ExtensionArray;
use crate::variants::ExtensionArrayTrait;
use crate::{ArrayDType, ArrayData, IntoArrayData};

/// An array wrapper for primitive values that have an associated temporal meaning.
///
/// This is a wrapper around ExtensionArrays containing numeric types, each of which corresponds to
/// either a timestamp or julian date (both referenced to UNIX epoch), OR a time since midnight.
///
/// ## Arrow compatibility
///
/// TemporalArray can be created from Arrow arrays containing the following datatypes:
/// * `Time32`
/// * `Time64`
/// * `Timestamp`
/// * `Date32`
/// * `Date64`
///
/// Anything that can be constructed and held in a `TemporalArray` can also be zero-copy converted
/// back to the relevant Arrow datatype.
#[derive(Clone, Debug)]
pub struct TemporalArray {
    /// The underlying Vortex extension array holding all the numeric values.
    ext: ExtensionArray,

    /// In-memory representation of the ExtMetadata that is held by the underlying extension array.
    ///
    /// We hold this directly to avoid needing to deserialize the metadata to access things like
    /// timezone and TimeUnit of the underlying array.
    temporal_metadata: TemporalMetadata,
}

macro_rules! assert_width {
    ($width:ty, $array:expr) => {{
        let DType::Primitive(ptype, _) = $array.dtype() else {
            panic!("array must have primitive type");
        };

        assert_eq!(
            <$width as vortex_dtype::NativePType>::PTYPE,
            *ptype,
            "invalid ptype {} for array, expected {}",
            <$width as vortex_dtype::NativePType>::PTYPE,
            *ptype
        );
    }};
}

impl TemporalArray {
    /// Create a new `TemporalArray` holding either i32 day offsets, or i64 millisecond offsets
    /// that are evenly divisible by the number of 86,400,000.
    ///
    /// This is equivalent to the data described by either of the `Date32` or `Date64` data types
    /// from Arrow.
    ///
    /// # Panics
    ///
    /// If the time unit is milliseconds, and the array is not of primitive I64 type, it panics.
    ///
    /// If the time unit is days, and the array is not of primitive I32 type, it panics.
    ///
    /// If any other time unit is provided, it panics.
    pub fn new_date(array: ArrayData, time_unit: TimeUnit) -> Self {
        match time_unit {
            TimeUnit::D => {
                assert_width!(i32, array);
            }
            TimeUnit::Ms => {
                assert_width!(i64, array);
            }
            _ => vortex_panic!("invalid TimeUnit {time_unit} for vortex.date"),
        };

        let ext_dtype = ExtDType::new(
            DATE_ID.clone(),
            Arc::new(array.dtype().clone()),
            Some(TemporalMetadata::Date(time_unit).into()),
        );

        Self {
            ext: ExtensionArray::new(Arc::new(ext_dtype), array),
            temporal_metadata: TemporalMetadata::Date(time_unit),
        }
    }

    /// Create a new `TemporalArray` holding one of the following values:
    ///
    /// * `i32` values representing seconds since midnight
    /// * `i32` values representing milliseconds since midnight
    /// * `i64` values representing microseconds since midnight
    /// * `i64` values representing nanoseconds since midnight
    ///
    /// Note, this is equivalent to the set of values represented by the Time32 or Time64 types
    /// from Arrow.
    ///
    /// # Panics
    ///
    /// If the time unit is seconds, and the array is not of primitive I32 type, it panics.
    ///
    /// If the time unit is milliseconds, and the array is not of primitive I32 type, it panics.
    ///
    /// If the time unit is microseconds, and the array is not of primitive I64 type, it panics.
    ///
    /// If the time unit is nanoseconds, and the array is not of primitive I64 type, it panics.
    pub fn new_time(array: ArrayData, time_unit: TimeUnit) -> Self {
        match time_unit {
            TimeUnit::S | TimeUnit::Ms => assert_width!(i32, array),
            TimeUnit::Us | TimeUnit::Ns => assert_width!(i64, array),
            TimeUnit::D => vortex_panic!("invalid unit D for vortex.time data"),
        }

        let temporal_metadata = TemporalMetadata::Time(time_unit);
        Self {
            ext: ExtensionArray::new(
                Arc::new(ExtDType::new(
                    TIME_ID.clone(),
                    Arc::new(array.dtype().clone()),
                    Some(temporal_metadata.clone().into()),
                )),
                array,
            ),
            temporal_metadata,
        }
    }

    /// Create a new `TemporalArray` holding Arrow spec compliant Timestamp data, with an
    /// optional timezone.
    ///
    /// # Panics
    ///
    /// If `array` does not hold Primitive i64 data, the function will panic.
    ///
    /// If the time_unit is days, the function will panic.
    pub fn new_timestamp(array: ArrayData, time_unit: TimeUnit, time_zone: Option<String>) -> Self {
        assert_width!(i64, array);

        let temporal_metadata = TemporalMetadata::Timestamp(time_unit, time_zone);

        Self {
            ext: ExtensionArray::new(
                Arc::new(ExtDType::new(
                    TIMESTAMP_ID.clone(),
                    Arc::new(array.dtype().clone()),
                    Some(temporal_metadata.clone().into()),
                )),
                array,
            ),
            temporal_metadata,
        }
    }
}

impl TemporalArray {
    /// Access the underlying temporal values in the underlying ExtensionArray storage.
    ///
    /// These values are to be interpreted based on the time unit and optional time-zone stored
    /// in the TemporalMetadata.
    pub fn temporal_values(&self) -> ArrayData {
        self.ext.storage()
    }

    /// Retrieve the temporal metadata.
    ///
    /// The metadata is used to provide semantic meaning to the temporal values Array, for example
    /// to understand the granularity of the samples and if they have an associated timezone.
    pub fn temporal_metadata(&self) -> &TemporalMetadata {
        &self.temporal_metadata
    }

    /// Retrieve the extension DType associated with the underlying array.
    pub fn ext_dtype(&self) -> Arc<ExtDType> {
        self.ext.ext_dtype().clone()
    }
}

impl From<TemporalArray> for ArrayData {
    fn from(value: TemporalArray) -> Self {
        value.ext.into_array()
    }
}

impl TryFrom<ArrayData> for TemporalArray {
    type Error = VortexError;

    /// Try to specialize a generic Vortex array as a TemporalArray.
    ///
    /// # Errors
    ///
    /// If the provided Array does not have `vortex.ext` encoding, an error will be returned.
    ///
    /// If the provided Array does not have recognized ExtMetadata corresponding to one of the known
    /// `TemporalMetadata` variants, an error is returned.
    fn try_from(value: ArrayData) -> Result<Self, Self::Error> {
        let ext = ExtensionArray::try_from(value)?;
        let temporal_metadata = TemporalMetadata::try_from(ext.ext_dtype().as_ref())?;

        Ok(Self {
            ext,
            temporal_metadata,
        })
    }
}

// Conversions to/from ExtensionArray
impl From<&TemporalArray> for ExtensionArray {
    fn from(value: &TemporalArray) -> Self {
        value.ext.clone()
    }
}

impl From<TemporalArray> for ExtensionArray {
    fn from(value: TemporalArray) -> Self {
        value.ext
    }
}

impl TryFrom<ExtensionArray> for TemporalArray {
    type Error = VortexError;

    fn try_from(ext: ExtensionArray) -> Result<Self, Self::Error> {
        let temporal_metadata = TemporalMetadata::try_from(ext.ext_dtype().as_ref())?;
        Ok(Self {
            ext,
            temporal_metadata,
        })
    }
}
