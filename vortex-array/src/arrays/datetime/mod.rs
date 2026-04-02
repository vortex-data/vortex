// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(test)]
mod test;

use std::sync::Arc;

use vortex_error::VortexError;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::Extension;
use crate::arrays::ExtensionArray;
use crate::dtype::DType;
use crate::dtype::extension::ExtDTypeRef;
use crate::extension::datetime::AnyTemporal;
use crate::extension::datetime::Date;
use crate::extension::datetime::TemporalMetadata;
use crate::extension::datetime::Time;
use crate::extension::datetime::TimeUnit;
use crate::extension::datetime::Timestamp;

/// An array wrapper for primitive values that have an associated temporal meaning.
///
/// This is a wrapper around ExtensionArrays containing numeric types, each of which corresponds to
/// either a timestamp or julian date (both referenced to UNIX epoch), OR a time since midnight.
///
/// ## Arrow compatibility
///
/// TemporalData can be created from Arrow arrays containing the following datatypes:
/// * `Time32`
/// * `Time64`
/// * `Timestamp`
/// * `Date32`
/// * `Date64`
///
/// Anything that can be constructed and held in a `TemporalData` can also be zero-copy converted
/// back to the relevant Arrow datatype.
#[derive(Clone, Debug)]
pub struct TemporalData {
    /// The underlying Vortex extension array holding all the numeric values.
    ext: ExtensionArray,
}

/// Type alias for backward compatibility.
pub type TemporalArray = TemporalData;

impl TemporalData {
    /// Create a new `TemporalData` holding either i32 day offsets, or i64 millisecond offsets
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
    pub fn new_date(array: ArrayRef, time_unit: TimeUnit) -> Self {
        Self {
            ext: ExtensionArray::new(
                Date::new(time_unit, array.dtype().nullability()).erased(),
                array,
            ),
        }
    }

    /// Create a new `TemporalData` holding one of the following values:
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
    pub fn new_time(array: ArrayRef, time_unit: TimeUnit) -> Self {
        Self {
            ext: ExtensionArray::new(
                Time::new(time_unit, array.dtype().nullability()).erased(),
                array,
            ),
        }
    }

    /// Create a new `TemporalData` holding Arrow spec compliant Timestamp data, with an
    /// optional timezone.
    ///
    /// # Panics
    ///
    /// If `array` does not hold Primitive i64 data, the function will panic.
    ///
    /// If the time_unit is days, the function will panic.
    pub fn new_timestamp(
        array: ArrayRef,
        time_unit: TimeUnit,
        time_zone: Option<Arc<str>>,
    ) -> Self {
        Self {
            ext: ExtensionArray::new(
                Timestamp::new_with_tz(time_unit, time_zone, array.dtype().nullability()).erased(),
                array,
            ),
        }
    }
}

impl TemporalData {
    /// Access the underlying temporal values in the underlying ExtensionArray storage.
    ///
    /// These values are to be interpreted based on the time unit and optional time-zone stored
    /// in the TemporalMetadata.
    pub fn temporal_values(&self) -> &ArrayRef {
        self.ext.storage_array()
    }

    /// Retrieve the temporal metadata.
    ///
    /// The metadata is used to provide semantic meaning to the temporal values Array, for example
    /// to understand the granularity of the samples and if they have an associated timezone.
    pub fn temporal_metadata(&self) -> TemporalMetadata<'_> {
        self.ext.dtype().as_extension().metadata::<AnyTemporal>()
    }

    /// Retrieve the extension DType associated with the underlying array.
    pub fn ext_dtype(&self) -> ExtDTypeRef {
        self.ext.ext_dtype().clone()
    }

    /// Retrieve the DType of the array. This will be a `DType::Extension` variant.
    pub fn dtype(&self) -> &DType {
        self.ext.dtype()
    }
}

impl From<TemporalData> for ArrayRef {
    fn from(value: TemporalData) -> Self {
        value.ext.into_array()
    }
}

impl IntoArray for TemporalData {
    fn into_array(self) -> ArrayRef {
        self.into()
    }
}

impl TryFrom<ArrayRef> for TemporalData {
    type Error = VortexError;

    /// Try to specialize a generic Vortex array as a TemporalData.
    ///
    /// # Errors
    ///
    /// If the provided Array does not have `vortex.ext` encoding, an error will be returned.
    ///
    /// If the provided Array does not have recognized ExtMetadata corresponding to one of the known
    /// `TemporalMetadata` variants, an error is returned.
    fn try_from(value: ArrayRef) -> Result<Self, Self::Error> {
        let ext = value
            .as_opt::<Extension>()
            .ok_or_else(|| vortex_err!("array must be an ExtensionArray"))?;
        if !ext.ext_dtype().is::<AnyTemporal>() {
            vortex_bail!(
                "array extension dtype {} is not a temporal type",
                ext.ext_dtype()
            );
        }
        Ok(Self {
            ext: ext.into_owned(),
        })
    }
}

// Conversions to/from ExtensionArray
impl From<&TemporalData> for ExtensionArray {
    fn from(value: &TemporalData) -> Self {
        value.ext.clone()
    }
}

impl From<TemporalData> for ExtensionArray {
    fn from(value: TemporalData) -> Self {
        value.ext
    }
}

impl TryFrom<ExtensionArray> for TemporalData {
    type Error = VortexError;

    fn try_from(ext: ExtensionArray) -> Result<Self, Self::Error> {
        if !ext.ext_dtype().is::<AnyTemporal>() {
            vortex_bail!(
                "array extension dtype {} is not a temporal type",
                ext.ext_dtype()
            );
        }
        Ok(Self { ext })
    }
}
