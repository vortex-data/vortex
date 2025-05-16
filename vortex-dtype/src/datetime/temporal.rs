use std::fmt::Display;
use std::sync::{Arc, LazyLock};

use jiff::civil::{Date, Time};
use jiff::{Timestamp, Zoned};
use vortex_error::{
    VortexError, VortexExpect, VortexResult, vortex_assert, vortex_bail, vortex_err, vortex_panic,
};

use crate::datetime::unit::TimeUnit;
use crate::{DType, ExtDType, ExtID, ExtMetadata, ExtensionType, PType};

/// ID for the Vortex time type.
pub static TIME_ID: LazyLock<ExtID> = LazyLock::new(|| ExtID::from("vortex.time"));
/// ID for the Vortex date type.
pub static DATE_ID: LazyLock<ExtID> = LazyLock::new(|| ExtID::from("vortex.date"));
/// ID for the Vortex timestamp type.
pub static TIMESTAMP_ID: LazyLock<ExtID> = LazyLock::new(|| ExtID::from("vortex.timestamp"));

/// Check if an `ExtID` is one of the temporal types.
pub fn is_temporal_ext_type(id: &ExtID) -> bool {
    [&DATE_ID as &ExtID, &TIME_ID, &TIMESTAMP_ID].contains(&id)
}

/// An [`ExtensionType`] for time of day.
///
/// All values of this type are referenced to midnight being the start of a day.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TimeType {
    /// The storage type for the time values.
    pub storage_type: DType,

    /// The metadata dictating the interpretation of the time scalars.
    pub metadata: TimeMetadata,
}

/// Metadata for [`TimeType`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TimeMetadata {
    /// The time unit for all values of this type.
    ///
    /// Valid values are `Second`, `Milli`, `Micro`, and `Nano`.
    pub unit: TimeUnit,
}

// Extension type metadata involved here instead...I think we want to make a new creator
// of this extension type? The metadata is really all that matters.

impl ExtensionType for TimeType {
    type Metadata = TimeMetadata;

    fn type_id() -> ExtID {
        TIME_ID.clone()
    }

    fn metadata(&self) -> &Self::Metadata {
        &self.metadata
    }

    fn serialize(&self) -> Option<ExtMetadata> {
        Some(ExtMetadata::new(vec![self.metadata.unit as u8].into()))
    }

    fn try_deserialize(serialized: &ExtMetadata) -> VortexResult<Self::Metadata>
    where
        Self: Sized,
    {
        vortex_assert!(
            serialized.as_ref().len() == 1,
            "TimeMetadata must be 1 byte: {:x?}",
            serialized.as_ref()
        );

        let unit = TimeUnit::try_from(serialized.as_ref()[0])
            .map_err(|e| vortex_err!(ComputeError: "invalid TimeUnit byte: {e}"))?;
        Ok(TimeMetadata { unit })
    }

    fn try_new(storage_type: DType, metadata: Self::Metadata) -> VortexResult<Self> {
        fn is_valid(dtype: &DType, unit: &TimeUnit) -> bool {
            match (dtype, unit) {
                (DType::Primitive(PType::I32, _), TimeUnit::Second | TimeUnit::Milli) => true,
                (DType::Primitive(PType::I64, _), TimeUnit::Micro | TimeUnit::Nano) => true,
                _ => false,
            }
        }

        vortex_assert!(
            is_valid(&storage_type, &metadata.unit),
            "Invalid storage type for TimeType: {:?} with unit {:?}",
            storage_type,
            metadata.unit
        );

        Ok(Self {
            storage_type,
            metadata,
        })
    }
}

/// Extension type for date values.
///
/// Date values are number of days since January 1, 1970 in multiple units.
#[derive(Debug, Clone)]
pub struct DateType {
    /// The storage type for the date values.
    pub storage_type: DType,

    /// Metadata used to interpret the values of this type.
    pub metadata: DateMetadata,
}

/// Metadata for interpreting values of a `DateType` array.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DateMetadata {
    /// The time unit for all values of this type.
    ///
    /// Valid values are `Day` and `Milli`.
    pub unit: TimeUnit,
}

impl ExtensionType for DateType {
    type Metadata = DateMetadata;

    fn type_id() -> ExtID {
        DATE_ID.clone()
    }

    fn metadata(&self) -> &Self::Metadata {
        &self.metadata
    }

    fn serialize(&self) -> Option<ExtMetadata> {
        Some(ExtMetadata::new(vec![self.metadata.unit as u8].into()))
    }

    fn try_deserialize(serialized: &ExtMetadata) -> VortexResult<Self::Metadata>
    where
        Self: Sized,
    {
        vortex_assert!(
            serialized.as_ref().len() == 1,
            "DateMetadata must be 1 byte: {:x?}",
            serialized.as_ref()
        );

        let unit = TimeUnit::try_from(serialized.as_ref()[0])
            .map_err(|e| vortex_err!(ComputeError: "invalid TimeUnit byte: {e}"))?;
        Ok(DateMetadata { unit })
    }

    fn try_new(storage_type: DType, metadata: Self::Metadata) -> VortexResult<Self> {
        fn is_valid(dtype: &DType, unit: &TimeUnit) -> bool {
            match (dtype, unit) {
                (DType::Primitive(PType::I32, _), TimeUnit::Day) => true,
                (DType::Primitive(PType::I64, _), TimeUnit::Milli) => true,
                _ => false,
            }
        }

        vortex_assert!(
            is_valid(&storage_type, &metadata.unit),
            "Invalid storage type for DateType: {:?} with unit {:?}",
            storage_type,
            metadata.unit
        );

        Ok(Self {
            storage_type,
            metadata,
        })
    }
}

/// Timestamp extension type.
#[derive(Debug, Clone)]
pub struct TimestampType {
    /// Storage type for timestamp values.
    pub storage_type: DType,

    /// Metadata used to interpret the values of this type.
    pub metadata: TimestampMetadata,
}

/// Metadata for interpreting values of a `TimestampType`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TimestampMetadata {
    /// Unit for timestamp values. Valid options are `Second`, `Milli`, `Micro`, and `Nano`.
    pub unit: TimeUnit,

    /// An optional time zone string.
    ///
    /// The timezone can be any of the formats supported by Arrow:
    ///
    /// * IANA tzdata zone names (e.g. "America/New_York")
    /// * An absolute zone offset in the form "+XX:XX" or "-XX:XX", e.g. "+07:30"
    pub tz: Option<String>,
}

impl ExtensionType for TimestampType {
    type Metadata = TimestampMetadata;

    fn type_id() -> ExtID {
        TIMESTAMP_ID.clone()
    }

    fn metadata(&self) -> &Self::Metadata {
        &self.metadata
    }

    fn serialize(&self) -> Option<ExtMetadata> {
        let mut serialized = Vec::new();
        serialized.push(self.metadata.unit as u8);

        if let Some(tz) = self.metadata.tz.as_ref() {
            serialized.push(tz.len().try_into().vortex_expect("tz len overflow"));
            serialized.extend_from_slice(tz.as_bytes());
        }

        Some(ExtMetadata::new(serialized.into()))
    }

    fn try_deserialize(serialized: &ExtMetadata) -> VortexResult<Self::Metadata>
    where
        Self: Sized,
    {
        vortex_assert!(
            !serialized.as_ref().is_empty(),
            "TimestampType must have at least 1 byte: {:x?}",
            serialized.as_ref()
        );
        let unit = TimeUnit::try_from(serialized.as_ref()[0])
            .map_err(|e| vortex_err!(ComputeError: "invalid TimeUnit byte: {e}"))?;

        let tz = (serialized.as_ref().len() > 1).then(|| {
            let tz_len = serialized.as_ref()[1];
            let tz_bytes = &serialized.as_ref()[2..(2 + (tz_len as usize))];
            String::from_utf8_lossy(tz_bytes).to_string()
        });

        Ok(TimestampMetadata { unit, tz })
    }

    fn try_new(storage_type: DType, metadata: Self::Metadata) -> VortexResult<Self>
    where
        Self: Sized,
    {
        fn is_valid(dtype: &DType, unit: &TimeUnit) -> bool {
            match (dtype, unit) {
                (DType::Primitive(PType::I32, _), TimeUnit::Second | TimeUnit::Milli) => true,
                (DType::Primitive(PType::I64, _), TimeUnit::Micro | TimeUnit::Nano) => true,
                _ => false,
            }
        }

        vortex_assert!(
            is_valid(&storage_type, &metadata.unit),
            "Invalid storage type for TimestampType: {:?} with unit {:?}",
            storage_type,
            metadata.unit
        );

        // TODO(aduffy): check that timezone is valid using jiff TimeZoneDatabase?

        Ok(Self {
            storage_type,
            metadata,
        })
    }
}

/// Metadata for TemporalArray.
///
/// There is one enum for each of the temporal array types we can load from Arrow.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TemporalMetadata {
    /// Metadata for a time array.
    Time(TimeUnit),
    /// Metadata for a date array.
    Date(TimeUnit),
    /// Metadata for a timestamp array.
    Timestamp(TimeUnit, Option<String>),
}

/// A Jiff representation of a temporal value.
pub enum TemporalJiff {
    /// A time value.
    Time(Time),
    /// A date value.
    Date(Date),
    /// A timestamp value.
    Timestamp(Timestamp),
    /// A zoned timestamp value.
    Zoned(Zoned),
}

impl Display for TemporalJiff {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TemporalJiff::Time(t) => write!(f, "{}", t),
            TemporalJiff::Date(d) => write!(f, "{}", d),
            TemporalJiff::Timestamp(ts) => write!(f, "{}", ts),
            TemporalJiff::Zoned(z) => write!(f, "{}", z),
        }
    }
}

impl TemporalMetadata {
    /// Retrieve the time unit associated with the array.
    ///
    /// All temporal arrays have a single intrinsic time unit for all of its numeric values.
    pub fn time_unit(&self) -> TimeUnit {
        match self {
            TemporalMetadata::Time(time_unit)
            | TemporalMetadata::Date(time_unit)
            | TemporalMetadata::Timestamp(time_unit, _) => *time_unit,
        }
    }

    /// Access the optional time-zone component of the metadata.
    pub fn time_zone(&self) -> Option<&str> {
        if let TemporalMetadata::Timestamp(_, tz) = self {
            tz.as_ref().map(|s| s.as_str())
        } else {
            None
        }
    }

    /// Convert a timestamp value to a Jiff value.
    pub fn to_jiff(&self, v: i64) -> VortexResult<TemporalJiff> {
        match self {
            TemporalMetadata::Time(TimeUnit::Day) => {
                vortex_bail!("Invalid TimeUnit TimeUnit::D for TemporalMetadata::Time")
            }
            TemporalMetadata::Time(unit) => Ok(TemporalJiff::Time(
                Time::MIN.checked_add(unit.to_jiff_span(v)?)?,
            )),
            TemporalMetadata::Date(unit) => match unit {
                TimeUnit::Day | TimeUnit::Milli => Ok(TemporalJiff::Date(
                    Date::new(1970, 1, 1)?.checked_add(unit.to_jiff_span(v)?)?,
                )),
                _ => {
                    vortex_bail!("Invalid TimeUnit {} for TemporalMetadata::Time", unit)
                }
            },
            TemporalMetadata::Timestamp(TimeUnit::Day, _) => {
                vortex_bail!("Invalid TimeUnit TimeUnit::D for TemporalMetadata::Timestamp")
            }
            TemporalMetadata::Timestamp(unit, None) => Ok(TemporalJiff::Timestamp(
                Timestamp::UNIX_EPOCH.checked_add(unit.to_jiff_span(v)?)?,
            )),
            TemporalMetadata::Timestamp(unit, Some(tz)) => Ok(TemporalJiff::Zoned(
                Timestamp::UNIX_EPOCH
                    .checked_add(unit.to_jiff_span(v)?)?
                    .in_tz(tz)?,
            )),
        }
    }
}

macro_rules! impl_temporal_metadata_try_from {
    ($typ:ty) => {
        impl TryFrom<$typ> for TemporalMetadata {
            type Error = VortexError;

            fn try_from(ext_dtype: $typ) -> Result<Self, Self::Error> {
                let metadata = ext_dtype
                    .metadata()
                    .ok_or_else(|| vortex_err!("ExtDType is missing metadata"))?;
                match ext_dtype.id().as_ref() {
                    x if x == TIME_ID.as_ref() => decode_time_metadata(metadata),
                    x if x == DATE_ID.as_ref() => decode_date_metadata(metadata),
                    x if x == TIMESTAMP_ID.as_ref() => decode_timestamp_metadata(metadata),
                    _ => {
                        vortex_bail!("ExtDType must be one of the known temporal types")
                    }
                }
            }
        }
    };
}

impl_temporal_metadata_try_from!(ExtDType);
impl_temporal_metadata_try_from!(&ExtDType);
impl_temporal_metadata_try_from!(Arc<ExtDType>);
impl_temporal_metadata_try_from!(&Arc<ExtDType>);
impl_temporal_metadata_try_from!(Box<ExtDType>);
impl_temporal_metadata_try_from!(&Box<ExtDType>);

fn decode_date_metadata(ext_meta: &ExtMetadata) -> VortexResult<TemporalMetadata> {
    let tag = ext_meta.as_ref()[0];
    let time_unit =
        TimeUnit::try_from(tag).map_err(|e| vortex_err!(ComputeError: "invalid unit tag: {e}"))?;
    Ok(TemporalMetadata::Date(time_unit))
}

fn decode_time_metadata(ext_meta: &ExtMetadata) -> VortexResult<TemporalMetadata> {
    let tag = ext_meta.as_ref()[0];
    let time_unit =
        TimeUnit::try_from(tag).map_err(|e| vortex_err!(ComputeError: "invalid unit tag: {e}"))?;
    Ok(TemporalMetadata::Time(time_unit))
}

fn decode_timestamp_metadata(ext_meta: &ExtMetadata) -> VortexResult<TemporalMetadata> {
    let tag = ext_meta.as_ref()[0];
    let time_unit =
        TimeUnit::try_from(tag).map_err(|e| vortex_err!(ComputeError: "invalid unit tag: {e}"))?;
    let tz_len_bytes = &ext_meta.as_ref()[1..3];
    let tz_len = u16::from_le_bytes(tz_len_bytes.try_into()?);
    if tz_len == 0 {
        return Ok(TemporalMetadata::Timestamp(time_unit, None));
    }

    // Attempt to load from len-prefixed bytes
    let tz_bytes = &ext_meta.as_ref()[3..(3 + (tz_len as usize))];
    let tz = String::from_utf8_lossy(tz_bytes).to_string();
    Ok(TemporalMetadata::Timestamp(time_unit, Some(tz)))
}

impl From<TemporalMetadata> for ExtMetadata {
    /// Infallibly serialize a `TemporalMetadata` as an `ExtMetadata` so it can be attached to
    /// an `ExtensionArray`.
    fn from(value: TemporalMetadata) -> Self {
        match value {
            // Time32/Time64 and Date32/Date64 only need to encode the unit in their metadata
            // The unit also unambiguously maps to the integer width of the backing array for all.
            TemporalMetadata::Time(time_unit) | TemporalMetadata::Date(time_unit) => {
                let mut meta = Vec::new();
                let unit_tag: u8 = time_unit.into();
                meta.push(unit_tag);

                ExtMetadata::from(meta.as_slice())
            }
            // Store both the time unit and zone in the metadata
            TemporalMetadata::Timestamp(time_unit, time_zone) => {
                let mut meta = Vec::new();
                let unit_tag: u8 = time_unit.into();

                meta.push(unit_tag);

                // Encode time_zone as u16 length followed by utf8 bytes.
                match time_zone {
                    None => meta.extend_from_slice(0u16.to_le_bytes().as_slice()),
                    Some(tz) => {
                        let tz_bytes = tz.as_bytes();
                        let tz_len = u16::try_from(tz_bytes.len())
                            .unwrap_or_else(|err| vortex_panic!("tz did not fit in u16: {}", err));
                        meta.extend_from_slice(tz_len.to_le_bytes().as_slice());
                        meta.extend_from_slice(tz_bytes);
                    }
                }
                ExtMetadata::from(meta.as_slice())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{ExtDType, ExtMetadata, PType};

    #[test]
    fn test_roundtrip_metadata() {
        let meta: ExtMetadata =
            TemporalMetadata::Timestamp(TimeUnit::Milli, Some("UTC".to_string())).into();

        assert_eq!(
            meta.as_ref(),
            vec![
                2u8, // Tag for TimeUnit::Ms
                0x3u8, 0x0u8, // u16 length
                b'U', b'T', b'C',
            ]
            .as_slice()
        );

        let temporal_metadata = TemporalMetadata::try_from(&ExtDType::new(
            TIMESTAMP_ID.clone(),
            Arc::new(PType::I64.into()),
            Some(meta),
        ))
        .unwrap();

        assert_eq!(
            temporal_metadata,
            TemporalMetadata::Timestamp(TimeUnit::Milli, Some("UTC".to_string()))
        );
    }
}
