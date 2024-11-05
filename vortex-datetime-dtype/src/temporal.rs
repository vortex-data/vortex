use std::fmt::Display;
use std::sync::{Arc, LazyLock};

use jiff::civil::{Date, Time};
use jiff::{Timestamp, Zoned};
use vortex_dtype::ExtID;

use crate::unit::TimeUnit;

pub static TIME_ID: LazyLock<ExtID> = LazyLock::new(|| ExtID::from("vortex.time"));
pub static DATE_ID: LazyLock<ExtID> = LazyLock::new(|| ExtID::from("vortex.date"));
pub static TIMESTAMP_ID: LazyLock<ExtID> = LazyLock::new(|| ExtID::from("vortex.timestamp"));

pub fn is_temporal_ext_type(id: &ExtID) -> bool {
    [&DATE_ID as &ExtID, &TIME_ID, &TIMESTAMP_ID].contains(&id)
}

/// Metadata for TemporalArray.
///
/// There is one enum for each of the temporal array types we can load from Arrow.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TemporalMetadata {
    Time(TimeUnit),
    Date(TimeUnit),
    Timestamp(TimeUnit, Option<String>),
}

pub enum TemporalJiff {
    Time(Time),
    Date(Date),
    Timestamp(Timestamp),
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

    pub fn to_jiff(&self, v: i64) -> VortexResult<TemporalJiff> {
        match self {
            TemporalMetadata::Time(TimeUnit::D) => {
                vortex_bail!("Invalid TimeUnit TimeUnit::D for TemporalMetadata::Time")
            }
            TemporalMetadata::Time(unit) => Ok(TemporalJiff::Time(
                Time::MIN.checked_add(unit.to_jiff_span(v)?)?,
            )),
            TemporalMetadata::Date(unit) => match unit {
                TimeUnit::D | TimeUnit::Ms => Ok(TemporalJiff::Date(
                    Date::new(1970, 1, 1)?.checked_add(unit.to_jiff_span(v)?)?,
                )),
                _ => {
                    vortex_bail!("Invalid TimeUnit {} for TemporalMetadata::Time", unit)
                }
            },
            TemporalMetadata::Timestamp(TimeUnit::D, _) => {
                vortex_bail!("Invalid TimeUnit TimeUnit::D for TemporalMetadata::Timestamp")
            }
            TemporalMetadata::Timestamp(unit, None) => Ok(TemporalJiff::Timestamp(
                Timestamp::UNIX_EPOCH.checked_add(unit.to_jiff_span(v)?)?,
            )),
            TemporalMetadata::Timestamp(unit, Some(tz)) => Ok(TemporalJiff::Zoned(
                Timestamp::UNIX_EPOCH
                    .checked_add(unit.to_jiff_span(v)?)?
                    .intz(tz)?,
            )),
        }
    }
}

use vortex_dtype::{ExtDType, ExtMetadata};
use vortex_error::{vortex_bail, vortex_err, vortex_panic, VortexError, VortexResult};

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
impl_temporal_metadata_try_from!(Box<ExtDType>);

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

    use vortex_dtype::{ExtDType, ExtMetadata, PType};

    use crate::{TemporalMetadata, TimeUnit, TIMESTAMP_ID};

    #[test]
    fn test_roundtrip_metadata() {
        let meta: ExtMetadata =
            TemporalMetadata::Timestamp(TimeUnit::Ms, Some("UTC".to_string())).into();

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
            TemporalMetadata::Timestamp(TimeUnit::Ms, Some("UTC".to_string()))
        );
    }
}
