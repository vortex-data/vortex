// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ArrayPlugin`] implementation for [`DateTimeParts`] across v1 and v2 wire formats.
//!
//! - `vortex.datetimeparts` (v1, frozen in `core2025.05.0`): `Timestamp` dtypes with
//!   `subseconds_divisor == 1` and the 3-field [`DateTimePartsMetadata`].
//! - `vortex.datetimeparts.v2` (v2, in `unstable`): `Time` dtypes or scaled subseconds
//!   (`subseconds_divisor > 1`), carrying [`DateTimePartsV2Metadata`].

use prost::Message;
use vortex_array::Array;
use vortex_array::ArrayDeserialization;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayPlugin;
use vortex_array::ArrayRef;
use vortex_array::ArraySerialization;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::smallvec::smallvec;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use super::DateTimeParts;
use super::DateTimePartsArraySlotsExt;
use super::DateTimePartsData;
use super::DateTimePartsMetadata;
use crate::timestamp::temporal_kind;
use crate::timestamp::validate_subseconds_divisor;

/// The frozen DateTimeParts serialized ID (`vortex.datetimeparts`), supporting `Timestamp` with
/// unscaled subseconds (`subseconds_divisor == 1`).
pub fn datetime_parts_v1_id() -> ArrayId {
    static ID: CachedId = CachedId::new("vortex.datetimeparts");
    *ID
}

/// The DateTimeParts v2 serialized ID (`vortex.datetimeparts.v2`), supporting `Time` dtypes and
/// coarse-grained `subseconds_divisor > 1`.
pub fn datetime_parts_v2_id() -> ArrayId {
    static ID: CachedId = CachedId::new("vortex.datetimeparts.v2");
    *ID
}

/// Wire metadata for `vortex.datetimeparts.v2`.
#[derive(Clone, prost::Message)]
pub struct DateTimePartsV2Metadata {
    #[prost(enumeration = "PType", tag = "1")]
    pub(crate) days_ptype: i32,
    #[prost(enumeration = "PType", tag = "2")]
    pub(crate) seconds_ptype: i32,
    #[prost(enumeration = "PType", tag = "3")]
    pub(crate) subseconds_ptype: i32,
    #[prost(uint32, tag = "4")]
    pub(crate) subseconds_divisor: u32,
}

impl DateTimePartsV2Metadata {
    pub fn get_days_ptype(&self) -> VortexResult<PType> {
        PType::try_from(self.days_ptype)
            .map_err(|_| vortex_err!("Invalid PType {}", self.days_ptype))
    }

    pub fn get_seconds_ptype(&self) -> VortexResult<PType> {
        PType::try_from(self.seconds_ptype)
            .map_err(|_| vortex_err!("Invalid PType {}", self.seconds_ptype))
    }

    pub fn get_subseconds_ptype(&self) -> VortexResult<PType> {
        PType::try_from(self.subseconds_ptype)
            .map_err(|_| vortex_err!("Invalid PType {}", self.subseconds_ptype))
    }
}

/// Serde plugin for [`DateTimeParts`] supporting both v1 and v2 wire formats.
///
/// Writes `vortex.datetimeparts` (v1) whenever the dtype is `Timestamp` and
/// `subseconds_divisor == 1`, preserving byte-identical output for all existing `core` files.
/// Writes `vortex.datetimeparts.v2` whenever the dtype is `Time` or `subseconds_divisor > 1`.
#[derive(Clone, Debug)]
pub struct DateTimePartsPlugin;

impl ArrayPlugin for DateTimePartsPlugin {
    fn id(&self) -> ArrayId {
        VTable::id(&DateTimeParts)
    }

    fn read_ids(&self) -> Vec<ArrayId> {
        vec![datetime_parts_v1_id(), datetime_parts_v2_id()]
    }

    fn write_ids(&self) -> Vec<ArrayId> {
        vec![datetime_parts_v1_id(), datetime_parts_v2_id()]
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        _session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        vortex_ensure_eq!(
            array.encoding_id(),
            ArrayPlugin::id(self),
            "DateTimeParts plugin cannot serialize an array of another encoding",
        );
        let array = array.as_::<DateTimeParts>();
        let kind = temporal_kind(array.dtype())?;
        let days_ptype = PType::try_from(array.days().dtype())? as i32;
        let seconds_ptype = PType::try_from(array.seconds().dtype())? as i32;
        let subseconds_ptype = PType::try_from(array.subseconds().dtype())? as i32;
        let subseconds_divisor = array.subseconds_divisor();

        if kind.is_timestamp() && subseconds_divisor == 1 {
            let metadata = DateTimePartsMetadata {
                days_ptype,
                seconds_ptype,
                subseconds_ptype,
            }
            .encode_to_vec();
            Ok(Some(ArraySerialization::from_array(
                datetime_parts_v1_id(),
                array.array(),
                metadata,
            )))
        } else {
            let metadata = DateTimePartsV2Metadata {
                days_ptype,
                seconds_ptype,
                subseconds_ptype,
                subseconds_divisor,
            }
            .encode_to_vec();
            Ok(Some(ArraySerialization::from_array(
                datetime_parts_v2_id(),
                array.array(),
                metadata,
            )))
        }
    }

    fn deserialize(
        &self,
        parts: ArrayDeserialization<'_>,
        _session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        let ArrayDeserialization {
            serialized_id,
            dtype,
            len,
            metadata,
            children,
            ..
        } = parts;

        let kind = temporal_kind(dtype)?;
        let (days_ptype, seconds_ptype, subseconds_ptype, subseconds_divisor) =
            if serialized_id == datetime_parts_v1_id() {
                vortex_ensure!(
                    kind.is_timestamp(),
                    "vortex.datetimeparts (v1) only supports Timestamp dtypes, got {dtype}"
                );
                let metadata_v2 = DateTimePartsV2Metadata::decode(metadata)?;
                vortex_ensure!(
                    metadata_v2.subseconds_divisor == 0,
                    "vortex.datetimeparts (v1) does not support subseconds_divisor {}",
                    metadata_v2.subseconds_divisor
                );
                let metadata = DateTimePartsMetadata::decode(metadata)?;
                vortex_ensure_eq!(
                    children.len(),
                    3,
                    "Expected 3 children for datetime-parts encoding, found {}",
                    children.len()
                );
                (
                    metadata.get_days_ptype()?,
                    metadata.get_seconds_ptype()?,
                    metadata.get_subseconds_ptype()?,
                    1,
                )
            } else if serialized_id == datetime_parts_v2_id() {
                let metadata = DateTimePartsV2Metadata::decode(metadata)?;
                validate_subseconds_divisor(kind.unit(), metadata.subseconds_divisor)?;
                vortex_ensure!(
                    children.len() >= 3,
                    "Expected at least 3 children for datetime-parts v2 encoding, found {}",
                    children.len()
                );
                (
                    metadata.get_days_ptype()?,
                    metadata.get_seconds_ptype()?,
                    metadata.get_subseconds_ptype()?,
                    metadata.subseconds_divisor,
                )
            } else {
                vortex_bail!(
                    "DateTimeParts plugin does not recognize the serialized ID {serialized_id}"
                );
            };

        let days = children.get(0, &DType::Primitive(days_ptype, dtype.nullability()), len)?;
        let seconds = children.get(
            1,
            &DType::Primitive(seconds_ptype, Nullability::NonNullable),
            len,
        )?;
        let subseconds = children.get(
            2,
            &DType::Primitive(subseconds_ptype, Nullability::NonNullable),
            len,
        )?;

        let slots = smallvec![Some(days), Some(seconds), Some(subseconds)];
        let data = DateTimePartsData { subseconds_divisor };
        Ok(Array::try_from_parts(ArrayParts::new(
            DateTimeParts,
            dtype.clone(),
            len,
            data,
            slots,
        ))?
        .into_array())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use prost::Message;
    use vortex_array::ArrayContext;
    use vortex_array::ArrayDeserialization;
    use vortex_array::ArrayId;
    use vortex_array::ArrayPlugin;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::TemporalArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::extension::datetime::Time;
    use vortex_array::extension::datetime::TimeUnit;
    use vortex_array::extension::datetime::Timestamp;
    use vortex_array::serde::SerializeOptions;
    use vortex_array::serde::SerializedArray;
    use vortex_array::session::ArraySessionExt;
    use vortex_array::validity::Validity;
    use vortex_buffer::ByteBufferMut;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;
    use vortex_session::registry::ReadContext;

    use super::*;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

    fn serde_roundtrip(array: &ArrayRef) -> VortexResult<(ArrayId, ArrayRef)> {
        let serialized_id = SESSION
            .array_serialize(array)?
            .expect("DateTimeParts is serializable")
            .serialized_id;
        let array_ctx = ArrayContext::empty();
        let mut bytes = ByteBufferMut::empty();
        for buf in array.serialize(&array_ctx, &SESSION, &SerializeOptions::default())? {
            bytes.extend_from_slice(buf.as_ref());
        }
        let decoded = SerializedArray::try_from(bytes.freeze())?.decode(
            array.dtype(),
            array.len(),
            &ReadContext::new(array_ctx.to_ids()),
            &SESSION,
        )?;
        Ok((serialized_id, decoded))
    }

    #[test]
    fn test_timestamp_unscaled_uses_v1_wire_format() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let temporal = TemporalArray::new_timestamp(
            buffer![0i64, 86_400_000_123, -86_400_000_456].into_array(),
            TimeUnit::Microseconds,
            Some("UTC".into()),
        );
        let dtp = DateTimeParts::try_from_temporal_v2(temporal.clone(), &mut ctx)?;
        assert_eq!(dtp.subseconds_divisor(), 1);

        let (id, decoded) = serde_roundtrip(&dtp.into_array())?;
        assert_eq!(id, datetime_parts_v1_id());
        assert_eq!(decoded.as_::<DateTimeParts>().subseconds_divisor(), 1);
        assert_arrays_eq!(decoded, temporal, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_timestamp_scaled_uses_v2_wire_format() -> VortexResult<()> {
        let cases: &[(TimeUnit, u32, &[i64])] = &[
            (
                TimeUnit::Microseconds,
                1_000,
                &[0, 1_000, 86_400_000_000 + 42_000, -500_000],
            ),
            (
                TimeUnit::Nanoseconds,
                1_000,
                &[0, 1_000, 86_400_000_000_000 + 123_000, -7_000],
            ),
            (
                TimeUnit::Nanoseconds,
                1_000_000,
                &[0, 1_000_000, 86_400_000_000_000 + 456_000_000, -9_000_000],
            ),
        ];
        let mut ctx = SESSION.create_execution_ctx();
        for &(unit, expected_divisor, values) in cases {
            let temporal = TemporalArray::new_timestamp(
                PrimitiveArray::from_iter(values.iter().copied()).into_array(),
                unit,
                Some("UTC".into()),
            );
            let dtp = DateTimeParts::try_from_temporal_v2(temporal.clone(), &mut ctx)?;
            assert_eq!(dtp.subseconds_divisor(), expected_divisor);

            let (id, decoded) = serde_roundtrip(&dtp.into_array())?;
            assert_eq!(id, datetime_parts_v2_id());
            assert_eq!(
                decoded.as_::<DateTimeParts>().subseconds_divisor(),
                expected_divisor
            );
            assert_arrays_eq!(decoded, temporal, &mut ctx);
        }
        Ok(())
    }

    #[test]
    fn test_time_uses_v2_wire_format() -> VortexResult<()> {
        let cases = [
            (
                TemporalArray::new_time(
                    buffer![0i32, 3_600, 86_399].into_array(),
                    TimeUnit::Seconds,
                ),
                1,
            ),
            (
                TemporalArray::new_time(
                    buffer![0i32, 3_600_500, 86_399_999].into_array(),
                    TimeUnit::Milliseconds,
                ),
                1,
            ),
            (
                TemporalArray::new_time(
                    buffer![0i64, 3_600_000_000, 86_399_000_000].into_array(),
                    TimeUnit::Microseconds,
                ),
                1,
            ),
            (
                TemporalArray::new_time(
                    buffer![0i64, 3_600_500_000, 86_399_123_000].into_array(),
                    TimeUnit::Microseconds,
                ),
                1_000,
            ),
            (
                TemporalArray::new_time(
                    buffer![0i64, 3_600_500_000_000, 86_399_123_000_000].into_array(),
                    TimeUnit::Nanoseconds,
                ),
                1_000_000,
            ),
            (
                TemporalArray::new_time(
                    buffer![0i64, 3_600_500_123_000, 86_399_123_456_000].into_array(),
                    TimeUnit::Nanoseconds,
                ),
                1_000,
            ),
        ];
        let mut ctx = SESSION.create_execution_ctx();
        for (temporal, expected_divisor) in cases {
            let dtp = DateTimeParts::try_from_temporal_v2(temporal.clone(), &mut ctx)?;
            assert_eq!(dtp.subseconds_divisor(), expected_divisor);

            let (id, decoded) = serde_roundtrip(&dtp.into_array())?;
            assert_eq!(id, datetime_parts_v2_id());
            assert_eq!(
                decoded.as_::<DateTimeParts>().subseconds_divisor(),
                expected_divisor
            );
            assert_arrays_eq!(decoded, temporal, &mut ctx);
        }
        Ok(())
    }

    #[test]
    fn test_v1_deserialize_rejects_time_dtype() {
        let dtype =
            DType::Extension(Time::new(TimeUnit::Microseconds, Nullability::NonNullable).erased());
        let metadata = DateTimePartsMetadata {
            days_ptype: PType::I32 as i32,
            seconds_ptype: PType::I32 as i32,
            subseconds_ptype: PType::I32 as i32,
        }
        .encode_to_vec();
        let children: [ArrayRef; 3] = [
            ConstantArray::new(0i32, 2).into_array(),
            ConstantArray::new(0i32, 2).into_array(),
            ConstantArray::new(0i32, 2).into_array(),
        ];
        let err = DateTimePartsPlugin
            .deserialize(
                ArrayDeserialization::new(
                    datetime_parts_v1_id(),
                    &dtype,
                    2,
                    &metadata,
                    &[],
                    &children,
                ),
                &SESSION,
            )
            .unwrap_err();
        assert!(err.to_string().contains("v1"), "{err}");
    }

    #[test]
    fn test_v1_deserialize_rejects_subseconds_divisor() {
        let dtype = DType::Extension(
            Timestamp::new(TimeUnit::Microseconds, Nullability::NonNullable).erased(),
        );
        let metadata = DateTimePartsV2Metadata {
            days_ptype: PType::I32 as i32,
            seconds_ptype: PType::I32 as i32,
            subseconds_ptype: PType::I32 as i32,
            subseconds_divisor: 1_000,
        }
        .encode_to_vec();
        let children: [ArrayRef; 3] = [
            ConstantArray::new(0i32, 2).into_array(),
            ConstantArray::new(0i32, 2).into_array(),
            ConstantArray::new(0i32, 2).into_array(),
        ];
        let err = DateTimePartsPlugin
            .deserialize(
                ArrayDeserialization::new(
                    datetime_parts_v1_id(),
                    &dtype,
                    2,
                    &metadata,
                    &[],
                    &children,
                ),
                &SESSION,
            )
            .unwrap_err();
        assert!(err.to_string().contains("subseconds_divisor"), "{err}");
    }

    #[test]
    fn test_v2_deserialize_rejects_invalid_divisor() {
        let cases = [
            (TimeUnit::Microseconds, 0),
            (TimeUnit::Microseconds, 7),
            (TimeUnit::Microseconds, 10),
            (TimeUnit::Seconds, 1_000),
            (TimeUnit::Milliseconds, 1_000),
            (TimeUnit::Microseconds, 1_000_000),
            (TimeUnit::Nanoseconds, 1_000_000_000),
        ];
        for (unit, divisor) in cases {
            let dtype = DType::Extension(Timestamp::new(unit, Nullability::NonNullable).erased());
            let metadata = DateTimePartsV2Metadata {
                days_ptype: PType::I32 as i32,
                seconds_ptype: PType::I32 as i32,
                subseconds_ptype: PType::I32 as i32,
                subseconds_divisor: divisor,
            }
            .encode_to_vec();
            let children: [ArrayRef; 3] = [
                ConstantArray::new(0i32, 2).into_array(),
                ConstantArray::new(0i32, 2).into_array(),
                ConstantArray::new(0i32, 2).into_array(),
            ];
            let err = DateTimePartsPlugin
                .deserialize(
                    ArrayDeserialization::new(
                        datetime_parts_v2_id(),
                        &dtype,
                        2,
                        &metadata,
                        &[],
                        &children,
                    ),
                    &SESSION,
                )
                .unwrap_err();
            assert!(err.to_string().contains("subseconds_divisor"), "{err}");

            // Direct construction also rejects the invalid divisor.
            let err = DateTimeParts::try_new_with_divisor(
                dtype,
                ConstantArray::new(0i32, 2).into_array(),
                ConstantArray::new(0i32, 2).into_array(),
                ConstantArray::new(0i32, 2).into_array(),
                divisor,
            )
            .unwrap_err();
            assert!(err.to_string().contains("subseconds_divisor"), "{err}");
        }
    }

    #[test]
    fn test_v2_deserialize_ignores_trailing_children() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let dtype = DType::Extension(
            Timestamp::new(TimeUnit::Microseconds, Nullability::NonNullable).erased(),
        );
        let metadata = DateTimePartsV2Metadata {
            days_ptype: PType::I32 as i32,
            seconds_ptype: PType::I32 as i32,
            subseconds_ptype: PType::I32 as i32,
            subseconds_divisor: 1_000,
        }
        .encode_to_vec();
        let base_children: [ArrayRef; 3] = [
            buffer![0i32, 1].into_array(),
            buffer![10i32, 20].into_array(),
            buffer![5i32, 500].into_array(),
        ];
        // A fourth trailing child with an unrelated dtype must be ignored without decoding.
        let extra_children: [ArrayRef; 4] = [
            base_children[0].clone(),
            base_children[1].clone(),
            base_children[2].clone(),
            buffer![7u8, 8].into_array(),
        ];

        let actual = DateTimePartsPlugin.deserialize(
            ArrayDeserialization::new(
                datetime_parts_v2_id(),
                &dtype,
                2,
                &metadata,
                &[],
                &extra_children,
            ),
            &SESSION,
        )?;
        let expected = DateTimePartsPlugin.deserialize(
            ArrayDeserialization::new(
                datetime_parts_v2_id(),
                &dtype,
                2,
                &metadata,
                &[],
                &base_children,
            ),
            &SESSION,
        )?;
        assert_arrays_eq!(actual, expected, &mut ctx);

        // Fewer than 3 children must still fail.
        let too_few: &[ArrayRef] = &base_children[..2];
        let err = DateTimePartsPlugin
            .deserialize(
                ArrayDeserialization::new(
                    datetime_parts_v2_id(),
                    &dtype,
                    2,
                    &metadata,
                    &[],
                    &too_few,
                ),
                &SESSION,
            )
            .unwrap_err();
        assert!(err.to_string().contains("at least 3 children"), "{err}");
        Ok(())
    }

    #[test]
    fn test_nullable_time_roundtrip() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::new(
            buffer![
                0i64,
                999_999_999_999,
                3_600_500_000,
                -123_456,
                86_399_123_000
            ],
            Validity::from_iter([true, false, true, false, true]),
        );
        let temporal = TemporalArray::new_time(values.into_array(), TimeUnit::Microseconds);
        let dtp = DateTimeParts::try_from_temporal_v2(temporal.clone(), &mut ctx)?;
        assert_eq!(dtp.subseconds_divisor(), 1_000);

        let (id, decoded) = serde_roundtrip(&dtp.into_array())?;
        assert_eq!(id, datetime_parts_v2_id());
        assert_arrays_eq!(decoded, temporal, &mut ctx);
        for i in 0..5 {
            assert_eq!(
                decoded.execute_scalar(i, &mut ctx)?,
                temporal.clone().into_array().execute_scalar(i, &mut ctx)?
            );
        }
        Ok(())
    }
}
