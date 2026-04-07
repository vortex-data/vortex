// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;

use prost::Message;
use vortex_array::ArrayId;
use vortex_array::ArrayPlugin;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::TypedArrayRef;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::arrays::ScalarFnVTable as ScalarFnArrayVTable;
use vortex_array::arrays::scalar_fn::ScalarFnArrayExt;
use vortex_array::arrays::scalar_fn::deserialize_scalar_fn_array;
use vortex_array::arrays::TemporalArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::extension::datetime::Timestamp;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ScalarFnRef;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_array::serde::ArrayChildren;
use vortex_error::VortexExpect as _;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::TemporalParts;
use crate::canonical::decode_to_temporal_parts;
use crate::split_temporal;

pub type DateTimePartsArray = ScalarFnArray;

#[derive(Clone, prost::Message)]
#[repr(C)]
pub struct DateTimePartsMetadata {
    #[prost(enumeration = "PType", tag = "1")]
    pub days_ptype: i32,
    #[prost(enumeration = "PType", tag = "2")]
    pub seconds_ptype: i32,
    #[prost(enumeration = "PType", tag = "3")]
    pub subseconds_ptype: i32,
}

impl DateTimePartsMetadata {
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

#[derive(Clone, PartialEq, Eq, Hash, prost::Message)]
pub struct DateTimePartsOptions {
    #[prost(uint32, tag = "1")]
    pub time_unit: u32,
    #[prost(string, optional, tag = "2")]
    pub timezone: Option<String>,
    #[prost(bool, tag = "3")]
    pub nullable: bool,
    #[prost(enumeration = "PType", tag = "4")]
    pub days_ptype: i32,
    #[prost(enumeration = "PType", tag = "5")]
    pub seconds_ptype: i32,
    #[prost(enumeration = "PType", tag = "6")]
    pub subseconds_ptype: i32,
}

impl DateTimePartsOptions {
    pub fn try_new(
        time_unit: TimeUnit,
        timezone: Option<String>,
        nullability: Nullability,
        days_ptype: PType,
        seconds_ptype: PType,
        subseconds_ptype: PType,
    ) -> Self {
        Self {
            time_unit: time_unit as u32,
            timezone,
            nullable: bool::from(nullability),
            days_ptype: days_ptype as i32,
            seconds_ptype: seconds_ptype as i32,
            subseconds_ptype: subseconds_ptype as i32,
        }
    }

    pub fn time_unit(&self) -> VortexResult<TimeUnit> {
        let time_unit: u8 = self
            .time_unit
            .try_into()
            .map_err(|_| vortex_err!("Invalid time unit {}", self.time_unit))?;
        TimeUnit::try_from(time_unit)
    }

    pub fn nullability(&self) -> Nullability {
        self.nullable.into()
    }

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

    pub fn result_dtype(&self) -> VortexResult<DType> {
        Ok(DType::Extension(
            Timestamp::new_with_tz(
                self.time_unit()?,
                self.timezone.clone().map(Into::into),
                self.nullability(),
            )
            .erased(),
        ))
    }

    pub fn child_dtypes(&self) -> VortexResult<[DType; NUM_SLOTS]> {
        Ok([
            DType::Primitive(self.get_days_ptype()?, self.nullability()),
            DType::Primitive(self.get_seconds_ptype()?, Nullability::NonNullable),
            DType::Primitive(self.get_subseconds_ptype()?, Nullability::NonNullable),
        ])
    }

    fn from_dtype(
        dtype: &DType,
        days_ptype: PType,
        seconds_ptype: PType,
        subseconds_ptype: PType,
    ) -> VortexResult<Self> {
        let DType::Extension(ext) = dtype else {
            vortex_bail!("DateTimeParts requires timestamp extension dtype, found {dtype}");
        };
        let Some(timestamp) = ext.metadata_opt::<Timestamp>() else {
            vortex_bail!("DateTimeParts requires timestamp extension dtype, found {dtype}");
        };
        Ok(Self::try_new(
            timestamp.unit,
            timestamp.tz.as_ref().map(|tz| tz.to_string()),
            dtype.nullability(),
            days_ptype,
            seconds_ptype,
            subseconds_ptype,
        ))
    }
}

impl fmt::Display for DateTimePartsOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "unit={}, tz={:?}, nullable={}, days={}, seconds={}, subseconds={}",
            self.time_unit().map_err(|_| fmt::Error)?,
            self.timezone,
            self.nullable,
            self.days_ptype,
            self.seconds_ptype,
            self.subseconds_ptype
        )
    }
}

pub(crate) const DAYS_SLOT: usize = 0;
pub(crate) const SECONDS_SLOT: usize = 1;
pub(crate) const SUBSECONDS_SLOT: usize = 2;
pub(crate) const NUM_SLOTS: usize = 3;
pub(crate) const SLOT_NAMES: [&str; NUM_SLOTS] = ["days", "seconds", "subseconds"];

pub trait DateTimePartsArrayExt: TypedArrayRef<ScalarFnArrayVTable> + ScalarFnArrayExt {
    fn date_time_parts_options(&self) -> &DateTimePartsOptions {
        self.scalar_fn()
            .as_opt::<DateTimeParts>()
            .vortex_expect("DateTimeParts scalar function")
    }

    fn days(&self) -> &ArrayRef {
        self.child_at(DAYS_SLOT)
    }

    fn seconds(&self) -> &ArrayRef {
        self.child_at(SECONDS_SLOT)
    }

    fn subseconds(&self) -> &ArrayRef {
        self.child_at(SUBSECONDS_SLOT)
    }
}

impl<T: TypedArrayRef<ScalarFnArrayVTable>> DateTimePartsArrayExt for T {}

#[derive(Clone, Debug)]
pub struct DateTimeParts;

impl DateTimeParts {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.datetimeparts.fn");
    pub const LEGACY_ID: ArrayId = ArrayId::new_ref("vortex.datetimeparts");

    pub fn try_new_fn(
        dtype: DType,
        days_ptype: PType,
        seconds_ptype: PType,
        subseconds_ptype: PType,
    ) -> VortexResult<ScalarFnRef> {
        Ok(Self.bind(DateTimePartsOptions::from_dtype(
            &dtype,
            days_ptype,
            seconds_ptype,
            subseconds_ptype,
        )?))
    }

    pub fn try_new_array(
        dtype: DType,
        days: ArrayRef,
        seconds: ArrayRef,
        subseconds: ArrayRef,
    ) -> VortexResult<DateTimePartsArray> {
        let len = days.len();
        validate_child_arrays(&dtype, &days, &seconds, &subseconds, len)?;

        let days_ptype = PType::try_from(days.dtype())?;
        let seconds_ptype = PType::try_from(seconds.dtype())?;
        let subseconds_ptype = PType::try_from(subseconds.dtype())?;
        let scalar_fn = Self::try_new_fn(dtype, days_ptype, seconds_ptype, subseconds_ptype)?;

        ScalarFnArray::try_new(scalar_fn, vec![days, seconds, subseconds], len)
    }

    pub fn try_new(
        dtype: DType,
        days: ArrayRef,
        seconds: ArrayRef,
        subseconds: ArrayRef,
    ) -> VortexResult<DateTimePartsArray> {
        Self::try_new_array(dtype, days, seconds, subseconds)
    }

    pub fn try_from_temporal(temporal: TemporalArray) -> VortexResult<DateTimePartsArray> {
        let dtype = temporal.dtype().clone();
        let TemporalParts {
            days,
            seconds,
            subseconds,
        } = split_temporal(temporal)?;
        Self::try_new_array(dtype, days, seconds, subseconds)
    }
}

impl ScalarFnVTable for DateTimeParts {
    type Options = DateTimePartsOptions;

    fn id(&self) -> vortex_array::scalar_fn::ScalarFnId {
        Self::ID
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(options.encode_to_vec()))
    }

    fn deserialize(&self, metadata: &[u8], _session: &VortexSession) -> VortexResult<Self::Options> {
        Ok(DateTimePartsOptions::decode(metadata)?)
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(NUM_SLOTS)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        ChildName::new_ref(SLOT_NAMES[child_idx])
    }

    fn fmt_sql(
        &self,
        _options: &Self::Options,
        expr: &vortex_array::expr::Expression,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        write!(f, "datetime_parts(")?;
        for idx in 0..NUM_SLOTS {
            if idx > 0 {
                write!(f, ", ")?;
            }
            expr.child(idx).fmt_sql(f)?;
        }
        write!(f, ")")
    }

    fn return_dtype(&self, options: &Self::Options, args: &[DType]) -> VortexResult<DType> {
        vortex_ensure!(
            args.len() == NUM_SLOTS,
            "DateTimeParts expects {} arguments, found {}",
            NUM_SLOTS,
            args.len()
        );
        let expected = options.child_dtypes()?;
        vortex_ensure!(
            args == expected.as_slice(),
            "DateTimeParts child dtypes do not match serialized options. Expected {:?}, found {:?}",
            expected,
            args
        );
        options.result_dtype()
    }

    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let days = args.get(DAYS_SLOT)?;
        let seconds = args.get(SECONDS_SLOT)?;
        let subseconds = args.get(SUBSECONDS_SLOT)?;
        Ok(decode_to_temporal_parts(options, &days, &seconds, &subseconds, ctx)?.into_array())
    }
}

#[derive(Clone, Debug)]
pub struct DateTimePartsArrayPlugin;

impl ArrayPlugin for DateTimePartsArrayPlugin {
    fn id(&self) -> ArrayId {
        DateTimeParts::ID
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        let array = deserialize_scalar_fn_array(dtype, len, metadata, buffers, children, session)?;
        vortex_ensure!(
            array
                .as_opt::<ScalarFnArrayVTable>()
                .is_some_and(|array| array.scalar_fn().is::<DateTimeParts>()),
            "Expected DateTimeParts scalar function array"
        );
        Ok(array)
    }
}

#[derive(Clone, Debug)]
pub struct LegacyDateTimePartsArrayPlugin;

impl ArrayPlugin for LegacyDateTimePartsArrayPlugin {
    fn id(&self) -> ArrayId {
        DateTimeParts::LEGACY_ID
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        vortex_ensure!(
            buffers.is_empty(),
            "DateTimeParts legacy encoding does not support serialized buffers"
        );

        let metadata = DateTimePartsMetadata::decode(metadata)?;
        vortex_ensure!(
            children.len() == NUM_SLOTS,
            "Expected {} children for datetime-parts encoding, found {}",
            NUM_SLOTS,
            children.len()
        );

        let days = children.get(
            DAYS_SLOT,
            &DType::Primitive(metadata.get_days_ptype()?, dtype.nullability()),
            len,
        )?;
        let seconds = children.get(
            SECONDS_SLOT,
            &DType::Primitive(metadata.get_seconds_ptype()?, Nullability::NonNullable),
            len,
        )?;
        let subseconds = children.get(
            SUBSECONDS_SLOT,
            &DType::Primitive(metadata.get_subseconds_ptype()?, Nullability::NonNullable),
            len,
        )?;

        Ok(DateTimeParts::try_new_array(dtype.clone(), days, seconds, subseconds)?.into_array())
    }
}

fn validate_child_arrays(
    dtype: &DType,
    days: &ArrayRef,
    seconds: &ArrayRef,
    subseconds: &ArrayRef,
    len: usize,
) -> VortexResult<()> {
    vortex_ensure!(days.len() == len, "expected len {len}, got {}", days.len());
    vortex_ensure!(seconds.len() == len, "expected len {len}, got {}", seconds.len());
    vortex_ensure!(
        subseconds.len() == len,
        "expected len {len}, got {}",
        subseconds.len()
    );

    let DType::Extension(ext) = dtype else {
        vortex_bail!("DateTimeParts requires timestamp extension dtype, found {dtype}");
    };
    vortex_ensure!(
        ext.metadata_opt::<Timestamp>().is_some(),
        "DateTimeParts requires timestamp extension dtype, found {dtype}"
    );

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

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::hash::Hasher;

    use prost::Message;
    use vortex_array::Array;
    use vortex_array::ArrayContext;
    use vortex_array::ArrayEq;
    use vortex_array::ArrayHash;
    use vortex_array::ArrayParts;
    use vortex_array::ArrayRef;
    use vortex_array::ArrayView;
    use vortex_array::ExecutionCtx;
    use vortex_array::ExecutionResult;
    use vortex_array::IntoArray;
    use vortex_array::NotSupported;
    use vortex_array::Precision;
    use vortex_array::ValidityChild;
    use vortex_array::ValidityVTableFromChild;
    use vortex_array::VTable;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::TemporalArray;
    use vortex_array::buffer::BufferHandle;
    use vortex_array::extension::datetime::TimeUnit;
    use vortex_array::serde::SerializeOptions;
    use vortex_array::serde::SerializedArray;
    use vortex_array::assert_arrays_eq;
    use vortex_buffer::{ByteBufferMut, buffer};
    use vortex_session::VortexSession;
    use vortex_session::registry::ReadContext;

    use super::*;
    use crate::initialize;

    #[derive(Clone, Debug)]
    struct LegacyTestDateTimePartsData;

    impl ArrayHash for LegacyTestDateTimePartsData {
        fn array_hash<H: Hasher>(&self, _state: &mut H, _precision: Precision) {}
    }

    impl ArrayEq for LegacyTestDateTimePartsData {
        fn array_eq(&self, _other: &Self, _precision: Precision) -> bool {
            true
        }
    }

    #[derive(Clone, Debug)]
    struct LegacyTestDateTimeParts;

    impl LegacyTestDateTimeParts {
        fn try_new(
            dtype: DType,
            days: ArrayRef,
            seconds: ArrayRef,
            subseconds: ArrayRef,
        ) -> VortexResult<Array<Self>> {
            let len = days.len();
            validate_child_arrays(&dtype, &days, &seconds, &subseconds, len)?;
            Ok(unsafe {
                Array::from_parts_unchecked(
                    ArrayParts::new(
                        Self,
                        dtype,
                        len,
                        LegacyTestDateTimePartsData,
                    )
                    .with_slots(vec![Some(days), Some(seconds), Some(subseconds)]),
                )
            })
        }
    }

    impl VTable for LegacyTestDateTimeParts {
        type ArrayData = LegacyTestDateTimePartsData;
        type OperationsVTable = NotSupported;
        type ValidityVTable = ValidityVTableFromChild;

        fn id(&self) -> ArrayId {
            DateTimeParts::LEGACY_ID
        }

        fn validate(
            &self,
            _data: &Self::ArrayData,
            dtype: &DType,
            len: usize,
            slots: &[Option<ArrayRef>],
        ) -> VortexResult<()> {
            validate_child_arrays(
                dtype,
                slots[DAYS_SLOT].as_ref().vortex_expect("days"),
                slots[SECONDS_SLOT].as_ref().vortex_expect("seconds"),
                slots[SUBSECONDS_SLOT]
                    .as_ref()
                    .vortex_expect("subseconds"),
                len,
            )
        }

        fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
            0
        }

        fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
            panic!("unexpected legacy datetimeparts buffer index {idx}")
        }

        fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
            None
        }

        fn serialize(array: ArrayView<'_, Self>) -> VortexResult<Option<Vec<u8>>> {
            Ok(Some(
                DateTimePartsMetadata {
                    days_ptype: PType::try_from(
                        array.slots()[DAYS_SLOT].as_ref().vortex_expect("days").dtype(),
                    )? as i32,
                    seconds_ptype: PType::try_from(
                        array.slots()[SECONDS_SLOT]
                            .as_ref()
                            .vortex_expect("seconds")
                            .dtype(),
                    )? as i32,
                    subseconds_ptype: PType::try_from(
                        array.slots()[SUBSECONDS_SLOT]
                            .as_ref()
                            .vortex_expect("subseconds")
                            .dtype(),
                    )? as i32,
                }
                .encode_to_vec(),
            ))
        }

        fn deserialize(
            &self,
            _dtype: &DType,
            _len: usize,
            _metadata: &[u8],
            _buffers: &[BufferHandle],
            _children: &dyn ArrayChildren,
            _session: &VortexSession,
        ) -> VortexResult<ArrayParts<Self>> {
            unreachable!("legacy test vtable is only used for serialization")
        }

        fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
            SLOT_NAMES[idx].to_string()
        }

        fn execute(
            _array: Array<Self>,
            _ctx: &mut ExecutionCtx,
        ) -> VortexResult<ExecutionResult> {
            unreachable!("legacy test vtable is only used for serialization")
        }
    }

    impl ValidityChild<LegacyTestDateTimeParts> for LegacyTestDateTimeParts {
        fn validity_child(array: ArrayView<'_, LegacyTestDateTimeParts>) -> ArrayRef {
            array.slots()[DAYS_SLOT]
                .as_ref()
                .vortex_expect("days")
                .clone()
        }
    }

    fn round_trip(array: ArrayRef) -> VortexResult<ArrayRef> {
        let dtype = array.dtype().clone();
        let len = array.len();

        let ctx = ArrayContext::empty();
        let serialized = array.serialize(&ctx, &SerializeOptions::default())?;

        let mut concat = ByteBufferMut::empty();
        for buffer in serialized {
            concat.extend_from_slice(buffer.as_ref());
        }

        let session = VortexSession::empty();
        initialize(&session);

        SerializedArray::try_from(concat.freeze())?.decode(
            &dtype,
            len,
            &ReadContext::new(ctx.to_ids()),
            &session,
        )
    }

    #[test]
    fn test_scalar_fn_datetime_parts_serde_roundtrip() -> VortexResult<()> {
        let array = DateTimeParts::try_from_temporal(TemporalArray::new_timestamp(
            PrimitiveArray::from_option_iter([Some(0i64), None, Some(86_401_234)]).into_array(),
            TimeUnit::Milliseconds,
            Some("UTC".into()),
        ))?
        .into_array();

        let decoded = round_trip(array.clone())?;
        let original = array
            .as_opt::<ScalarFnArrayVTable>()
            .vortex_expect("DateTimeParts scalar-fn array");
        let decoded_view = decoded
            .as_opt::<ScalarFnArrayVTable>()
            .vortex_expect("DateTimeParts scalar-fn array");

        assert!(decoded_view.scalar_fn().is::<DateTimeParts>());
        assert_eq!(decoded_view.dtype(), array.dtype());
        assert_eq!(decoded_view.child_at(DAYS_SLOT).dtype(), original.child_at(DAYS_SLOT).dtype());
        assert_eq!(
            decoded_view.child_at(SECONDS_SLOT).dtype(),
            original.child_at(SECONDS_SLOT).dtype()
        );
        assert_eq!(
            decoded_view.child_at(SUBSECONDS_SLOT).dtype(),
            original.child_at(SUBSECONDS_SLOT).dtype()
        );
        assert_arrays_eq!(decoded.to_canonical()?.into_array(), array.to_canonical()?.into_array());
        Ok(())
    }

    #[test]
    fn test_legacy_datetime_parts_deserializes_to_scalar_fn_array() -> VortexResult<()> {
        let dtype = DType::Extension(
            Timestamp::new_with_tz(TimeUnit::Milliseconds, Some("UTC".into()), Nullability::Nullable)
                .erased(),
        );
        let legacy = LegacyTestDateTimeParts::try_new(
            dtype,
            PrimitiveArray::from_option_iter([Some(0i64), None, Some(1)]).into_array(),
            buffer![0i32, 1, 2].into_array(),
            buffer![0i16, 234, 999].into_array(),
        )?
        .into_array();

        let decoded = round_trip(legacy)?;
        let decoded = decoded
            .as_opt::<ScalarFnArrayVTable>()
            .vortex_expect("DateTimeParts scalar-fn array");

        assert!(decoded.scalar_fn().is::<DateTimeParts>());
        assert_eq!(decoded.encoding_id(), DateTimeParts::ID);
        Ok(())
    }
}
