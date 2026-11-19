// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Checked conversions for the built-in temporal dtypes.
//!
//! Dispatch uses the concrete temporal metadata before a conversion reads the storage.
//! Scalars and arrays share the same count arithmetic.

use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;

use super::AnyTemporal;
use super::TemporalMetadata;
use super::TimeUnit;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ExtensionArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::extension::ExtensionArrayExt;
use crate::canonical::Canonical;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::PType;
use crate::dtype::extension::ExtDTypeRef;
use crate::scalar::Scalar;

/// Read the metadata of a built-in temporal dtype.
pub(crate) fn temporal_metadata(dtype: &DType) -> Option<TemporalMetadata<'_>> {
    dtype.as_extension_opt()?.metadata_opt::<AnyTemporal>()
}

/// Return the size of a unit in nanoseconds. Temporal counts use 86,400-second days.
pub(crate) fn nanos_per_unit(unit: TimeUnit) -> i64 {
    match unit {
        TimeUnit::Nanoseconds => 1,
        TimeUnit::Microseconds => 1_000,
        TimeUnit::Milliseconds => 1_000_000,
        TimeUnit::Seconds => 1_000_000_000,
        TimeUnit::Days => 86_400_000_000_000,
    }
}

/// Return the number of unit counts in one day.
pub(crate) fn units_per_day(unit: TimeUnit) -> i64 {
    nanos_per_unit(TimeUnit::Days) / nanos_per_unit(unit)
}

/// A value-preserving conversion authorized by the concrete source and target dtypes.
pub(crate) struct TemporalCast {
    source_unit: TimeUnit,
    target_unit: TimeUnit,
    scale: Scale,
    input_domain: InputDomain,
}

/// Extra range constraints on the input count.
enum InputDomain {
    AnyCount,
    DayAlignedMilliseconds,
    TimeOfDay,
}

/// Exact scale factor between the units.
enum Scale {
    Multiply(i64),
    Divide(i64),
}

impl TemporalCast {
    /// Select a cast only for a supported pair of concrete temporal dtypes.
    pub(crate) fn classify(source: &DType, target: &DType) -> Option<Self> {
        let source = temporal_metadata(source)?;
        let target = temporal_metadata(target)?;
        let source_unit = source.time_unit();
        let target_unit = target.time_unit();
        let source_nanos = nanos_per_unit(source_unit);
        let target_nanos = nanos_per_unit(target_unit);
        let finer = target_nanos < source_nanos;
        let input_domain = match (source, target) {
            (
                TemporalMetadata::Timestamp(_, source_zone),
                TemporalMetadata::Timestamp(_, target_zone),
            ) if finer && source_zone == target_zone => InputDomain::AnyCount,
            (TemporalMetadata::Time(_), TemporalMetadata::Time(_)) if finer => {
                InputDomain::TimeOfDay
            }
            (
                TemporalMetadata::Date(TimeUnit::Days),
                TemporalMetadata::Date(TimeUnit::Milliseconds),
            )
            | (TemporalMetadata::Date(TimeUnit::Days), TemporalMetadata::Timestamp(_, None)) => {
                InputDomain::AnyCount
            }
            (
                TemporalMetadata::Date(TimeUnit::Milliseconds),
                TemporalMetadata::Timestamp(_, None),
            ) => InputDomain::DayAlignedMilliseconds,
            _ => return None,
        };
        let scale = if source_nanos >= target_nanos {
            Scale::Multiply(source_nanos / target_nanos)
        } else {
            Scale::Divide(target_nanos / source_nanos)
        };

        Some(Self {
            source_unit,
            target_unit,
            scale,
            input_domain,
        })
    }

    fn convert(&self, value: i64) -> VortexResult<i64> {
        match self.input_domain {
            InputDomain::AnyCount => {}
            InputDomain::TimeOfDay => check_time(value, self.source_unit)?,
            InputDomain::DayAlignedMilliseconds => {
                vortex_ensure_eq!(
                    value % units_per_day(TimeUnit::Milliseconds),
                    0,
                    "Expected a day-aligned Date(ms), got {value}"
                );
            }
        }

        match self.scale {
            Scale::Multiply(factor) => value.checked_mul(factor).ok_or_else(|| {
                vortex_err!(
                    "Temporal cast from {} to {} overflows for {value}",
                    self.source_unit,
                    self.target_unit,
                )
            }),
            // This occurs only for day-aligned Date(ms) to Timestamp(s).
            Scale::Divide(divisor) => Ok(value / divisor),
        }
    }

    /// Convert a scalar to the target dtype used to select the cast.
    pub(crate) fn cast_scalar(&self, input: &Scalar, target: &DType) -> VortexResult<Scalar> {
        transform_scalar(input, target, |value| self.convert(value))
    }

    /// Convert the visible array rows to the target dtype used to select the cast.
    pub(crate) fn cast_array(
        &self,
        input: ArrayRef,
        target: &DType,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        transform_temporal(input, target, ctx, |value| self.convert(value))
    }
}

/// Check the time-of-day range before converting a count or returning it as a time.
pub(crate) fn check_time(value: i64, unit: TimeUnit) -> VortexResult<()> {
    vortex_ensure!(
        (0..units_per_day(unit)).contains(&value),
        "Expected a time of day in {unit}, got {value}"
    );
    Ok(())
}

/// Apply a count conversion with the target storage width and nullability.
fn transform_scalar(
    input: &Scalar,
    target: &DType,
    convert: impl Fn(i64) -> VortexResult<i64>,
) -> VortexResult<Scalar> {
    if input.is_null() {
        vortex_ensure!(target.is_nullable(), "Cannot cast null to {target}");
        return Ok(Scalar::null(target.clone()));
    }

    let target_ext = target
        .as_extension_opt()
        .vortex_expect("conversion target is temporal");
    let storage = input.as_extension().to_storage_scalar();
    let value = match storage.as_primitive().ptype() {
        PType::I32 => storage.as_primitive().typed_value::<i32>().map(i64::from),
        PType::I64 => storage.as_primitive().typed_value::<i64>(),
        _ => vortex_bail!("Expected integer temporal storage, got {}", storage.dtype()),
    }
    .vortex_expect("the input is valid at this row");
    let value = convert(value)?;
    let storage = match target_ext.storage_dtype().as_ptype() {
        PType::I32 => Scalar::primitive(i32::try_from(value)?, target.nullability()),
        PType::I64 => Scalar::primitive(value, target.nullability()),
        _ => vortex_bail!("Expected integer temporal storage, got {target}"),
    };
    Scalar::try_extension_ref(target_ext.clone(), storage)
}

/// Apply a conversion to the non-null counts of a typed temporal array.
pub(crate) fn transform_temporal(
    input: ArrayRef,
    target: &DType,
    ctx: &mut ExecutionCtx,
    convert: impl Fn(i64) -> VortexResult<i64>,
) -> VortexResult<ArrayRef> {
    if input.is_empty() {
        return Ok(Canonical::empty(target).into_array());
    }
    if let Some(scalar) = input.as_constant() {
        let converted = transform_scalar(scalar, target, convert)?;
        return Ok(ConstantArray::new(converted, input.len()).into_array());
    }

    let ext = input.execute::<ExtensionArray>(ctx)?;
    let storage = ext.storage_array().clone().execute::<PrimitiveArray>(ctx)?;
    let target = target
        .as_extension_opt()
        .vortex_expect("conversion target is temporal");
    match storage.ptype() {
        PType::I32 => transform_buffer(&storage, &storage.to_buffer::<i32>(), target, ctx, convert),
        PType::I64 => transform_buffer(&storage, &storage.to_buffer::<i64>(), target, ctx, convert),
        _ => vortex_bail!("Expected integer temporal storage, got {}", storage.ptype()),
    }
}

/// Select the output storage width and apply the conversion to valid rows.
fn transform_buffer<T: NativePType + Into<i64>>(
    storage: &PrimitiveArray,
    values: &[T],
    target: &ExtDTypeRef,
    ctx: &mut ExecutionCtx,
    convert: impl Fn(i64) -> VortexResult<i64>,
) -> VortexResult<ArrayRef> {
    let validity = storage.validity()?.cast_nullability(
        target.storage_dtype().nullability(),
        values.len(),
        ctx,
    )?;
    let mask = validity.execute_mask(values.len(), ctx)?;
    let converted =
        values.iter().zip(mask.iter()).map(
            |(&value, valid)| {
                if valid { convert(value.into()) } else { Ok(0) }
            },
        );
    let output = match target.storage_dtype().as_ptype() {
        PType::I32 => {
            let mut buffer = BufferMut::<i32>::with_capacity(values.len());
            for value in converted {
                buffer.push(i32::try_from(value?)?);
            }
            PrimitiveArray::new(buffer.freeze(), validity)
        }
        PType::I64 => {
            let mut buffer = BufferMut::<i64>::with_capacity(values.len());
            for value in converted {
                buffer.push(value?);
            }
            PrimitiveArray::new(buffer.freeze(), validity)
        }
        _ => vortex_bail!(
            "Expected integer temporal storage, got {}",
            target.storage_dtype()
        ),
    };
    Ok(ExtensionArray::try_new(target.clone(), output.into_array())?.into_array())
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_session::VortexSession;

    use super::*;
    use crate::VortexSessionExecute;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::Nullability;
    use crate::extension::datetime::Date;
    use crate::extension::datetime::Time;
    use crate::extension::datetime::Timestamp;
    use crate::test_harness::ExpectedOutcome;
    use crate::test_harness::assert_execution_paths;
    use crate::validity::Validity;

    const NON_NULLABLE: Nullability = Nullability::NonNullable;

    #[test]
    fn test_widen_timestamp_skips_null_payload() -> VortexResult<()> {
        let source = Timestamp::new_with_tz(
            TimeUnit::Milliseconds,
            Some("UTC".into()),
            Nullability::Nullable,
        )
        .erased();
        let target = Timestamp::new_with_tz(
            TimeUnit::Nanoseconds,
            Some("UTC".into()),
            Nullability::Nullable,
        )
        .erased();
        let storage = PrimitiveArray::new(
            buffer![1_000i64, i64::MAX],
            Validity::from_iter([true, false]),
        );
        let input = ExtensionArray::try_new(source, storage.into_array())?.into_array();
        let expected = ExtensionArray::try_new(
            target.clone(),
            PrimitiveArray::from_option_iter([Some(1_000_000_000i64), None]).into_array(),
        )?
        .into_array();
        let target = DType::Extension(target);
        let mut ctx = VortexSession::empty().create_execution_ctx();
        assert_execution_paths(
            &[input],
            |inputs| inputs[0].cast(target.clone()),
            ExpectedOutcome::Array(expected),
            &mut ctx,
        );
        Ok(())
    }

    #[test]
    fn test_widen_timestamp_overflow() -> VortexResult<()> {
        let source = Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased();
        let target = DType::Extension(
            Timestamp::new(TimeUnit::Nanoseconds, Nullability::NonNullable).erased(),
        );
        let input = ExtensionArray::try_new(source, buffer![i64::MAX].into_array())?.into_array();

        assert_execution_paths(
            &[input],
            |inputs| inputs[0].cast(target.clone()),
            ExpectedOutcome::Error,
            &mut VortexSession::empty().create_execution_ctx(),
        );
        Ok(())
    }

    #[rstest]
    #[case::date_to_milliseconds(
        Date::new(TimeUnit::Days, NON_NULLABLE).erased(),
        buffer![-1i32, 2].into_array(),
        Date::new(TimeUnit::Milliseconds, NON_NULLABLE).erased(),
        buffer![-86_400_000i64, 172_800_000].into_array(),
    )]
    #[case::date_to_timestamp(
        Date::new(TimeUnit::Days, NON_NULLABLE).erased(),
        buffer![-1i32, 2].into_array(),
        Timestamp::new(TimeUnit::Milliseconds, NON_NULLABLE).erased(),
        buffer![-86_400_000i64, 172_800_000].into_array(),
    )]
    #[case::time_to_nanoseconds(
        Time::new(TimeUnit::Seconds, NON_NULLABLE).erased(),
        buffer![0i32, 86_399].into_array(),
        Time::new(TimeUnit::Nanoseconds, NON_NULLABLE).erased(),
        buffer![0i64, 86_399_000_000_000].into_array(),
    )]
    fn test_exact_date_and_time_casts(
        #[case] source: ExtDTypeRef,
        #[case] storage: ArrayRef,
        #[case] target: ExtDTypeRef,
        #[case] expected_storage: ArrayRef,
    ) -> VortexResult<()> {
        let input = ExtensionArray::try_new(source, storage)?.into_array();
        let expected = ExtensionArray::try_new(target.clone(), expected_storage)?.into_array();
        assert_execution_paths(
            &[input],
            |inputs| inputs[0].cast(DType::Extension(target.clone())),
            ExpectedOutcome::Array(expected),
            &mut VortexSession::empty().create_execution_ctx(),
        );
        Ok(())
    }

    #[test]
    fn test_date_to_timestamp_requires_day_alignment() -> VortexResult<()> {
        let source = Date::new(TimeUnit::Milliseconds, NON_NULLABLE).erased();
        let target = Timestamp::new(TimeUnit::Milliseconds, NON_NULLABLE).erased();
        let input = ExtensionArray::try_new(source, buffer![1i64].into_array())?.into_array();
        assert_execution_paths(
            &[input],
            |inputs| inputs[0].cast(DType::Extension(target.clone())),
            ExpectedOutcome::Error,
            &mut VortexSession::empty().create_execution_ctx(),
        );
        Ok(())
    }
}
