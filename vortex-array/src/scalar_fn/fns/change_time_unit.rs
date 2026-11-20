// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Convert a temporal value to a coarser unit with an explicit rounding mode.
//!
//! `cast` performs only the temporal conversions that are exact for every valid value. A coarser
//! unit can drop part of a count, so this function makes the caller choose a [`TemporalRounding`].

use std::fmt;

use prost::Message;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::dtype::DType;
use crate::extension::datetime::AnyTemporal;
use crate::extension::datetime::Date;
use crate::extension::datetime::TemporalMetadata;
use crate::extension::datetime::TemporalRounding;
use crate::extension::datetime::Time;
use crate::extension::datetime::TimeUnit;
use crate::extension::datetime::Timestamp;
use crate::extension::datetime::check_time_of_day;
use crate::extension::datetime::counts::map_counts;
use crate::proto::expr as pb;
use crate::proto::expr::change_time_unit_opts::Rounding as PbRounding;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;

/// Options for [`ChangeTimeUnit`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChangeTimeUnitOptions {
    /// The target unit, which must be coarser than the input unit.
    pub unit: TimeUnit,
    /// How to round a count that the target unit cannot represent exactly.
    pub rounding: TemporalRounding,
}

impl fmt::Display for ChangeTimeUnitOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unit={}, rounding={}", self.unit, self.rounding)
    }
}

/// Convert a [`Date`], [`Time`], or [`Timestamp`] to a coarser unit.
///
/// See [`change_time_unit`](crate::expr::change_time_unit) for the contract.
#[derive(Clone)]
pub struct ChangeTimeUnit;

impl ScalarFnVTable for ChangeTimeUnit {
    type Options = ChangeTimeUnitOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.change_time_unit");
        *ID
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        let rounding = match options.rounding {
            TemporalRounding::Exact => PbRounding::Exact,
            TemporalRounding::TowardZero => PbRounding::TowardZero,
            TemporalRounding::Floor => PbRounding::Floor,
            TemporalRounding::NearestTiesAwayFromZero => PbRounding::NearestTiesAwayFromZero,
        };

        Ok(Some(
            pb::ChangeTimeUnitOpts {
                unit: u32::from(u8::from(options.unit)),
                rounding: rounding.into(),
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(
        &self,
        metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        let options = pb::ChangeTimeUnitOpts::decode(metadata)?;
        let rounding = match PbRounding::try_from(options.rounding) {
            Ok(PbRounding::Exact) => TemporalRounding::Exact,
            Ok(PbRounding::TowardZero) => TemporalRounding::TowardZero,
            Ok(PbRounding::Floor) => TemporalRounding::Floor,
            Ok(PbRounding::NearestTiesAwayFromZero) => TemporalRounding::NearestTiesAwayFromZero,
            Err(_) => vortex_bail!(
                "Expected a known temporal rounding mode, got {}",
                options.rounding
            ),
        };

        Ok(ChangeTimeUnitOptions {
            unit: TimeUnit::try_from(u8::try_from(options.unit)?)?,
            rounding,
        })
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _options: &Self::Options, _child_idx: usize) -> ChildName {
        ChildName::from("input")
    }

    fn return_dtype(&self, options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        let input = &arg_dtypes[0];
        let metadata = input
            .as_extension_opt()
            .and_then(|ext| ext.metadata_opt::<AnyTemporal>())
            .ok_or_else(|| vortex_err!("Expected a temporal dtype, got {input}"))?;
        let source_unit = metadata.time_unit();
        let unit = options.unit;
        vortex_ensure!(
            unit.nanos_per_unit() > source_unit.nanos_per_unit(),
            "Expected a unit coarser than {source_unit}, got {unit}"
        );

        let nullability = input.nullability();
        let target = match metadata {
            TemporalMetadata::Timestamp(_, zone) => {
                Timestamp::try_new_with_tz(unit, zone.clone(), nullability)?.erased()
            }
            TemporalMetadata::Time(_) => Time::try_new(unit, nullability)?.erased(),
            TemporalMetadata::Date(_) => Date::try_new(unit, nullability)?.erased(),
        };
        Ok(DType::Extension(target))
    }

    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let input = args.get(0)?;
        let target = self.return_dtype(options, std::slice::from_ref(input.dtype()))?;
        let metadata = input
            .dtype()
            .as_extension_opt()
            .and_then(|ext| ext.metadata_opt::<AnyTemporal>())
            .vortex_expect("`return_dtype` above accepts only temporal inputs");
        let is_time = matches!(metadata, TemporalMetadata::Time(_));
        let source_unit = metadata.time_unit();
        let ChangeTimeUnitOptions { unit, rounding } = *options;
        let divisor = unit.nanos_per_unit() / source_unit.nanos_per_unit();

        // A time must stay within one day, and rounding up the last fraction of a day reaches
        // midnight, which is outside that range.
        if is_time {
            return map_counts(input, &target, ctx, |value| {
                check_time_of_day(value, source_unit)?;
                let result = rounding.divide(value, divisor)?;
                check_time_of_day(result, unit)?;
                Ok(result)
            });
        }

        map_counts(input, &target, ctx, |value| rounding.divide(value, divisor))
    }

    fn is_strict(&self, _options: &Self::Options) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::ExtensionArray;
    use crate::arrays::PrimitiveArray;
    use crate::dtype::Nullability;
    use crate::dtype::extension::ExtDTypeRef;
    use crate::expr::Expression;
    use crate::expr::change_time_unit;
    use crate::expr::proto::ExprSerializeProtoExt;
    use crate::expr::root;
    use crate::extension::datetime::TemporalRounding;
    use crate::extension::datetime::Time;
    use crate::extension::datetime::TimeUnit;
    use crate::extension::datetime::Timestamp;
    use crate::test_harness::ExpectedOutcome;
    use crate::test_harness::assert_execution_paths;

    const NON_NULLABLE: Nullability = Nullability::NonNullable;

    #[rstest]
    #[case::toward_zero(TemporalRounding::TowardZero, [Some(0), Some(0), None])]
    #[case::floor(TemporalRounding::Floor, [Some(-1), Some(0), None])]
    #[case::nearest(TemporalRounding::NearestTiesAwayFromZero, [Some(-1), Some(1), None])]
    fn test_change_timestamp_unit_rounding(
        #[case] mode: TemporalRounding,
        #[case] values: [Option<i64>; 3],
    ) -> VortexResult<()> {
        let source = Timestamp::new(TimeUnit::Nanoseconds, Nullability::Nullable).erased();
        let target = Timestamp::new(TimeUnit::Milliseconds, Nullability::Nullable).erased();
        let input = ExtensionArray::try_new(
            source,
            PrimitiveArray::from_option_iter([Some(-750_000i64), Some(500_000), None]).into_array(),
        )?
        .into_array();
        let expected = ExtensionArray::try_new(
            target,
            PrimitiveArray::from_option_iter(values).into_array(),
        )?;
        let session = VortexSession::empty();
        let mut ctx = session.create_execution_ctx();
        let operation = change_time_unit(root(), TimeUnit::Milliseconds, mode);
        let restored = Expression::from_proto(&operation.serialize_proto()?, &session)?;
        assert_execution_paths(
            &[input],
            |inputs| inputs[0].clone().apply(&restored),
            ExpectedOutcome::Array(expected.into_array()),
            &mut ctx,
        );
        Ok(())
    }

    #[rstest]
    #[case::time_rounds_up_to_midnight(
        Time::new(TimeUnit::Milliseconds, NON_NULLABLE).erased(),
        buffer![86_399_500i32].into_array(),
        TimeUnit::Seconds,
        TemporalRounding::NearestTiesAwayFromZero,
    )]
    #[case::exact_with_remainder(
        Timestamp::new(TimeUnit::Nanoseconds, NON_NULLABLE).erased(),
        buffer![1_500_000i64].into_array(),
        TimeUnit::Milliseconds,
        TemporalRounding::Exact,
    )]
    fn test_change_time_unit_errors(
        #[case] source: ExtDTypeRef,
        #[case] storage: ArrayRef,
        #[case] unit: TimeUnit,
        #[case] rounding: TemporalRounding,
    ) -> VortexResult<()> {
        let input = ExtensionArray::try_new(source, storage)?;
        let operation = change_time_unit(root(), unit, rounding);
        assert_execution_paths(
            &[input.into_array()],
            |inputs| inputs[0].clone().apply(&operation),
            ExpectedOutcome::Error,
            &mut VortexSession::empty().create_execution_ctx(),
        );
        Ok(())
    }
}
