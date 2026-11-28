// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Extract the local calendar date of a timestamp.
//!
//! A timestamp with a time zone uses the offset of that zone at its instant, and a timestamp
//! without a time zone already counts local time.

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::dtype::DType;
use vortex_array::extension::datetime::Date;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::extension::datetime::Timestamp;
use vortex_array::extension::datetime::local_count;
use vortex_array::extension::datetime::map_counts;
use vortex_array::extension::datetime::resolve_time_zone;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

/// Extract the local calendar date of a [`Timestamp`] as a [`Date`] in days.
///
/// See [`timestamp_to_date`](crate::expr::timestamp_to_date) for the contract.
#[derive(Clone)]
pub struct TimestampToDate;

impl ScalarFnVTable for TimestampToDate {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.timestamp_to_date");
        *ID
    }

    fn serialize(&self, _options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(
        &self,
        _metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        Ok(EmptyOptions)
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _options: &Self::Options, _child_idx: usize) -> ChildName {
        ChildName::from("input")
    }

    fn return_dtype(&self, _options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        let input = &arg_dtypes[0];
        vortex_ensure!(
            input
                .as_extension_opt()
                .is_some_and(|ext| ext.is::<Timestamp>()),
            "Expected a timestamp, got {input}"
        );

        Ok(DType::Extension(
            Date::new(TimeUnit::Days, input.nullability()).erased(),
        ))
    }

    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let input = args.get(0)?;
        let target = self.return_dtype(options, std::slice::from_ref(input.dtype()))?;
        let timestamp = input
            .dtype()
            .as_extension_opt()
            .and_then(|ext| ext.metadata_opt::<Timestamp>())
            .vortex_expect("`return_dtype` above accepts only timestamps");
        let unit = timestamp.unit;
        let zone = timestamp
            .tz
            .as_deref()
            .map(|zone| resolve_time_zone(Some(zone)))
            .transpose()?;
        let units_per_day = i128::from(unit.units_per_day());

        map_counts(input, &target, ctx, |value| {
            let local = local_count(value, unit, zone.as_ref())?;
            Ok(i64::try_from(local.div_euclid(units_per_day))?)
        })
    }

    fn is_strict(&self, _options: &Self::Options) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ExtensionArray;
    use vortex_array::dtype::Nullability;
    use vortex_array::extension::datetime::Date;
    use vortex_array::extension::datetime::TimeUnit;
    use vortex_array::extension::datetime::Timestamp;
    use vortex_array::test_harness::ExpectedOutcome;
    use vortex_array::test_harness::assert_execution_paths;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_expr::ArrayExprExt;
    use vortex_expr::Expression;
    use vortex_expr::proto::ExprSerializeProtoExt;
    use vortex_expr::root;
    use vortex_session::VortexSession;

    use crate::expr::timestamp_to_date;

    #[test]
    fn test_timestamp_to_date_uses_zone_offset() -> VortexResult<()> {
        let input = ExtensionArray::try_new(
            Timestamp::new_with_tz(
                TimeUnit::Seconds,
                Some("America/New_York".into()),
                Nullability::NonNullable,
            )
            .erased(),
            buffer![
                0i64,          // The epoch, which is 1969-12-31 19:00:00 EST in New York.
                1_710_053_999, // 2024-03-10 01:59:59 EST, the last second before DST starts.
                1_710_054_000, // 2024-03-10 03:00:00 EDT, after the offset moves from -5h to -4h.
            ]
            .into_array(),
        )?;
        let expected = ExtensionArray::try_new(
            Date::new(TimeUnit::Days, Nullability::NonNullable).erased(),
            buffer![-1i32, 19_792, 19_792].into_array(),
        )?;
        let session = VortexSession::empty();
        crate::initialize(&session);
        let operation = timestamp_to_date(root());
        let restored = Expression::from_proto(&operation.serialize_proto()?, &session)?;

        assert_execution_paths(
            &[input.into_array()],
            |inputs| inputs[0].clone().apply(&restored),
            ExpectedOutcome::Array(expected.into_array()),
            &mut session.create_execution_ctx(),
        );
        Ok(())
    }
}
