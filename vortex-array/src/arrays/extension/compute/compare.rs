// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::ConstantArray;
use crate::arrays::Extension;
use crate::arrays::extension::ExtensionArrayExt;
use crate::builtins::ArrayBuiltins;
use crate::scalar_fn::fns::binary::CompareKernel;
use crate::scalar_fn::fns::operators::CompareOperator;
use crate::scalar_fn::fns::operators::Operator;

impl CompareKernel for Extension {
    fn compare(
        lhs: ArrayView<'_, Extension>,
        rhs: &ArrayRef,
        operator: CompareOperator,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // If the RHS is a constant, we can extract the storage scalar.
        if let Some(const_ext) = rhs.as_constant() {
            let storage_scalar = const_ext.as_extension().to_storage_scalar();
            return lhs
                .storage_array()
                .clone()
                .binary(
                    ConstantArray::new(storage_scalar, lhs.len()).into_array(),
                    Operator::from(operator),
                )
                .map(Some);
        }

        // If the RHS is an extension array matching ours, we can extract the storage.
        if let Some(rhs_ext) = rhs.as_opt::<Extension>() {
            return lhs
                .storage_array()
                .clone()
                .binary(rhs_ext.storage_array().clone(), Operator::from(operator))
                .map(Some);
        }

        // Otherwise, we need the RHS to handle this comparison.
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::Canonical;
    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::arrays::BoolArray;
    use crate::arrays::ExtensionArray;
    use crate::arrays::datetime::TemporalArray;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::Nullability;
    use crate::extension::datetime::Date;
    use crate::extension::datetime::TimeUnit;
    use crate::scalar_fn::fns::operators::Operator;

    /// Timestamps store UTC instants regardless of timezone (Arrow semantics).
    /// The raw i64 is always relative to epoch, timezone is display metadata.
    /// Same raw values + same time unit = same instant, so result should be all-true.
    #[test]
    fn compare_timestamps_different_timezones() -> VortexResult<()> {
        let values = buffer![1_000_000i64, 2_000_000, 3_000_000].into_array();

        let ts_utc =
            TemporalArray::new_timestamp(values.clone(), TimeUnit::Seconds, Some(Arc::from("UTC")))
                .into_array();

        let ts_ny = TemporalArray::new_timestamp(
            values,
            TimeUnit::Seconds,
            Some(Arc::from("America/New_York")),
        )
        .into_array();

        let mut ctx = LEGACY_SESSION.create_execution_ctx();

        let result = ts_utc
            .binary(ts_ny, Operator::Eq)?
            .execute::<Canonical>(&mut ctx)?
            .into_array();

        assert_arrays_eq!(result, BoolArray::from_iter([true, true, true]));

        Ok(())
    }

    /// BUG: Comparing Timestamp(seconds) vs Date(milliseconds) — different extension
    /// type IDs — should yield all-false (or error), but the kernel silently compares
    /// raw i64 storage and returns all-true.
    #[test]
    fn compare_timestamp_vs_date() -> VortexResult<()> {
        let ts_values = buffer![86_400i64, 172_800, 259_200].into_array();
        let ts_array =
            TemporalArray::new_timestamp(ts_values, TimeUnit::Seconds, None).into_array();

        let date_values = buffer![86_400i64, 172_800, 259_200].into_array();
        let date_ext_dtype = Date::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased();
        let date_array = ExtensionArray::new(date_ext_dtype, date_values).into_array();

        let mut ctx = LEGACY_SESSION.create_execution_ctx();

        let result = ts_array
            .binary(date_array, Operator::Eq)?
            .execute::<Canonical>(&mut ctx)?
            .into_array();

        // Different extension types should never be equal.
        assert_arrays_eq!(result, BoolArray::from_iter([false, false, false]));

        Ok(())
    }

    /// BUG: Comparing Timestamp(milliseconds) vs Timestamp(seconds) — same extension
    /// type ID but different metadata — should yield all-false (or error), but the
    /// kernel silently compares raw i64 storage. 1000ms != 1000s.
    #[test]
    fn compare_timestamps_different_units() -> VortexResult<()> {
        let millis = buffer![1000i64, 2000, 3000].into_array();
        let seconds = buffer![1000i64, 2000, 3000].into_array();

        let ts_millis =
            TemporalArray::new_timestamp(millis, TimeUnit::Milliseconds, None).into_array();
        let ts_seconds =
            TemporalArray::new_timestamp(seconds, TimeUnit::Seconds, None).into_array();

        let mut ctx = LEGACY_SESSION.create_execution_ctx();

        let result = ts_millis
            .binary(ts_seconds, Operator::Eq)?
            .execute::<Canonical>(&mut ctx)?
            .into_array();

        // 1000ms != 1000s, should be all-false.
        assert_arrays_eq!(result, BoolArray::from_iter([false, false, false]));

        Ok(())
    }
}
