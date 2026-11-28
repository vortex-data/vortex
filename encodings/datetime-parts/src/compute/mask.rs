// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::scalar_fn::builtin::mask::MaskReduce;
use vortex_error::VortexResult;

use crate::DateTimeParts;
use crate::array::DateTimePartsArraySlotsExt;

impl MaskReduce for DateTimeParts {
    fn mask(array: ArrayView<'_, Self>, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        let masked_days = array.days().clone().mask(mask.clone())?;
        Ok(Some(
            DateTimeParts::try_new(
                array.dtype().as_nullable(),
                masked_days,
                array.seconds().clone(),
                array.subseconds().clone(),
            )?
            .into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::TemporalArray;
    use vortex_array::compute::conformance::mask::test_mask_conformance;
    use vortex_array::extension::datetime::TimeUnit;
    use vortex_array::optimizer::ArrayOptimizer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::DateTimeParts;
    use crate::DateTimePartsArray;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

    fn dtp_from_temporal(temporal: TemporalArray) -> DateTimePartsArray {
        DateTimeParts::try_from_temporal(temporal, &mut SESSION.create_execution_ctx()).unwrap()
    }

    #[rstest]
    #[case::seconds(dtp_from_temporal(TemporalArray::new_timestamp(
        buffer![0i64, 86_400, 172_800, 259_200, 345_600].into_array(),
        TimeUnit::Seconds,
        Some("UTC".into()),
    )))]
    #[case::milliseconds(dtp_from_temporal(TemporalArray::new_timestamp(
        buffer![123_456_789i64, 234_567_890, 345_678_901, 456_789_012, 567_890_123].into_array(),
        TimeUnit::Milliseconds,
        Some("UTC".into()),
    )))]
    #[case::microseconds(dtp_from_temporal(TemporalArray::new_timestamp(
        buffer![0i64, 86_400_000_000, 172_800_000_500, 259_200_123_456, 345_600_000_000].into_array(),
        TimeUnit::Microseconds,
        None,
    )))]
    #[case::nanoseconds(dtp_from_temporal(TemporalArray::new_timestamp(
        buffer![0i64, 86_400_000_000_000, 172_800_000_000_001].into_array(),
        TimeUnit::Nanoseconds,
        Some("UTC".into()),
    )))]
    #[case::nullable(dtp_from_temporal(TemporalArray::new_timestamp(
        PrimitiveArray::from_option_iter([Some(0i64), None, Some(86_400), Some(172_800), None]).into_array(),
        TimeUnit::Seconds,
        Some("UTC".into()),
    )))]
    #[case::single(dtp_from_temporal(TemporalArray::new_timestamp(
        buffer![1_234_567_890i64].into_array(),
        TimeUnit::Seconds,
        Some("UTC".into()),
    )))]
    #[case::large(dtp_from_temporal(TemporalArray::new_timestamp(
        PrimitiveArray::from_iter((0..1500i64).map(|i| i * 86_400 + i % 100)).into_array(),
        TimeUnit::Seconds,
        Some("UTC".into()),
    )))]
    fn test_mask_datetime_parts_conformance(#[case] array: DateTimePartsArray) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = array.into_array();
        test_mask_conformance(&array, &mut ctx);
        if array.len() > 2 {
            let sliced = array.slice(1..array.len() - 1)?.optimize_ctx(&SESSION)?;
            assert!(sliced.is::<DateTimeParts>());
            test_mask_conformance(&sliced, &mut ctx);
        }
        Ok(())
    }
}
