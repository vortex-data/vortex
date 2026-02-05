// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::Array;
use crate::ArrayRef;
use crate::IntoArray;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::expr::stats::StatsProvider;
use crate::expr::stats::StatsProviderExt;
use crate::stats::StatsSet;

/// Creates a new array using the elements from the input `array` indexed by `indices`.
///
/// For example, if we have an `array` `[1, 2, 3, 4, 5]` and `indices` `[4, 2]`, the resulting
/// array would be `[5, 3]`.
///
/// The output array will have the same length as the `indices` array.
pub fn take(array: &dyn Array, indices: &dyn Array) -> VortexResult<ArrayRef> {
    array
        .take(indices.to_array())?
        .to_canonical()
        .map(|c| c.into_array())
}

pub(crate) fn propagate_take_stats(
    source: &dyn Array,
    target: &dyn Array,
    indices: &dyn Array,
) -> VortexResult<()> {
    target.statistics().with_mut_typed_stats_set(|mut st| {
        if indices.all_valid().unwrap_or(false) {
            let is_constant = source.statistics().get_as::<bool>(Stat::IsConstant);
            if is_constant == Some(Precision::Exact(true)) {
                // Any combination of elements from a constant array is still const
                st.set(Stat::IsConstant, Precision::exact(true));
            }
        }
        let inexact_min_max = [Stat::Min, Stat::Max]
            .into_iter()
            .filter_map(|stat| {
                source
                    .statistics()
                    .get(stat)
                    .map(|v| (stat, v.map(|s| s.into_value()).into_inexact()))
            })
            .collect::<Vec<_>>();
        st.combine_sets(
            &(unsafe { StatsSet::new_unchecked(inexact_min_max) }).as_typed_ref(source.dtype()),
        )
    })
}
