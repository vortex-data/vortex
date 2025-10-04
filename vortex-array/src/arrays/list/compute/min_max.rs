// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_scalar::ListScalar;

use crate::arrays::{ListArray, ListVTable};
use crate::compute::{MinMaxKernel, MinMaxKernelAdapter, MinMaxResult};
use crate::register_kernel;
use crate::vtable::OperationsVTable;

impl MinMaxKernel for ListVTable {
    fn min_max(&self, array: &ListArray) -> VortexResult<Option<MinMaxResult>> {
        // Find the lexicographically minimum and maximum lists.
        let scalars: Vec<_> = (0..array.len())
            .filter_map(|i| {
                if array.is_valid(i) {
                    Some(ListVTable::scalar_at(array, i))
                } else {
                    None
                }
            })
            .collect();

        if scalars.is_empty() {
            return Ok(None);
        }

        let minmax = scalars.iter().minmax_by(|a, b| {
            let a_list = ListScalar::try_from(*a).ok();
            let b_list = ListScalar::try_from(*b).ok();
            a_list
                .partial_cmp(&b_list)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(match minmax {
            itertools::MinMaxResult::NoElements => None,
            itertools::MinMaxResult::OneElement(scalar) => Some(MinMaxResult {
                min: (*scalar).clone(),
                max: (*scalar).clone(),
            }),
            itertools::MinMaxResult::MinMax(min, max) => Some(MinMaxResult {
                min: (*min).clone(),
                max: (*max).clone(),
            }),
        })
    }
}

register_kernel!(MinMaxKernelAdapter(ListVTable).lift());
