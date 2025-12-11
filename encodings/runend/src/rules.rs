// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::AnyScalarFn;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::ConstantVTable;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::optimizer::rules::ArrayParentReduceRule;
use vortex_array::optimizer::rules::Exact;
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::RunEndArray;
use crate::RunEndVTable;

/// A rule to push down scalar functions through run-end encoding into the values array.
///
/// This only works if all other children of the scalar function array are constants.
#[derive(Debug)]
pub(crate) struct RunEndScalarFnRule;

impl ArrayParentReduceRule<Exact<RunEndVTable>, AnyScalarFn> for RunEndScalarFnRule {
    fn child(&self) -> Exact<RunEndVTable> {
        Exact::from(&RunEndVTable)
    }

    fn parent(&self) -> AnyScalarFn {
        AnyScalarFn
    }

    fn reduce_parent(
        &self,
        run_end: &RunEndArray,
        parent: &ScalarFnArray,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        for (idx, child) in parent.children().iter().enumerate() {
            if idx == child_idx {
                // Skip ourselves
                continue;
            }

            if !child.is::<ConstantVTable>() {
                // We can only push down if all other children are constants
                return Ok(None);
            }
        }

        // TODO(ngates): relax this constraint and implement run-end decoding for all vector types.
        if !matches!(parent.dtype(), DType::Bool(_) | DType::Primitive(..)) {
            return Ok(None);
        }

        let values_len = run_end.values().len();
        let mut new_children: Vec<ArrayRef> = parent.children().to_vec();
        for (idx, child) in new_children.iter_mut().enumerate() {
            if idx == child_idx {
                // Replace ourselves with run end values
                *child = run_end.values().clone();
                continue;
            }

            // Replace other children with their constant scalar value with length adjusted
            // to the length of the run end values.
            let constant = child.as_::<ConstantVTable>();
            *child = ConstantArray::new(constant.scalar().clone(), values_len).into_array();
        }

        let new_values =
            ScalarFnArray::try_new(parent.scalar_fn().clone(), new_children, values_len)?
                .into_array();

        Ok(Some(
            RunEndArray::try_new_offset_length(
                run_end.ends().clone(),
                new_values,
                run_end.offset(),
                run_end.len(),
            )?
            .into_array(),
        ))
    }
}
