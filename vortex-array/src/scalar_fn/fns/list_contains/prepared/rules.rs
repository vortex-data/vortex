// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use super::PreparedSetArray;
use super::PreparedSetData;
use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Constant;
use crate::arrays::ScalarFn;
use crate::arrays::scalar_fn::ExactScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayExt;
use crate::arrays::scalar_fn::ScalarFnArrayView;
use crate::dtype::DType;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::optimizer::rules::ArrayReduceRule;
use crate::scalar_fn::fns::list_contains::ListContains;
use crate::scalar_fn::fns::list_contains::PreparedSetLiteral;

/// Prepares the constant list of a [`ListContains`] as a set before any push-down runs.
///
/// A rule that pushes the function into the needle, for example into each chunk of a chunked
/// array, then resizes a [`PreparedSetArray`] that shares the one set, rather than a constant that
/// every chunk would prepare again. The set is built on the first probe, so this costs nothing
/// until then.
#[derive(Debug)]
pub(crate) struct PrepareConstantListRule;

impl ArrayParentReduceRule<Constant> for PrepareConstantListRule {
    type Parent = ExactScalarFn<ListContains>;

    fn reduce_parent(
        &self,
        list: ArrayView<'_, Constant>,
        parent: ScalarFnArrayView<'_, ListContains>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        if child_idx != 0 {
            return Ok(None);
        }
        let scalar = list.scalar();
        let DType::List(element_dtype, _) = scalar.dtype() else {
            return Ok(None);
        };
        // A null list has no set, and a needle of another dtype is an error for execution to
        // report.
        let parent_array = parent
            .as_opt::<ScalarFn>()
            .vortex_expect("ExactScalarFn matcher confirmed ScalarFnArray");
        let needle = parent_array.get_child(1);
        if scalar.is_null() || !element_dtype.eq_ignore_nullability(needle.dtype()) {
            return Ok(None);
        }

        let set = PreparedSetArray::new(PreparedSetData::try_new(scalar.clone())?, list.len());
        // SAFETY: every row of the prepared set holds the constant's list.
        Ok(Some(unsafe {
            (*parent).clone().with_slot(0, set.into_array())?
        }))
    }
}

/// Replaces a [`PreparedSetLiteral`] applied to an array with the [`PreparedSetArray`] its
/// execution gives, so that push-down rules see the prepared set without executing anything.
#[derive(Debug)]
pub(crate) struct PreparedSetLiteralRule;

impl ArrayReduceRule<ScalarFn> for PreparedSetLiteralRule {
    fn reduce(&self, array: ArrayView<'_, ScalarFn>) -> VortexResult<Option<ArrayRef>> {
        Ok(array
            .scalar_fn()
            .as_opt::<PreparedSetLiteral>()
            .map(|set| PreparedSetArray::new(set.clone(), array.len()).into_array()))
    }
}
