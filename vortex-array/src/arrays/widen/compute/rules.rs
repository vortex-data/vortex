// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Optimizer rules for a [`Widen`] and its parents.
//!
//! Structural encodings stay visible at the root so their codes or run ends can be reused.

use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::Array;
use crate::array::ArrayParts;
use crate::array::ArrayView;
use crate::arrays::Dict;
use crate::arrays::Filter;
use crate::arrays::RunEnd;
use crate::arrays::Slice;
use crate::arrays::Widen;
use crate::arrays::WidenArray;
use crate::arrays::dict::DictArraySlotsExt;
use crate::arrays::dict::DictSlots;
use crate::arrays::filter::FilterReduceAdaptor;
use crate::arrays::runend::RunEndArraySlotsExt;
use crate::arrays::runend::RunEndSlots;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::arrays::widen::WidenArraySlotsExt;
use crate::optimizer::kernels::ArrayKernelsExt;
use crate::optimizer::rules::ArrayReduceRule;
use crate::optimizer::rules::ReduceRuleSet;
use crate::scalar_fn::builtin::cast::Cast;
use crate::scalar_fn::builtin::cast::CastReduceAdaptor;
use crate::scalar_fn::builtin::fill_null::FillNull;
use crate::scalar_fn::builtin::fill_null::FillNullReduceAdaptor;
use crate::scalar_fn::builtin::mask::Mask as MaskFn;
use crate::scalar_fn::builtin::mask::MaskReduceAdaptor;

/// Move decimal widening from a structural array to its values.
pub(in crate::arrays::widen) const DECIMAL_STRUCTURAL_RULES: ReduceRuleSet<Widen> =
    ReduceRuleSet::new(&[&WidenDictionaryValues, &WidenRunEndValues]);

pub(crate) fn register_parent_rules(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_reduce_parent_rule(Cast, Widen, const { &CastReduceAdaptor(Widen) });
    kernels.register_reduce_parent_rule(FillNull, Widen, const { &FillNullReduceAdaptor(Widen) });
    kernels.register_reduce_parent_rule(Filter, Widen, const { &FilterReduceAdaptor(Widen) });
    kernels.register_reduce_parent_rule(MaskFn, Widen, const { &MaskReduceAdaptor(Widen) });
    kernels.register_reduce_parent_rule(Slice, Widen, const { &SliceReduceAdaptor(Widen) });
}

#[derive(Debug)]
struct WidenDictionaryValues;

impl ArrayReduceRule<Widen> for WidenDictionaryValues {
    fn reduce(&self, array: ArrayView<'_, Widen>) -> VortexResult<Option<ArrayRef>> {
        let Some(dict) = array.values().as_opt::<Dict>() else {
            return Ok(None);
        };

        // Nullable codes can make the dictionary nullable even when its values are non-nullable.
        let values_dtype = array
            .dtype()
            .with_nullability(dict.values().dtype().nullability());
        let values = WidenArray::try_new(dict.values().clone(), values_dtype)?.into_array();
        let slots = DictSlots {
            codes: dict.codes().clone(),
            values,
        };
        let rebuilt = Array::try_from_parts(ArrayParts::new(
            Dict,
            array.dtype().clone(),
            array.len(),
            dict.data().clone(),
            slots.into_slots(),
        ))?;

        Ok(Some(rebuilt.into_array()))
    }
}

#[derive(Debug)]
struct WidenRunEndValues;

impl ArrayReduceRule<Widen> for WidenRunEndValues {
    fn reduce(&self, array: ArrayView<'_, Widen>) -> VortexResult<Option<ArrayRef>> {
        let Some(run_end) = array.values().as_opt::<RunEnd>() else {
            return Ok(None);
        };

        let values = WidenArray::try_new(run_end.values().clone(), array.dtype().clone())?
            .into_array();
        let slots = RunEndSlots {
            ends: run_end.ends().clone(),
            values,
        };
        // The view's offset is relative to the unchanged run ends.
        let rebuilt = Array::try_from_parts(ArrayParts::new(
            RunEnd,
            array.dtype().clone(),
            array.len(),
            run_end.data().clone(),
            slots.into_slots(),
        ))?;

        Ok(Some(rebuilt.into_array()))
    }
}
