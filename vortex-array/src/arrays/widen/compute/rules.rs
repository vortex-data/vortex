// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Physical rewrites for a [`Widen`] and its parents.

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
use crate::arrays::widen::WidenSlots;
use crate::dtype::DType;
use crate::optimizer::kernels::ArrayKernels;
use crate::optimizer::kernels::ArrayKernelsExt;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::scalar_fn::builtin::cast::Cast;
use crate::scalar_fn::builtin::cast::CastReduceAdaptor;
use crate::scalar_fn::builtin::fill_null::FillNull;
use crate::scalar_fn::builtin::fill_null::FillNullReduceAdaptor;
use crate::scalar_fn::builtin::mask::Mask as MaskFn;
use crate::scalar_fn::builtin::mask::MaskReduceAdaptor;

pub(crate) fn register_parent_rules(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_reduce_parent_rule(Cast, Widen, const { &CastReduceAdaptor(Widen) });
    kernels.register_reduce_parent_rule(FillNull, Widen, const { &FillNullReduceAdaptor(Widen) });
    kernels.register_reduce_parent_rule(Filter, Widen, const { &FilterReduceAdaptor(Widen) });
    kernels.register_reduce_parent_rule(MaskFn, Widen, const { &MaskReduceAdaptor(Widen) });
    kernels.register_reduce_parent_rule(Slice, Widen, const { &SliceReduceAdaptor(Widen) });
}

/// Let the writer lower stored values before expanding dictionary codes or runs.
pub(crate) fn register_write_parent_rules(kernels: &ArrayKernels) {
    kernels.register_reduce_parent_rule(Widen, Dict, const { &WidenDictionaryValues });
    kernels.register_reduce_parent_rule(Widen, RunEnd, const { &WidenRunEndValues });
}

#[derive(Debug)]
struct WidenDictionaryValues;

impl ArrayParentReduceRule<Dict> for WidenDictionaryValues {
    type Parent = Widen;

    fn reduce_parent(
        &self,
        dict: ArrayView<'_, Dict>,
        widen: ArrayView<'_, Widen>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        if child_idx != WidenSlots::VALUES || !matches!(widen.dtype(), DType::Decimal(..)) {
            return Ok(None);
        }

        // Nullable codes can make the dictionary nullable even when its values are non-nullable.
        let values_dtype = widen
            .dtype()
            .with_nullability(dict.values().dtype().nullability());
        let values = WidenArray::try_new(dict.values().clone(), values_dtype)?.into_array();
        let slots = DictSlots {
            codes: dict.codes().clone(),
            values,
        };
        // The codes and value count stay the same, so the reference guarantee still holds.
        let dict = Array::try_from_parts(ArrayParts::new(
            Dict,
            widen.dtype().clone(),
            widen.len(),
            dict.data().clone(),
            slots.into_slots(),
        ))?;
        Ok(Some(dict.into_array()))
    }
}

#[derive(Debug)]
struct WidenRunEndValues;

impl ArrayParentReduceRule<RunEnd> for WidenRunEndValues {
    type Parent = Widen;

    fn reduce_parent(
        &self,
        run_end: ArrayView<'_, RunEnd>,
        widen: ArrayView<'_, Widen>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        if child_idx != WidenSlots::VALUES || !matches!(widen.dtype(), DType::Decimal(..)) {
            return Ok(None);
        }

        let values = WidenArray::try_new(run_end.values().clone(), widen.dtype().clone())?
            .into_array();
        let slots = RunEndSlots {
            ends: run_end.ends().clone(),
            values,
        };
        // The view's offset is relative to the unchanged run ends.
        let run_end = Array::try_from_parts(ArrayParts::new(
            RunEnd,
            widen.dtype().clone(),
            widen.len(),
            run_end.data().clone(),
            slots.into_slots(),
        ))?;
        Ok(Some(run_end.into_array()))
    }
}
