// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Physical rewrites for a [`Widen`] and its parents.

use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::Array;
use crate::array::ArrayParts;
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
use crate::dtype::DType;
use crate::optimizer::kernels::ArrayKernelsExt;
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

/// Push decimal widening into the values of a dictionary or run-end array.
///
/// Write normalization can then lower the stored values without expanding the logical rows.
pub(crate) fn push_decimal_widen_into_values(array: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
    let Some(widen) = array.as_opt::<Widen>() else {
        return Ok(None);
    };

    if !matches!(widen.dtype(), DType::Decimal(..)) {
        return Ok(None);
    }

    if let Some(dict) = widen.values().as_opt::<Dict>() {
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
        return Ok(Some(dict.into_array()));
    }

    if let Some(run_end) = widen.values().as_opt::<RunEnd>() {
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
        return Ok(Some(run_end.into_array()));
    }

    Ok(None)
}
