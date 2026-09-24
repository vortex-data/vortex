// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Built-in constant detection and encoding.
//!
//! Constant arrays are not compressed through a pluggable [`Scheme`]: the compressor always
//! detects constant leaf arrays itself, before evaluating any registered scheme. Detection is
//! skipped while compressing samples, since a constant sample does not imply that the full array
//! is constant.
//!
//! [`Scheme`]: crate::scheme::Scheme

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::aggregate_fn::fns::is_constant::is_constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Extension;
use vortex_array::arrays::MaskedArray;
use vortex_array::arrays::VarBinView;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::varbinview::Ref;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_decimal_value_type;
use vortex_array::match_each_native_ptype;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::scheme::SchemeId;
use crate::trace;

/// Synthetic scheme ID reported in traces when the compressor's built-in constant encoding wins.
const CONSTANT_SCHEME_ID: SchemeId = SchemeId {
    name: "vortex.compressor.constant",
};

/// Checks for equal valid values without generating compression statistics.
///
/// The caller has already handled empty and all-null arrays. Unsupported nullable storage types
/// conservatively return false. All-valid arrays reuse the ordinary constant check.
pub(crate) fn is_constant_for_compression(
    array: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    if let Some(extension) = array.as_opt::<Extension>() {
        // Compare storage directly: extension min/max can hide NaNs in floating-point storage.
        let storage = extension
            .storage_array()
            .clone()
            .execute::<Canonical>(ctx)?
            .into_array();
        return is_constant_for_compression(&storage, ctx);
    }

    let mask = array.validity()?.execute_mask(array.len(), ctx)?;
    let Some(first) = mask.first() else {
        return Ok(!array.is_empty());
    };

    if mask.all_true() {
        return is_constant(array, ctx);
    }
    if let Some(strings) = array.as_opt::<VarBinView>() {
        return Ok(is_constant_varbinview(strings, &mask, first));
    }
    let Mask::Values(valid) = &mask else {
        unreachable!("all-valid and all-null masks handled above");
    };

    Ok(match array.clone().execute::<Canonical>(ctx)? {
        Canonical::Primitive(array) => {
            match_each_native_ptype!(array.ptype(), |T| {
                let values = array.as_slice::<T>();
                let first = values[first];
                all_valid_match(&mask, |i| values[i].is_eq(first))
            })
        }
        Canonical::Bool(array) => {
            let values = array.bit_buffer_view();
            let expected = if values.value(first) { u64::MAX } else { 0 };
            values
                .chunks()
                .iter_padded()
                .zip(valid.bit_buffer().chunks().iter_padded())
                .all(|(values, valid)| ((values ^ expected) & valid) == 0)
        }
        Canonical::Decimal(array) => {
            match_each_decimal_value_type!(array.values_type(), |T| {
                let values = array.buffer::<T>();
                let first = values[first];
                all_valid_match(&mask, |i| values[i] == first)
            })
        }
        _ => false,
    })
}

fn all_valid_match(mask: &Mask, mut matches: impl FnMut(usize) -> bool) -> bool {
    match mask {
        Mask::AllTrue(len) => (0..*len).all(matches),
        Mask::AllFalse(_) => true,
        Mask::Values(valid) => valid
            .bit_buffer()
            .try_for_each_set_index(|i| if matches(i) { Ok(()) } else { Err(()) })
            .is_ok(),
    }
}

fn is_constant_varbinview(array: ArrayView<'_, VarBinView>, mask: &Mask, first: usize) -> bool {
    let views = array.views();
    let first = &views[first];
    if first.is_inlined() {
        return all_valid_match(mask, |i| {
            let view = &views[i];
            view == first
                || (view.is_inlined() && view.as_inlined().value() == first.as_inlined().value())
        });
    }

    let first = first.as_view();
    let buffers = array
        .data_buffers()
        .iter()
        .map(|buffer| buffer.as_host())
        .collect::<Vec<_>>();
    let bytes = |view: &Ref| &buffers[view.buffer_index as usize][view.as_range()];
    let first_bytes = bytes(first);
    all_valid_match(mask, |i| {
        let view = &views[i];
        if view.len() != first.size {
            return false;
        }
        let view = view.as_view();
        view.prefix == first.prefix
            && ((view.buffer_index == first.buffer_index && view.offset == first.offset)
                || bytes(view) == first_bytes)
    })
}

/// Encodes an array whose valid values are all equal as the winning "scheme", recording the result
/// in the trace.
///
/// Returns the original array if the constant encoding is not smaller.
///
/// # Errors
///
/// Returns an error if [`compress_constant`] fails.
pub(crate) fn compress_as_constant(
    array: ArrayRef,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let before_nbytes = array.nbytes();
    let _winner_span = trace::winner_compress_span(CONSTANT_SCHEME_ID, before_nbytes).entered();
    let compressed = compress_constant(&array, exec_ctx)?;

    let after_nbytes = compressed.nbytes();
    let actual_ratio = (after_nbytes != 0).then(|| before_nbytes as f64 / after_nbytes as f64);
    let accepted = after_nbytes < before_nbytes;
    trace::record_winner_compress_result(after_nbytes, None, actual_ratio, accepted);

    Ok(if accepted { compressed } else { array })
}

/// Encodes an array whose valid values are all equal.
///
/// Returns a [`ConstantArray`], wrapped in a [`MaskedArray`] when the array has some nulls, or a
/// null [`ConstantArray`] when the array is all-null.
///
/// # Errors
///
/// Returns an error if computing validity or extracting the constant scalar fails.
fn compress_constant(source: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    let validity = source.validity()?;
    let mask = validity.execute_mask(source.len(), ctx)?;

    let Some(first_valid) = mask.first() else {
        return Ok(
            ConstantArray::new(Scalar::null(source.dtype().clone()), source.len()).into_array(),
        );
    };

    let scalar = source.execute_scalar(first_valid, ctx)?;
    let const_arr = ConstantArray::new(scalar, source.len()).into_array();

    if mask.all_true() {
        Ok(const_arr)
    } else {
        Ok(MaskedArray::try_new(const_arr, validity)?.into_array())
    }
}

#[cfg(test)]
mod tests;
