// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! FSST compression entry points.
//!
//! [`fsst_compress`] and [`fsst_train_compressor`] take an [`ArrayRef`] and dispatch
//! on the input encoding ([`VarBinView`] or [`VarBin`]). Callers don't need to know
//! which string encoding they hold.

use std::sync::Arc;

use fsst::Compressor;
use num_traits::AsPrimitive;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBin;
use vortex_array::arrays::VarBinView;
use vortex_array::arrays::varbin::VarBinArrayExt;
use vortex_array::arrays::varbin::builder::VarBinBuilder;
use vortex_array::arrays::varbinview::BinaryView;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::IntegerPType;
use vortex_array::match_each_integer_ptype;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use crate::FSST;
use crate::FSSTArray;

/// FSST worst case: every input byte expands to an escape + literal (2x).
const FSST_PER_BYTE_OVERHEAD: usize = 2;

/// Starting capacity for the per-row `compress_into` scratch buffer; grown monotonically.
const DEFAULT_BUFFER_LEN: usize = 1024 * 1024;

/// Compress a string array using FSST.
///
/// Accepts any [`VarBinView`] or [`VarBin`]-encoded array; other encodings error.
pub fn fsst_compress(
    array: &ArrayRef,
    compressor: &Compressor,
    ctx: &mut ExecutionCtx,
) -> VortexResult<FSSTArray> {
    if let Some(view) = array.as_opt::<VarBinView>() {
        compress_varbinview(view, compressor, ctx)
    } else if let Some(varbin) = array.as_opt::<VarBin>() {
        compress_varbin_array(varbin, compressor, ctx)
    } else {
        vortex_bail!(
            "fsst_compress requires VarBinView or VarBin encoding, got {}",
            array.encoding_id()
        )
    }
}

/// Train an FSST [`Compressor`] from a string array's non-null rows.
///
/// Accepts any [`VarBinView`] or [`VarBin`]-encoded array; other encodings error.
pub fn fsst_train_compressor(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Compressor> {
    if let Some(view) = array.as_opt::<VarBinView>() {
        train_varbinview(view, ctx)
    } else if let Some(varbin) = array.as_opt::<VarBin>() {
        train_varbin_array(varbin, ctx)
    } else {
        vortex_bail!(
            "fsst_train_compressor requires VarBinView or VarBin encoding, got {}",
            array.encoding_id()
        )
    }
}

fn compress_varbinview(
    strings: ArrayView<VarBinView>,
    compressor: &Compressor,
    ctx: &mut ExecutionCtx,
) -> VortexResult<FSSTArray> {
    let mask = strings.validity()?.execute_mask(strings.len(), ctx)?;
    let views = strings.views();

    let total_input_bytes = match mask.bit_buffer() {
        AllOr::All => views.iter().map(|v| v.len() as usize).sum(),
        AllOr::None => 0,
        AllOr::Some(bits) => views
            .iter()
            .zip(bits.iter())
            .filter(|&(_, b)| b)
            .map(|(v, _)| v.len() as usize)
            .sum(),
    };

    if fsst_output_fits_in_i32_offsets(total_input_bytes) {
        compress_views::<i32>(strings, &mask, compressor, ctx)
    } else {
        compress_views::<i64>(strings, &mask, compressor, ctx)
    }
}

fn compress_varbin_array(
    strings: ArrayView<VarBin>,
    compressor: &Compressor,
    ctx: &mut ExecutionCtx,
) -> VortexResult<FSSTArray> {
    let mask = strings.validity()?.execute_mask(strings.len(), ctx)?;
    let offsets = strings.offsets().clone().execute::<PrimitiveArray>(ctx)?;
    let total_input_bytes = match_each_integer_ptype!(offsets.ptype(), |O| {
        let off = offsets.as_slice::<O>();
        let first: usize = off[0].as_();
        let last: usize = off[off.len() - 1].as_();
        last - first
    });

    if fsst_output_fits_in_i32_offsets(total_input_bytes) {
        compress_varbin::<i32>(strings, &offsets, &mask, compressor, ctx)
    } else {
        compress_varbin::<i64>(strings, &offsets, &mask, compressor, ctx)
    }
}

fn train_varbinview(
    strings: ArrayView<VarBinView>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Compressor> {
    let mask = strings.validity()?.execute_mask(strings.len(), ctx)?;
    let views = strings.views();
    let buffers = strings.data_buffers();
    let mut lines: Vec<&[u8]> = Vec::with_capacity(mask.true_count());

    match mask.bit_buffer() {
        AllOr::All => {
            for view in views {
                lines.push(view_bytes(view, buffers));
            }
        }
        AllOr::None => {}
        AllOr::Some(bits) => {
            for (view, valid) in views.iter().zip(bits.iter()) {
                if valid {
                    lines.push(view_bytes(view, buffers));
                }
            }
        }
    }

    Ok(Compressor::train(&lines))
}

fn train_varbin_array(
    strings: ArrayView<VarBin>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Compressor> {
    let mask = strings.validity()?.execute_mask(strings.len(), ctx)?;
    let offsets = strings.offsets().clone().execute::<PrimitiveArray>(ctx)?;
    let bytes = strings.bytes().as_slice();
    let mut lines: Vec<&[u8]> = Vec::with_capacity(mask.true_count());

    match_each_integer_ptype!(offsets.ptype(), |I| {
        let off = offsets.as_slice::<I>();
        for_each_varbin_row(off, bytes, &mask, |row| {
            if let Some(s) = row {
                lines.push(s);
            }
        });
    });

    Ok(Compressor::train(&lines))
}

#[inline]
fn fsst_output_fits_in_i32_offsets(total_input_bytes: usize) -> bool {
    let worst = total_input_bytes.saturating_mul(FSST_PER_BYTE_OVERHEAD);
    worst <= i32::MAX as usize
}

#[inline]
fn view_bytes<'a>(view: &'a BinaryView, buffers: &'a Arc<[BufferHandle]>) -> &'a [u8] {
    if view.is_inlined() {
        view.as_inlined().value()
    } else {
        let r = view.as_view();
        &buffers[r.buffer_index as usize].as_host()[r.as_range()]
    }
}

fn compress_views<O>(
    strings: ArrayView<VarBinView>,
    mask: &Mask,
    compressor: &Compressor,
    ctx: &mut ExecutionCtx,
) -> VortexResult<FSSTArray>
where
    O: IntegerPType + 'static,
{
    let mut sink = FsstSink::<O>::with_capacity(strings.len(), compressor);
    let views = strings.views();
    let buffers = strings.data_buffers();
    match mask.bit_buffer() {
        AllOr::All => {
            for view in views {
                sink.emit(Some(view_bytes(view, buffers)));
            }
        }
        AllOr::None => {
            for _ in 0..mask.len() {
                sink.emit(None);
            }
        }
        AllOr::Some(bits) => {
            for (view, valid) in views.iter().zip(bits.iter()) {
                sink.emit(valid.then(|| view_bytes(view, buffers)));
            }
        }
    }
    sink.finish(strings.dtype().clone(), ctx)
}

fn compress_varbin<O>(
    strings: ArrayView<VarBin>,
    offsets: &PrimitiveArray,
    mask: &Mask,
    compressor: &Compressor,
    ctx: &mut ExecutionCtx,
) -> VortexResult<FSSTArray>
where
    O: IntegerPType + 'static,
{
    let mut sink = FsstSink::<O>::with_capacity(strings.len(), compressor);
    let bytes = strings.bytes().as_slice();
    match_each_integer_ptype!(offsets.ptype(), |I| {
        let off = offsets.as_slice::<I>();
        for_each_varbin_row(off, bytes, mask, |row| sink.emit(row));
    });
    sink.finish(strings.dtype().clone(), ctx)
}

/// Call `f` once per row of a `VarBinArray` with the row bytes or `None`.
/// Validity dispatch is hoisted out of the per-row loop.
#[inline]
fn for_each_varbin_row<'a, I, F>(off: &[I], bytes: &'a [u8], mask: &Mask, mut f: F)
where
    I: IntegerPType + 'static,
    F: FnMut(Option<&'a [u8]>),
{
    match mask.bit_buffer() {
        AllOr::All => {
            for w in off.windows(2) {
                f(Some(&bytes[w[0].as_()..w[1].as_()]));
            }
        }
        AllOr::None => {
            for _ in 0..mask.len() {
                f(None);
            }
        }
        AllOr::Some(bits) => {
            for (w, valid) in off.windows(2).zip(bits.iter()) {
                f(valid.then(|| &bytes[w[0].as_()..w[1].as_()]));
            }
        }
    }
}

/// Per-row output state for an FSST compression pass.
struct FsstSink<'c, O: IntegerPType + 'static> {
    buffer: Vec<u8>,
    builder: VarBinBuilder<O>,
    uncompressed_lengths: BufferMut<i32>,
    compressor: &'c Compressor,
}

impl<'c, O: IntegerPType + 'static> FsstSink<'c, O> {
    fn with_capacity(len: usize, compressor: &'c Compressor) -> Self {
        Self {
            buffer: Vec::with_capacity(DEFAULT_BUFFER_LEN),
            builder: VarBinBuilder::<O>::with_capacity(len),
            uncompressed_lengths: BufferMut::with_capacity(len),
            compressor,
        }
    }

    #[inline]
    fn emit(&mut self, row: Option<&[u8]>) {
        let Some(s) = row else {
            self.builder.append_null();
            self.uncompressed_lengths.push(0);
            return;
        };

        // A single row > i32::MAX (2 GiB) is not supported.
        self.uncompressed_lengths.push(
            i32::try_from(s.len()).vortex_expect("per-row uncompressed length must fit in i32"),
        );

        let target = FSST_PER_BYTE_OVERHEAD * s.len();
        if target > self.buffer.len() {
            self.buffer.reserve(target - self.buffer.len());
        }

        // SAFETY: `self.buffer` has capacity for the FSST worst-case output of `s`.
        unsafe { self.compressor.compress_into(s, &mut self.buffer) };

        self.builder.append_value(&self.buffer);
    }

    fn finish(self, dtype: DType, ctx: &mut ExecutionCtx) -> VortexResult<FSSTArray> {
        let codes = self.builder.finish(DType::Binary(dtype.nullability()));
        FSST::try_new(
            dtype,
            Buffer::copy_from(self.compressor.symbol_table()),
            Buffer::<u8>::copy_from(self.compressor.symbol_lengths()),
            codes,
            self.uncompressed_lengths.into_array(),
            ctx,
        )
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::arrays::varbin::VarBinArrayExt;
    use vortex_array::dtype::PType;
    use vortex_error::VortexResult;

    use super::fsst_compress;
    use super::fsst_output_fits_in_i32_offsets;
    use super::fsst_train_compressor;
    use crate::array::FSSTArrayExt;

    /// Regression for #7833: the i32-vs-i64 codes-offsets decision must cross at
    /// `i32::MAX` against the worst-case bound `2 * total + 7 * non_null`.
    #[test]
    fn offset_width_boundary() {
        let m = i32::MAX as usize;
        assert!(fsst_output_fits_in_i32_offsets(m / 2 - 7));
        assert!(fsst_output_fits_in_i32_offsets(m / 2));
        assert!(fsst_output_fits_in_i32_offsets(0));
        assert!(!fsst_output_fits_in_i32_offsets(usize::MAX));
    }

    /// Small inputs fit the i32 bound, so `fsst_compress` must pick i32 offsets.
    /// The i64 branch is covered by `tests::fsst_compress_offsets_overflow_i32`.
    #[test]
    fn codes_offsets_dtype_small_input_is_i32() -> VortexResult<()> {
        let array = VarBinViewArray::from_iter_str(["hello", "world", "fsst encoded"]);
        let mut ctx = array_session().create_execution_ctx();
        let compressor = fsst_train_compressor(array.as_array(), &mut ctx)?;
        let fsst = fsst_compress(array.as_array(), &compressor, &mut ctx)?;
        assert_eq!(fsst.codes().offsets().dtype().as_ptype(), PType::I32);
        Ok(())
    }
}
