// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A size-first pass over a compressed array that swaps in high-level zstd wherever it is smaller.
//!
//! BtrBlocks' compact preset compresses strings with zstd level 3 in frames of 8192 values and has
//! no zstd for integers. Label data repeats over long distances (row codes follow repeating trace
//! shapes, messages repeat with small edits), which high zstd levels over large frames capture.
//! This pass tries that on every integer and string node and keeps the smaller encoding. It
//! trades random access within a frame for size.

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Dict;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_array::dtype::DType;
use vortex_array::match_each_integer_ptype;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_zstd::Zstd;

/// Nodes smaller than this are left alone.
const MIN_BYTES: u64 = 4096;
/// Integer values per zstd frame.
const INT_FRAME: usize = 1 << 20;
/// Target uncompressed bytes per string frame.
const STRING_FRAME_BYTES: usize = 4 << 20;

/// Re-encodes integer and string nodes of `array` with zstd at `level` wherever that is smaller,
/// keeping every node's dtype.
pub fn squeeze(array: &ArrayRef, level: i32, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    if array.nbytes() < MIN_BYTES {
        return Ok(array.clone());
    }
    if let Some(dict) = array.as_opt::<Dict>() {
        let codes = squeeze_codes(dict.codes(), level, ctx)?;
        let values = squeeze(dict.values(), level, ctx)?;
        let candidate = DictArray::try_new(codes, values)?.into_array();
        return Ok(smaller(array.clone(), candidate));
    }
    match array.dtype() {
        DType::Utf8(_) | DType::Binary(_) => {
            let strings = array.clone().execute::<VarBinViewArray>(ctx)?;
            let bytes: usize = strings.data_buffers().iter().map(|b| b.len()).sum::<usize>()
                + strings.len() * 4;
            let per_value = (bytes / strings.len().max(1)).max(1);
            let frame = (STRING_FRAME_BYTES / per_value).clamp(1024, strings.len().max(1024));
            let candidate = Zstd::from_var_bin_view(&strings, level, frame, ctx)?.into_array();
            Ok(smaller(array.clone(), candidate))
        }
        DType::Primitive(ptype, _) if ptype.is_int() => {
            let primitive = array.clone().execute::<PrimitiveArray>(ctx)?;
            let candidate = Zstd::from_primitive(&primitive, level, INT_FRAME, ctx)?.into_array();
            Ok(smaller(array.clone(), candidate))
        }
        _ => {
            let slots = array
                .slots()
                .iter()
                .map(|slot| slot.as_ref().map(|c| squeeze(c, level, ctx)).transpose())
                .collect::<VortexResult<_>>()?;
            // SAFETY: every replaced child keeps its dtype, length and values.
            unsafe { array.clone().with_slots(slots) }
        }
    }
}

/// Dictionary codes in their narrowest unsigned type, as zstd or as already compressed,
/// whichever is smaller. Codes may change type since a dictionary accepts any integer codes.
fn squeeze_codes(codes: &ArrayRef, level: i32, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    if codes.nbytes() < MIN_BYTES {
        return Ok(codes.clone());
    }
    let primitive = codes.clone().execute::<PrimitiveArray>(ctx)?;
    let validity = primitive.validity()?;
    let max = match_each_integer_ptype!(primitive.ptype(), |P| {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        primitive.as_slice::<P>().iter().map(|&c| c as u64).max().unwrap_or(0)
    });
    let narrow = match_each_integer_ptype!(primitive.ptype(), |P| {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let values = primitive.as_slice::<P>();
        if max <= u64::from(u8::MAX) {
            PrimitiveArray::new(values.iter().map(|&c| c as u8).collect::<Buffer<u8>>(), validity)
        } else if max <= u64::from(u16::MAX) {
            PrimitiveArray::new(values.iter().map(|&c| c as u16).collect::<Buffer<u16>>(), validity)
        } else {
            PrimitiveArray::new(values.iter().map(|&c| c as u32).collect::<Buffer<u32>>(), validity)
        }
    });
    let candidate = Zstd::from_primitive(&narrow, level, INT_FRAME, ctx)?.into_array();
    Ok(smaller(codes.clone(), candidate))
}

fn smaller(current: ArrayRef, candidate: ArrayRef) -> ArrayRef {
    if candidate.nbytes() < current.nbytes() {
        candidate
    } else {
        current
    }
}
