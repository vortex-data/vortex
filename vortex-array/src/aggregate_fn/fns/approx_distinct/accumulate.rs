// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Hashing of columnar batches into a [`HllSketch`].

use std::ops::BitAnd;

use twox_hash::XxHash3_64;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use super::sketch::HllSketch;
use super::sketch::mix64;
use crate::Canonical;
use crate::Columnar;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::DecimalArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::VarBinViewArray;
use crate::arrays::bool::BoolArrayExt;
use crate::arrays::extension::ExtensionArrayExt;
use crate::arrays::varbinview::BinaryView;
use crate::dtype::NativeDecimalType;
use crate::dtype::PType;
use crate::match_each_decimal_value_type;
use crate::match_each_integer_ptype;

/// Values are hashed into a stack buffer of this many hashes before they update the registers,
/// so the hashing loop has no data-dependent stores and vectorizes.
const HASH_BATCH: usize = 1024;

/// Adds the non-null values of `batch` to `sketch`.
pub(super) fn accumulate_columnar(
    batch: &Columnar,
    sketch: &mut HllSketch,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    match batch {
        Columnar::Constant(constant) => accumulate_constant(constant, sketch, ctx),
        Columnar::Canonical(canonical) => accumulate_canonical(canonical, sketch, ctx),
    }
}

/// A constant holds at most one distinct value, hashed like a canonical array of it.
fn accumulate_constant(
    constant: &ConstantArray,
    sketch: &mut HllSketch,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    if constant.scalar().is_null() {
        return Ok(());
    }
    let single = ConstantArray::new(constant.scalar().clone(), 1)
        .into_array()
        .execute::<Canonical>(ctx)?;
    accumulate_canonical(&single, sketch, ctx)
}

fn accumulate_canonical(
    canonical: &Canonical,
    sketch: &mut HllSketch,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    match canonical {
        Canonical::Null(_) => Ok(()),
        Canonical::Bool(array) => accumulate_bool(array, sketch, ctx),
        Canonical::Primitive(array) => accumulate_primitive(array, sketch, ctx),
        Canonical::Decimal(array) => accumulate_decimal(array, sketch, ctx),
        Canonical::VarBinView(array) => accumulate_varbinview(array, sketch, ctx),
        Canonical::Extension(array) => {
            let storage = array.storage_array().clone().execute::<Columnar>(ctx)?;
            accumulate_columnar(&storage, sketch, ctx)
        }
        Canonical::Struct(_)
        | Canonical::List(_)
        | Canonical::FixedSizeList(_)
        | Canonical::Variant(_)
        | Canonical::Union(_)
        | Canonical::Map(_) => {
            vortex_bail!("approx_distinct does not support {}", canonical.dtype())
        }
    }
}

/// Hashes `values[i]` for every valid `i` with `hash` and adds them to `sketch`.
fn insert_valid<T: Copy>(
    values: &[T],
    validity: &Mask,
    sketch: &mut HllSketch,
    hash: impl Fn(T) -> u64,
) {
    let mut insert_run = |run: &[T]| {
        let mut hashes = [0u64; HASH_BATCH];
        for chunk in run.chunks(HASH_BATCH) {
            let hashes = &mut hashes[..chunk.len()];
            for (slot, &value) in hashes.iter_mut().zip(chunk) {
                *slot = hash(value);
            }
            sketch.insert_hashes(hashes);
        }
    };
    match validity {
        Mask::AllTrue(_) => insert_run(values),
        Mask::AllFalse(_) => {}
        Mask::Values(mask) => {
            for &(start, end) in mask.slices() {
                insert_run(&values[start..end]);
            }
        }
    }
}

fn accumulate_bool(
    array: &BoolArray,
    sketch: &mut HllSketch,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let mask = array.validity()?.execute_mask(array.len(), ctx)?;
    let (true_count, valid_count) = match mask.bit_buffer() {
        AllOr::None => return Ok(()),
        AllOr::All => (array.bit_buffer_view().true_count(), array.len()),
        AllOr::Some(validity) => (
            array.to_bit_buffer().bitand(validity).true_count(),
            validity.true_count(),
        ),
    };
    if true_count > 0 {
        sketch.insert_hash(mix64(1));
    }
    if true_count < valid_count {
        sketch.insert_hash(mix64(0));
    }
    Ok(())
}

fn accumulate_primitive(
    array: &PrimitiveArray,
    sketch: &mut HllSketch,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let validity = array.validity()?.execute_mask(array.len(), ctx)?;
    match array.ptype() {
        // Sign- or zero-extending to 64 bits keeps equal values equal; a column has one ptype.
        #[allow(
            clippy::cast_sign_loss,
            clippy::cast_possible_wrap,
            clippy::unnecessary_cast,
            reason = "reinterpreting the bits of every integer width"
        )]
        ptype if ptype.is_int() => match_each_integer_ptype!(ptype, |T| {
            insert_valid(array.as_slice::<T>(), &validity, sketch, |value| {
                mix64(value as i64 as u64)
            })
        }),
        PType::F16 => insert_valid(array.as_slice::<half::f16>(), &validity, sketch, |value| {
            mix64(u64::from(normalize_f16(value).to_bits()))
        }),
        PType::F32 => insert_valid(array.as_slice::<f32>(), &validity, sketch, |value| {
            mix64(u64::from(normalize_f32(value).to_bits()))
        }),
        PType::F64 => insert_valid(array.as_slice::<f64>(), &validity, sketch, |value| {
            mix64(normalize_f64(value).to_bits())
        }),
        ptype => vortex_bail!("approx_distinct does not support {ptype}"),
    }
    Ok(())
}

fn normalize_f16(value: half::f16) -> half::f16 {
    if value.is_nan() {
        half::f16::NAN
    } else if value == half::f16::ZERO {
        half::f16::ZERO
    } else {
        value
    }
}

fn normalize_f32(value: f32) -> f32 {
    if value.is_nan() {
        f32::NAN
    } else if value == 0.0 {
        0.0
    } else {
        value
    }
}

fn normalize_f64(value: f64) -> f64 {
    if value.is_nan() {
        f64::NAN
    } else if value == 0.0 {
        0.0
    } else {
        value
    }
}

/// Hashes a decimal by its unscaled value, so chunks stored in different integer widths agree.
#[inline]
fn hash_decimal<D: NativeDecimalType>(value: D) -> u64 {
    match value.to_i64() {
        #[expect(clippy::cast_sign_loss, reason = "reinterpreting the bits")]
        Some(value) => mix64(value as u64),
        None => {
            let (low, high) = value
                .to_i256()
                .vortex_expect("every decimal value fits in i256")
                .to_parts();
            let mut bytes = [0u8; 32];
            bytes[..16].copy_from_slice(&low.to_le_bytes());
            bytes[16..].copy_from_slice(&high.to_le_bytes());
            XxHash3_64::oneshot(&bytes)
        }
    }
}

fn accumulate_decimal(
    array: &DecimalArray,
    sketch: &mut HllSketch,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let validity = array.validity()?.execute_mask(array.len(), ctx)?;
    match_each_decimal_value_type!(array.values_type(), |D| {
        let values = array.buffer::<D>();
        insert_valid(values.as_slice(), &validity, sketch, hash_decimal::<D>);
    });
    Ok(())
}

fn accumulate_varbinview(
    array: &VarBinViewArray,
    sketch: &mut HllSketch,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let validity = array.validity()?.execute_mask(array.len(), ctx)?;
    let buffers = array
        .data_buffers()
        .iter()
        .map(|buffer| buffer.as_host())
        .collect::<Vec<&Buffer<u8>>>();
    let hash_view = |view: &BinaryView| {
        if view.is_inlined() {
            XxHash3_64::oneshot(view.as_inlined().value())
        } else {
            let view = view.as_view();
            XxHash3_64::oneshot(&buffers[view.buffer_index as usize][view.as_range()])
        }
    };
    let views = array.views();
    match &validity {
        Mask::AllTrue(_) => {
            for view in views {
                sketch.insert_hash(hash_view(view));
            }
        }
        Mask::AllFalse(_) => {}
        Mask::Values(mask) => {
            for &(start, end) in mask.slices() {
                for view in &views[start..end] {
                    sketch.insert_hash(hash_view(view));
                }
            }
        }
    }
    Ok(())
}
