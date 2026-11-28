// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Feeds the valid values of a batch into a [`HyperLogLog`] sketch.
//!
//! Every value is hashed through its little-endian (for primitives) or raw (for strings and
//! binary) bytes. A constant and the canonical array it expands to must hash to the same bytes,
//! or a column stored as constant in one chunk and canonical in another would be counted twice.

use std::ops::BitAnd;

use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use super::sketch::HyperLogLog;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::ExtensionArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::VarBinViewArray;
use crate::arrays::bool::BoolArrayExt;
use crate::arrays::extension::ExtensionArrayExt;
use crate::arrays::varbinview::BinaryView;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::ToBytes;
use crate::match_each_native_ptype;
use crate::match_each_ptype_width;
use crate::scalar::Scalar;

const FALSE_BYTE: u8 = 0;
const TRUE_BYTE: u8 = 1;

pub(super) fn is_supported_dtype(dtype: &DType) -> bool {
    match dtype {
        DType::Extension(extension) => is_supported_dtype(extension.storage_dtype()),
        DType::Null | DType::Bool(_) | DType::Primitive(..) | DType::Utf8(_) | DType::Binary(_) => {
            true
        }
        // A decimal's physical width can differ between chunks, so hashing its bytes would count
        // one value several times.
        _ => false,
    }
}

pub(super) fn accumulate_canonical(
    canonical: &Canonical,
    sketch: &mut HyperLogLog,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    match canonical {
        Canonical::Null(_) => {}
        Canonical::Bool(array) => accumulate_bool(array, sketch, ctx)?,
        Canonical::Primitive(array) => accumulate_primitive(array, sketch, ctx)?,
        Canonical::VarBinView(array) => accumulate_varbinview(array, sketch, ctx)?,
        Canonical::Extension(array) => accumulate_extension(array, sketch, ctx)?,
        Canonical::Decimal(_)
        | Canonical::Struct(_)
        | Canonical::List(_)
        | Canonical::FixedSizeList(_)
        | Canonical::Variant(_)
        | Canonical::Union(_)
        | Canonical::Map(_) => {
            vortex_bail!(
                "approx_distinct does not support dtype {}",
                canonical.dtype()
            )
        }
    }
    Ok(())
}

pub(super) fn accumulate_constant(
    constant: &ConstantArray,
    sketch: &mut HyperLogLog,
) -> VortexResult<()> {
    // An empty constant array still carries a scalar.
    if constant.is_empty() {
        return Ok(());
    }
    insert_scalar(constant.scalar(), sketch)
}

/// Inserts `scalar` with the same bytes the canonical path hashes.
fn insert_scalar(scalar: &Scalar, sketch: &mut HyperLogLog) -> VortexResult<()> {
    if scalar.is_null() {
        return Ok(());
    }
    match scalar.dtype() {
        DType::Extension(_) => insert_scalar(&scalar.as_extension().to_storage_scalar(), sketch)?,
        DType::Bool(_) => {
            let value = scalar
                .as_bool()
                .value()
                .vortex_expect("non-null boolean scalar has a value");
            sketch.insert(&[if value { TRUE_BYTE } else { FALSE_BYTE }]);
        }
        DType::Primitive(ptype, _) => {
            match_each_native_ptype!(*ptype, |T| { insert_primitive_scalar::<T>(scalar, sketch) })
        }
        DType::Utf8(_) => {
            let value = scalar
                .as_utf8()
                .value()
                .vortex_expect("non-null utf8 scalar has a value");
            sketch.insert(value.as_bytes());
        }
        DType::Binary(_) => {
            let value = scalar
                .as_binary()
                .value()
                .vortex_expect("non-null binary scalar has a value");
            sketch.insert(value.as_slice());
        }
        other => vortex_bail!("approx_distinct does not support dtype {other}"),
    }
    Ok(())
}

fn insert_primitive_scalar<T>(scalar: &Scalar, sketch: &mut HyperLogLog)
where
    T: ToBytes + NativePType,
{
    let value = scalar
        .as_primitive()
        .typed_value::<T>()
        .vortex_expect("non-null primitive scalar has a value");
    sketch.insert(value.to_le_bytes());
}

fn accumulate_bool(
    array: &BoolArray,
    sketch: &mut HyperLogLog,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let validity = array.validity()?.execute_mask(array.len(), ctx)?;
    let (true_count, valid_count) = match validity.bit_buffer() {
        AllOr::None => return Ok(()),
        AllOr::All => (array.bit_buffer_view().true_count(), array.len()),
        AllOr::Some(valid) => (
            array.to_bit_buffer().bitand(valid).true_count(),
            valid.true_count(),
        ),
    };
    // At most two distinct values, so insert each once instead of once per row.
    if true_count > 0 {
        sketch.insert(&[TRUE_BYTE]);
    }
    if true_count < valid_count {
        sketch.insert(&[FALSE_BYTE]);
    }
    Ok(())
}

fn accumulate_primitive(
    array: &PrimitiveArray,
    sketch: &mut HyperLogLog,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let validity = array.validity()?.execute_mask(array.len(), ctx)?;
    let values = array.buffer_handle().to_host_sync();
    // Hashing only sees bytes, so types of the same width share one instantiation.
    match_each_ptype_width!(
        array.ptype(),
        |W| { insert_fixed_width_values::<W>(values, &validity, sketch) },
        _ => { unreachable!("primitive widths are specialized") },
    );
    Ok(())
}

fn insert_fixed_width_values<const W: usize>(
    values: ByteBuffer,
    validity: &Mask,
    sketch: &mut HyperLogLog,
) {
    let records = Buffer::<[u8; W]>::from_byte_buffer_aligned(values, Alignment::new(W));
    // Floats hash their raw bits, so `-0.0` and `0.0` are distinct, matching `NativePType::is_eq`.
    match validity {
        Mask::AllTrue(_) => {
            for record in records.iter() {
                sketch.insert(record);
            }
        }
        Mask::AllFalse(_) => {}
        Mask::Values(mask_values) => {
            for &(start, end) in mask_values.slices() {
                for record in &records[start..end] {
                    sketch.insert(record);
                }
            }
        }
    }
}

fn accumulate_varbinview(
    array: &VarBinViewArray,
    sketch: &mut HyperLogLog,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let validity = array.validity()?.execute_mask(array.len(), ctx)?;
    if validity.all_false() {
        return Ok(());
    }
    let buffers = array
        .data_buffers()
        .iter()
        .map(|buffer| buffer.as_host())
        .collect::<Vec<_>>();
    let views = array.views();
    match &validity {
        Mask::AllTrue(_) => {
            for view in views.iter() {
                sketch.insert(view_bytes(view, &buffers));
            }
        }
        Mask::AllFalse(_) => {}
        Mask::Values(mask_values) => {
            for &(start, end) in mask_values.slices() {
                for view in &views[start..end] {
                    sketch.insert(view_bytes(view, &buffers));
                }
            }
        }
    }
    Ok(())
}

fn view_bytes<'a>(view: &'a BinaryView, buffers: &[&'a Buffer<u8>]) -> &'a [u8] {
    // Hash the value's bytes, never the view, so a string counts once wherever it is stored.
    if view.is_inlined() {
        view.as_inlined().value()
    } else {
        let reference = view.as_view();
        &buffers[reference.buffer_index as usize][reference.as_range()]
    }
}

fn accumulate_extension(
    array: &ExtensionArray,
    sketch: &mut HyperLogLog,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let storage = array.storage_array().clone().execute::<Canonical>(ctx)?;
    accumulate_canonical(&storage, sketch, ctx)
}
