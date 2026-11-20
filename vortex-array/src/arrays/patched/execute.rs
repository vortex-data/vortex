// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Canonicalization of [`Patched`] arrays.

use std::sync::Arc;

use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::ChunkedArray;
use crate::arrays::DecimalArray;
use crate::arrays::ExtensionArray;
use crate::arrays::Patched;
use crate::arrays::PrimitiveArray;
use crate::arrays::StructArray;
use crate::arrays::VarBinViewArray;
use crate::arrays::bool::BoolArrayExt;
use crate::arrays::extension::ExtensionArrayExt;
use crate::arrays::patched::PatchLayout;
use crate::arrays::patched::PatchedSlots;
use crate::arrays::primitive::PrimitiveDataParts;
use crate::arrays::struct_::StructArrayExt;
use crate::arrays::varbinview::BinaryView;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::match_each_decimal_value_type;
use crate::match_each_native_ptype;
use crate::validity::Validity;

/// The executed children of a patched array.
pub(crate) struct PatchedCanonicalParts<'a> {
    pub dtype: DType,
    pub layout: PatchLayout<'a>,
    /// The un-executed lane offsets and indices, which lazy results reuse.
    pub lane_offsets: ArrayRef,
    pub patch_indices: ArrayRef,
    pub inner: Canonical,
    pub values: Canonical,
}

/// Canonicalize a patched array whose inner and values children are canonical.
pub(crate) fn execute_canonical(
    parts: PatchedCanonicalParts<'_>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let PatchedCanonicalParts {
        dtype,
        layout,
        lane_offsets,
        patch_indices,
        inner,
        values,
    } = parts;

    if layout.all_patches().is_empty() {
        return Ok(inner.into_array());
    }

    match (inner, values) {
        (Canonical::Null(inner), _) => Ok(inner.into_array()),
        (Canonical::Primitive(inner), Canonical::Primitive(values)) => {
            Ok(patch_primitive(inner, &values, &layout, ctx)?.into_array())
        }
        (Canonical::Bool(inner), Canonical::Bool(values)) => {
            patch_bool(inner, &values, &layout, ctx)
        }
        (Canonical::Decimal(inner), Canonical::Decimal(values))
            if inner.values_type() == values.values_type() =>
        {
            patch_decimal(inner, &values, &layout, ctx)
        }
        (Canonical::VarBinView(inner), Canonical::VarBinView(values)) => {
            patch_varbinview(inner, &values, &layout, ctx)
        }
        (Canonical::Struct(inner), Canonical::Struct(values)) => {
            let validity = patch_validity(
                inner.as_ref().validity()?,
                values.as_ref().validity()?,
                values.len(),
                &layout,
                ctx,
            )?;
            let fields = inner
                .iter_unmasked_fields()
                .zip(values.iter_unmasked_fields())
                .map(|(inner, values)| {
                    lazy_patched(
                        inner.dtype().clone(),
                        &layout,
                        &lane_offsets,
                        &patch_indices,
                        inner.clone(),
                        values.clone(),
                    )
                })
                .collect::<Vec<_>>();
            Ok(StructArray::try_new_with_dtype(
                fields,
                inner.struct_fields().clone(),
                layout.len(),
                validity,
            )?
            .into_array())
        }
        (Canonical::Extension(inner), Canonical::Extension(values)) => {
            let storage = lazy_patched(
                inner.storage_array().dtype().clone(),
                &layout,
                &lane_offsets,
                &patch_indices,
                inner.storage_array().clone(),
                values.storage_array().clone(),
            );
            Ok(ExtensionArray::try_new(inner.ext_dtype().clone(), storage)?.into_array())
        }
        (inner, values) => {
            patch_with_take(dtype, inner.into_array(), values.into_array(), &layout)?
                .execute::<Canonical>(ctx)
                .map(IntoArray::into_array)
        }
    }
}

/// Wrap the children of a patched array's field or storage in a new lazy patched array.
fn lazy_patched(
    dtype: DType,
    layout: &PatchLayout<'_>,
    lane_offsets: &ArrayRef,
    patch_indices: &ArrayRef,
    inner: ArrayRef,
    values: ArrayRef,
) -> ArrayRef {
    let slots = PatchedSlots {
        inner,
        lane_offsets: lane_offsets.clone(),
        patch_indices: patch_indices.clone(),
        patch_values: values,
    }
    .into_slots();
    // SAFETY: the field shares the length, dtype and layout of its patched parent.
    unsafe {
        Patched::new_unchecked(
            dtype,
            layout.len(),
            slots,
            layout.n_lanes(),
            layout.offset(),
        )
    }
    .into_array()
}

/// The validity of a patched array, given the validity of its inner and values children.
pub(crate) fn patch_validity(
    inner: Validity,
    values: Validity,
    n_values: usize,
    layout: &PatchLayout<'_>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Validity> {
    if matches!(inner, Validity::NonNullable) {
        return Ok(Validity::NonNullable);
    }
    if inner.definitely_no_nulls() && values.definitely_no_nulls() {
        return Ok(Validity::AllValid);
    }

    let values = values.execute_mask(n_values, ctx)?;
    let inner = inner.execute_mask(layout.len(), ctx)?;
    if values.all_true() && inner.all_true() {
        return Ok(Validity::AllValid);
    }

    let mut bits = match inner {
        Mask::AllTrue(len) => BitBufferMut::new_set(len),
        Mask::AllFalse(len) => BitBufferMut::new_unset(len),
        inner @ Mask::Values(_) => inner
            .into_bit_buffer()
            .try_into_mut()
            .unwrap_or_else(|bits| BitBufferMut::copy_from(&bits)),
    };
    match values {
        Mask::AllTrue(_) => layout.for_each(|pos, _| bits.set(pos)),
        Mask::AllFalse(_) => layout.for_each(|pos, _| bits.unset(pos)),
        Mask::Values(values) => {
            let values = values.bit_buffer();
            layout.for_each(|pos, patch| bits.set_to(pos, values.value(patch)))
        }
    }
    Ok(Validity::from(bits.freeze()))
}

/// Overwrite the patched positions of `output` with `values`.
#[inline]
pub(crate) fn scatter<V: Copy>(output: &mut [V], values: &[V], layout: &PatchLayout<'_>) {
    assert_eq!(output.len(), layout.len());
    assert!(layout.all_patches().end <= values.len());
    layout.for_each(|pos, patch| output[pos] = values[patch]);
}

fn patch_primitive(
    inner: PrimitiveArray,
    values: &PrimitiveArray,
    layout: &PatchLayout<'_>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let validity = patch_validity(
        inner.validity()?,
        values.validity()?,
        values.len(),
        layout,
        ctx,
    )?;
    let PrimitiveDataParts { buffer, ptype, .. } = inner.into_data_parts();
    Ok(match_each_native_ptype!(ptype, |V| {
        let mut output = Buffer::<V>::from_byte_buffer(buffer.unwrap_host()).into_mut();
        scatter(&mut output, values.as_slice::<V>(), layout);
        PrimitiveArray::new(output.freeze(), validity)
    }))
}

fn patch_bool(
    inner: BoolArray,
    values: &BoolArray,
    layout: &PatchLayout<'_>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let validity = patch_validity(
        inner.validity()?,
        values.validity()?,
        values.len(),
        layout,
        ctx,
    )?;
    let patch_bits = values.to_bit_buffer();
    let mut bits = inner
        .into_bit_buffer()
        .try_into_mut()
        .unwrap_or_else(|bits| BitBufferMut::copy_from(&bits));
    layout.for_each(|pos, patch| bits.set_to(pos, patch_bits.value(patch)));
    Ok(BoolArray::new(bits.freeze(), validity).into_array())
}

fn patch_decimal(
    inner: DecimalArray,
    values: &DecimalArray,
    layout: &PatchLayout<'_>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let validity = patch_validity(
        inner.validity()?,
        values.validity()?,
        values.len(),
        layout,
        ctx,
    )?;
    let decimal_dtype = inner.decimal_dtype();
    let values_type = inner.values_type();
    let output = match_each_decimal_value_type!(values_type, |D| {
        let mut output = inner.buffer::<D>().into_mut();
        scatter(&mut output, values.buffer::<D>().as_slice(), layout);
        output.freeze().into_byte_buffer()
    });
    // SAFETY: every value is either a value of `inner` or of `values`, which share the decimal
    // dtype and value type, and the validity has one entry per value.
    Ok(unsafe {
        DecimalArray::new_unchecked_handle(
            BufferHandle::new_host(output),
            values_type,
            decimal_dtype,
            validity,
        )
    }
    .into_array())
}

#[expect(clippy::cast_possible_truncation, reason = "buffer counts fit in u32")]
fn patch_varbinview(
    inner: VarBinViewArray,
    values: &VarBinViewArray,
    layout: &PatchLayout<'_>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let validity = patch_validity(
        inner.validity()?,
        values.validity()?,
        values.len(),
        layout,
        ctx,
    )?;

    let n_inner_buffers = inner.data_buffers().len();
    let buffer_shift = n_inner_buffers as u32;
    let buffers: Arc<[_]> = inner
        .data_buffers()
        .iter()
        .chain(values.data_buffers().iter())
        .cloned()
        .collect();

    let mut views = BufferMut::<BinaryView>::copy_from(inner.views());
    let value_views = values.views();
    layout.for_each(|pos, patch| {
        let view = value_views[patch];
        views[pos] = if view.is_inlined() || buffer_shift == 0 {
            view
        } else {
            let r = view.as_view();
            BinaryView::from(r.with_buffer_and_offset(r.buffer_index + buffer_shift, r.offset))
        };
    });

    // SAFETY: the views of `inner` reference its own buffers unchanged, and the views of `values`
    // reference its buffers, which follow those of `inner`.
    Ok(unsafe {
        VarBinViewArray::new_handle_unchecked(
            BufferHandle::new_host(views.freeze().into_byte_buffer()),
            buffers,
            inner.dtype().clone(),
            validity,
        )
    }
    .into_array())
}

/// Patch an array of any type by taking from the concatenation of the inner and values arrays.
fn patch_with_take(
    dtype: DType,
    inner: ArrayRef,
    values: ArrayRef,
    layout: &PatchLayout<'_>,
) -> VortexResult<ArrayRef> {
    let len = layout.len();
    let mut take_indices: BufferMut<u64> = (0..len as u64).collect();
    layout.for_each(|pos, patch| take_indices[pos] = (len + patch) as u64);
    let concat = ChunkedArray::try_new([inner, values], dtype)?.into_array();
    concat.take(PrimitiveArray::new(take_indices.freeze(), Validity::NonNullable).into_array())
}
