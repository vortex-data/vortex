use std::ops::Deref;

use num_traits::AsPrimitive;
use vortex_buffer::{Buffer, ByteBuffer};
use vortex_dtype::match_each_integer_ptype;
use vortex_error::{vortex_bail, VortexResult};
use vortex_mask::Mask;

use crate::arrays::{BinaryView, VarBinViewArray, VarBinViewEncoding};
use crate::builders::{ArrayBuilder, VarBinViewBuilder};
use crate::compute::TakeFn;
use crate::variants::PrimitiveArrayTrait;
use crate::{Array, ArrayRef, ToCanonical};

/// Take involves creating a new array that references the old array, just with the given set of views.
impl TakeFn<&VarBinViewArray> for VarBinViewEncoding {
    fn take(&self, array: &VarBinViewArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        // Compute the new validity

        // This is valid since all elements (of all arrays) even null values are inside must be the
        // min-max valid range.
        let validity = array.validity().take(indices)?;
        let indices = indices.to_primitive()?;

        let views_buffer = match_each_integer_ptype!(indices.ptype(), |$I| {
        // This is valid since all elements even null values are inside the min-max valid range.
            take_views(array.views(), indices.as_slice::<$I>())
        });

        Ok(VarBinViewArray::try_new(
            views_buffer,
            array.buffers().to_vec(),
            array.dtype().with_nullability(
                (array.dtype().is_nullable() || indices.dtype().is_nullable()).into(),
            ),
            validity,
        )?
        .into_array())
    }

    unsafe fn take_unchecked(
        &self,
        array: &VarBinViewArray,
        indices: &dyn Array,
    ) -> VortexResult<ArrayRef> {
        // Compute the new validity
        let validity = array.validity().take(indices)?;
        let indices = indices.to_primitive()?;

        let views_buffer = match_each_integer_ptype!(indices.ptype(), |$I| {
            take_views_unchecked(array.views(), indices.as_slice::<$I>())
        });

        Ok(VarBinViewArray::try_new(
            views_buffer,
            array.buffers().to_vec(),
            array.dtype().clone(),
            validity,
        )?
        .into_array())
    }

    fn take_into(
        &self,
        array: &VarBinViewArray,
        indices: &dyn Array,
        builder: &mut dyn ArrayBuilder,
    ) -> VortexResult<()> {
        if array.len() == 0 {
            vortex_bail!("Cannot take_into from an empty array");
        }

        let Some(builder) = builder.as_any_mut().downcast_mut::<VarBinViewBuilder>() else {
            vortex_bail!(
                "Cannot take_into a non-varbinview builder {:?}",
                builder.as_any().type_id()
            );
        };
        // Compute the new validity

        // This is valid since all elements (of all arrays) even null values are inside must be the
        // min-max valid range.
        // TODO(joe): impl validity_mask take
        let validity = array.validity().take(indices)?;
        let mask = validity.to_logical(indices.len())?;
        let indices = indices.to_primitive()?;

        match_each_integer_ptype!(indices.ptype(), |$I| {
            // This is valid since all elements even null values are inside the min-max valid range.
            take_views_into(array.views(), array.buffers(), indices.as_slice::<$I>(), mask, builder)?;
        });

        Ok(())
    }
}

fn take_views_into<I: AsPrimitive<usize>>(
    views: &Buffer<BinaryView>,
    buffers: &[ByteBuffer],
    indices: &[I],
    mask: Mask,
    builder: &mut VarBinViewBuilder,
) -> VortexResult<()> {
    let buffers_offset = u32::try_from(builder.completed_block_count())?;
    // NOTE(ngates): this deref is not actually trivial, so we run it once.
    let views_ref = views.deref();
    builder.push_buffer_and_adjusted_views(
        buffers.iter().cloned(),
        indices
            .iter()
            .map(|i| views_ref[i.as_()].offset_view(buffers_offset)),
        mask,
    );
    Ok(())
}

fn take_views<I: AsPrimitive<usize>>(
    views: &Buffer<BinaryView>,
    indices: &[I],
) -> Buffer<BinaryView> {
    // NOTE(ngates): this deref is not actually trivial, so we run it once.
    let views_ref = views.deref();
    Buffer::<BinaryView>::from_iter(indices.iter().map(|i| views_ref[i.as_()]))
}

fn take_views_unchecked<I: AsPrimitive<usize>>(
    views: &Buffer<BinaryView>,
    indices: &[I],
) -> Buffer<BinaryView> {
    // NOTE(ngates): this deref is not actually trivial, so we run it once.
    let views_ref = views.deref();
    Buffer::from_iter(
        indices
            .iter()
            .map(|i| unsafe { *views_ref.get_unchecked(i.as_()) }),
    )
}
