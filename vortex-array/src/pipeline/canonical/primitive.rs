use arrow_buffer::BooleanBuffer;
use vortex_buffer::{Alignment, BufferMut};
use vortex_dtype::{NativePType, Nullability};
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::arrays::PrimitiveArray;
use crate::pipeline::bits::{BitVector, BitView, BitViewMut};
use crate::pipeline::view::ViewMut;
use crate::pipeline::{Element, Kernel, KernelContext, N, N_WORDS};
use crate::validity::Validity;

pub(super) fn export_primitive_nonnull<T: Element + NativePType>(
    len: usize,
    pipeline: &mut dyn Kernel,
) -> VortexResult<PrimitiveArray> {
    let capacity = len.next_multiple_of(N) + N;

    let mut elements = BufferMut::<T>::with_capacity(capacity);
    unsafe { elements.set_len(capacity) };

    let mut remaining = len;
    while remaining >= N {
        let mut elements_view = ViewMut::new(&mut elements[len - remaining..][..N], None);
        let dummy_ctx = KernelContext::default();
        pipeline.step(&dummy_ctx, BitView::all_true(), &mut elements_view)?;
        remaining -= N;
    }

    if remaining > 0 {
        let mut elements_view = ViewMut::new(&mut elements[len - remaining..][..N], None);
        let mask = BitVector::true_until(remaining);
        let dummy_ctx = KernelContext::default();
        pipeline.step(&dummy_ctx, mask.as_view(), &mut elements_view)?;
    }

    unsafe { elements.set_len(len) };

    Ok(PrimitiveArray::new(
        elements.freeze(),
        Validity::NonNullable,
    ))
}

pub(super) fn export_primitive_null<T: Element + NativePType>(
    len: usize,
    pipeline: &mut dyn Kernel,
) -> VortexResult<PrimitiveArray> {
    let capacity = len.next_multiple_of(N) + N;

    let mut elements = BufferMut::<T>::with_capacity(capacity);
    unsafe { elements.set_len(capacity) };

    // should we allocate in multiples of N_WORDS?
    let mut mask =
        BufferMut::<usize>::full(0, len.div_ceil(N_WORDS) * N_WORDS).aligned(Alignment::new(1024));

    let mut remaining = len;

    while remaining >= N {
        let head = len - remaining;
        let mut slice: &mut [usize; N_WORDS] =
            unsafe { extract_step_slice(&mut (mask[head / (u32::BITS as usize)..][..N_WORDS])) };
        let val_view = BitViewMut::new(&mut slice);
        let mut elements_view = ViewMut::new(&mut elements[head..][..N], Some(val_view));
        let dummy_ctx = KernelContext::default();
        pipeline.step(&dummy_ctx, BitView::all_true(), &mut elements_view)?;
        remaining -= N;
    }

    if remaining > 0 {
        let head = len - remaining;
        let slice: &mut [usize; N_WORDS] =
            unsafe { extract_step_slice(&mut mask[head / (u32::BITS as usize)..][..N_WORDS]) };

        let val_view = BitViewMut::new(slice);
        let mut elements_view = ViewMut::new(&mut elements[head..][..N], Some(val_view));
        let mask = BitVector::true_until(remaining);
        let dummy_ctx = KernelContext::default();

        pipeline.step(&dummy_ctx, mask.as_view(), &mut elements_view)?;
    }

    unsafe { elements.set_len(len) };

    let abuf = arrow_buffer::Buffer::from(mask.freeze().into_inner());
    let buf = BooleanBuffer::new(abuf, 0, len);
    let mask = Mask::from_buffer(buf);
    Ok(PrimitiveArray::new(
        elements.freeze(),
        Validity::from_mask(mask, Nullability::Nullable),
    ))
}

unsafe fn extract_step_slice(slice: &mut [usize]) -> &mut [usize; N_WORDS] {
    unsafe { &mut *(slice.as_mut_ptr() as *mut [usize; N_WORDS]) }
}

pub(super) fn export_primitive_nonnull_masked<T: Element + NativePType>(
    mask: &Mask,
    pipeline: &mut dyn Kernel,
) -> VortexResult<PrimitiveArray> {
    let len = mask.len();
    let capacity = mask.true_count().next_multiple_of(N) + N;

    let mut elements = BufferMut::<T>::with_capacity(capacity);
    unsafe { elements.set_len(capacity) };

    let mask_buffer = mask.to_boolean_buffer();
    let mut mask_iter = mask_buffer.bit_chunks().iter_padded();

    let mut mask = [0usize; N_WORDS];
    let mut mask_view = BitViewMut::new(&mut mask);

    let mut offset = 0;
    let mut remaining = len;
    while remaining > 0 {
        let mut elements_view = ViewMut::new(&mut elements[offset..][..N], None);

        mask_view.clear();
        mask_view.fill_with_words(&mut mask_iter);

        let dummy_ctx = KernelContext::default();
        pipeline.step(&dummy_ctx, mask_view.as_view(), &mut elements_view)?;
        offset += mask_view.true_count();

        remaining = remaining.saturating_sub(N);
    }

    unsafe { elements.set_len(offset) };

    Ok(PrimitiveArray::new(
        elements.freeze(),
        Validity::NonNullable,
    ))
}
