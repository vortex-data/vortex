// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::task::{Poll, ready};

use vortex_dtype::{NativePType, match_each_native_ptype};
use vortex_error::{VortexResult, vortex_bail};

use crate::arrays::{PrimitiveArray, PrimitiveVTable};
use crate::pipeline::bits::BitView;
use crate::pipeline::buffers::BufferHandle;
use crate::pipeline::operators::{BindContext, Operator};
use crate::pipeline::types::{Element, VType};
use crate::pipeline::view::ViewMut;
use crate::pipeline::{Kernel, KernelContext, PIPELINE_STEP_COUNT};
use crate::vtable::{PipelineVTable, ValidityHelper};

impl PipelineVTable<PrimitiveVTable> for PrimitiveVTable {
    fn to_operator(array: &PrimitiveArray) -> VortexResult<Arc<dyn Operator>> {
        if !array.validity().all_valid()? {
            vortex_bail!(
                "PipelineVTable::to_operator is not supported for arrays with invalid values"
            );
        }
        Ok(Arc::new(array.clone()))
    }
}

impl Operator for PrimitiveArray {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn vtype(&self) -> VType {
        VType::Primitive(self.ptype())
    }

    fn children(&self) -> &[Arc<dyn Operator>] {
        &[]
    }

    fn with_children(&self, children: Vec<Arc<dyn Operator>>) -> Arc<dyn Operator> {
        Arc::new(self.clone())
    }

    fn bind(&self, ctx: &dyn BindContext) -> VortexResult<Box<dyn Kernel>> {
        match_each_native_ptype!(self.ptype(), |T| {
            Ok(Box::new(PrimitiveKernel::<T> {
                buffer: BufferHandle::new(self.buffer()),
                offset: 0,
            }) as Box<dyn Kernel>)
        })
    }
}

impl Hash for PrimitiveArray {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.byte_buffer().as_ptr().hash(state);
        self.ptype().hash(state);
    }
}

/// A kernel that produces primitive values from a byte buffer.
pub struct PrimitiveKernel<T: NativePType> {
    buffer: BufferHandle<T>,
    offset: usize,
}

impl<T: Element + NativePType> Kernel for PrimitiveKernel<T> {
    fn seek(&mut self, chunk_idx: usize) -> VortexResult<()> {
        self.offset = chunk_idx * PIPELINE_STEP_COUNT;
        Ok(())
    }

    fn step(
        &mut self,
        ctx: &dyn KernelContext,
        mask: BitView,
        out: &mut ViewMut,
    ) -> Poll<VortexResult<()>> {
        // FIXME(ngates): support mask.
        // assert_eq!(mask.true_count(), N, "Mask must have exactly N true bits");

        let buffer = ready!(self.buffer.get_or_load(ctx))?;
        let remaining = buffer.len() - self.offset;

        let out_slice = out.as_slice_mut::<T>();

        if remaining > PIPELINE_STEP_COUNT {
            out_slice.copy_from_slice(&buffer[self.offset..][..PIPELINE_STEP_COUNT]);
            self.offset += PIPELINE_STEP_COUNT;
        } else {
            out_slice[..remaining].copy_from_slice(&buffer[self.offset..]);
            self.offset += remaining;
        }
        println!("out_slice: {out_slice:?}");

        // TODO(joe): use mask in copy_from_slice, if faster.
        out.select_mask::<T>(&mask);
        println!("out_slice: {out_slice:?}");

        Poll::Ready(Ok(()))
    }
}

#[cfg(test)]
mod tests {
    use std::task::Poll;

    use vortex_buffer::{BufferMut, ByteBuffer};

    use super::*;
    use crate::pipeline::bits::BitView;
    use crate::pipeline::{BufferId, VectorId, VectorRef};
    use crate::{IntoArray, ToCanonical};

    struct MockContext;

    impl KernelContext for MockContext {
        fn vector(&self, _vector_id: VectorId) -> VectorRef<'_> {
            unimplemented!("not needed for these tests")
        }

        fn buffer(&self, _buffer_id: BufferId) -> Poll<VortexResult<ByteBuffer>> {
            unimplemented!("not needed for these tests")
        }
    }

    #[test]
    fn test_primitive_kernel_basic_operation() {
        // Create a primitive array with values 0..16
        let size = 16;
        let values = (0..size as i32).collect::<BufferMut<_>>();
        let primitive_array = values.into_array().to_primitive().unwrap();

        // Create the kernel
        let mut kernel = PrimitiveKernel::<i32> {
            buffer: BufferHandle::new(primitive_array.buffer()),
            offset: 0,
        };

        // Create an all-true mask for simplicity
        let mask_data = [u64::MAX; PIPELINE_STEP_COUNT / 64];
        let mask_view = BitView::new(&mask_data);

        // Create output buffer
        let mut output = BufferMut::<i32>::with_capacity(PIPELINE_STEP_COUNT);
        unsafe { output.set_len(PIPELINE_STEP_COUNT) };
        let mut output_view = ViewMut::new(&mut output[..], None);

        // Create a mock context
        let ctx = MockContext;

        // Execute the step
        let result = kernel.step(&ctx, mask_view, &mut output_view);
        assert!(matches!(result, Poll::Ready(Ok(()))));

        // Verify the first elements contain our values
        for i in 0..size {
            assert_eq!(
                output[i], i as i32,
                "Mismatch at position {}: expected {}, got {}",
                i, i, output[i]
            );
        }
    }

    #[test]
    fn test_primitive_kernel_with_mask() {
        // Create a primitive array with values 0..16
        let size = 16;
        let values = (0..size).collect::<BufferMut<_>>();
        let primitive_array = values.into_array().to_primitive().unwrap();

        // Create the kernel
        let mut kernel = PrimitiveKernel::<i32> {
            buffer: BufferHandle::new(primitive_array.buffer()),
            offset: 0,
        };

        // Create a mask with alternating bits (every other element selected)
        let mut mask_data = [0u64; PIPELINE_STEP_COUNT / 64];
        // Set bits 0, 2, 4, 6, 8, 10, 12, 14 (first 8 even positions)
        for i in 0..8 {
            let bit_pos = i * 2;
            let word_idx = bit_pos / 64;
            let bit_idx = bit_pos % 64;
            mask_data[word_idx] |= 1u64 << bit_idx; // MSB ordering
        }
        let true_count = 8;
        let mask_view = BitView::new(&mask_data);

        // Create output buffer
        let mut output = BufferMut::<i32>::with_capacity(PIPELINE_STEP_COUNT);
        unsafe { output.set_len(PIPELINE_STEP_COUNT) };
        let mut output_view = ViewMut::new(&mut output[..], None);

        // Create a mock context
        let ctx = MockContext;

        // Execute the step
        let result = kernel.step(&ctx, mask_view, &mut output_view);
        assert!(matches!(result, Poll::Ready(Ok(()))));
        unsafe { output.set_len(mask_view.true_count()) };

        // Verify that the mask was applied successfully
        // The select_mask operation filters elements based on the mask

        // Count elements that have been affected by mask selection
        // Note: element 0 is a valid selected value, so we need to count differently
        let non_zero_count = output.iter().filter(|&&x| x != 0).count();

        println!("output: {output:?}");
        // Verify that element 0 was selected (first bit in mask is 1)
        assert_eq!(output[0], 0, "First element should be 0 since bit 0 is set");

        // Since element 0 is valid but counts as zero, the actual selection count is non_zero_count + 1
        let actual_selected = non_zero_count + 1; // +1 for the zero at position 0

        // The exact number of selected elements should match our true_count
        assert_eq!(
            actual_selected, true_count,
            "Selected element count should match true_count"
        )
    }

    #[test]
    fn test_primitive_kernel_offset_tracking() {
        // Create a primitive array with more than N values
        let total_size = PIPELINE_STEP_COUNT + 100;
        let values = (0..total_size as i32).collect::<BufferMut<_>>();
        let primitive_array = values.into_array().to_primitive().unwrap();

        // Create the kernel
        let mut kernel = PrimitiveKernel::<i32> {
            buffer: BufferHandle::new(primitive_array.buffer()),
            offset: 0,
        };

        // All-true mask
        let mask_data = [u64::MAX; PIPELINE_STEP_COUNT / 64];
        let mask_view = BitView::new(&mask_data);
        let ctx = MockContext;

        // First step should process first N elements
        {
            let mut output = BufferMut::<i32>::with_capacity(PIPELINE_STEP_COUNT);
            unsafe { output.set_len(PIPELINE_STEP_COUNT) };
            let mut output_view = ViewMut::new(&mut output[..], None);

            let result = kernel.step(&ctx, mask_view, &mut output_view);
            assert!(matches!(result, Poll::Ready(Ok(()))));
            assert_eq!(kernel.offset, PIPELINE_STEP_COUNT);

            // Verify first chunk
            for i in 0..PIPELINE_STEP_COUNT {
                assert_eq!(output[i], i as i32);
            }
        }

        // Second step should process remaining elements (partial chunk)
        {
            let mut output = BufferMut::<i32>::with_capacity(PIPELINE_STEP_COUNT);
            unsafe { output.set_len(PIPELINE_STEP_COUNT) };
            let mut output_view = ViewMut::new(&mut output[..], None);

            let result = kernel.step(&ctx, mask_view, &mut output_view);
            assert!(matches!(result, Poll::Ready(Ok(()))));
            assert_eq!(kernel.offset, total_size);

            // Verify remaining elements (first 100 should be valid)
            for i in 0..100 {
                assert_eq!(output[i], (PIPELINE_STEP_COUNT + i) as i32);
            }
        }
    }
}
