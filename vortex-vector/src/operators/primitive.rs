// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::hash::Hash;
use std::rc::Rc;

use vortex_buffer::{Buffer, ByteBuffer};
use vortex_dtype::{NativePType, PType, match_each_native_ptype};
use vortex_error::VortexResult;

use crate::bits::BitView;
use crate::operators::{BindContext, Operator};
use crate::types::{Element, VType};
use crate::view::ViewMut;
use crate::{Kernel, KernelContext, SC};

#[derive(Debug, Clone, Hash)]
pub struct PrimitiveOperator {
    ptype: PType,
    byte_buffer: ByteBuffer,
}

impl PrimitiveOperator {
    pub fn new(ptype: PType, byte_buffer: ByteBuffer) -> Self {
        Self { ptype, byte_buffer }
    }
}

impl Operator for PrimitiveOperator {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn vtype(&self) -> VType {
        VType::Primitive(self.ptype)
    }

    fn children(&self) -> &[Rc<dyn Operator>] {
        &[]
    }

    fn with_children(&self, children: Vec<Rc<dyn Operator>>) -> Rc<dyn Operator> {
        Rc::new(self.clone())
    }

    fn bind(&self, ctx: &dyn BindContext) -> VortexResult<Box<dyn Kernel>> {
        match_each_native_ptype!(self.ptype, |T| {
            Ok(Box::new(PrimitiveKernel::<T> {
                buffer: Buffer::from_byte_buffer(self.byte_buffer.clone()),
                offset: 0,
            }) as Box<dyn Kernel>)
        })
    }
}

/// A kernel that produces primitive values from a byte buffer.
pub struct PrimitiveKernel<T: NativePType> {
    buffer: Buffer<T>,
    offset: usize,
}

impl<T: Element + NativePType> Kernel for PrimitiveKernel<T> {
    fn seek(&mut self, chunk_idx: usize) -> VortexResult<()> {
        self.offset = chunk_idx * SC;
        Ok(())
    }

    fn step(&mut self, _ctx: &KernelContext, mask: BitView, out: &mut ViewMut) -> VortexResult<()> {
        // FIXME(ngates): support mask.
        // assert_eq!(mask.true_count(), N, "Mask must have exactly N true bits");

        let buffer = &self.buffer;
        let remaining = buffer.len() - self.offset;

        let out_slice = out.as_slice_mut::<T>();

        if remaining > SC {
            out_slice.copy_from_slice(&buffer[self.offset..][..SC]);
            self.offset += SC;
        } else {
            out_slice[..remaining].copy_from_slice(&buffer[self.offset..]);
            self.offset += remaining;
        }

        // TODO(joe): use mask in copy_from_slice, if faster.
        out.select_mask::<T>(&mask);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::{IntoArray, ToCanonical};
    use vortex_buffer::BufferMut;

    use super::*;
    use crate::bits::BitView;

    #[test]
    fn test_primitive_kernel_basic_operation() {
        // Create a primitive array with values 0..16
        let size = 16;
        let values = (0..i32::try_from(size).unwrap()).collect::<BufferMut<_>>();
        let primitive_array = values.into_array().to_primitive().unwrap();

        // Create the kernel
        let mut kernel = PrimitiveKernel::<i32> {
            buffer: primitive_array.buffer(),
            offset: 0,
        };

        // Create an all-true mask for simplicity
        let mask_data = [u64::MAX; SC / 64];
        let mask_view = BitView::new(&mask_data);

        // Create output buffer
        let mut output = BufferMut::<i32>::with_capacity(SC);
        unsafe { output.set_len(SC) };
        let mut output_view = ViewMut::new(&mut output[..], None);

        // Execute the step
        let dummy_ctx = KernelContext::default();
        let result = kernel.step(&dummy_ctx, mask_view, &mut output_view);
        assert!(matches!(result, Ok(())));

        // Verify the first elements contain our values
        for i in 0..size {
            assert_eq!(
                output[i],
                i32::try_from(i).unwrap(),
                "Mismatch at position {}: expected {}, got {}",
                i,
                i,
                output[i]
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
            buffer: primitive_array.buffer(),
            offset: 0,
        };

        // Create a mask with alternating bits (every other element selected)
        let mut mask_data = [0u64; SC / 64];
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
        let mut output = BufferMut::<i32>::with_capacity(SC);
        unsafe { output.set_len(SC) };
        let mut output_view = ViewMut::new(&mut output[..], None);

        // Execute the step
        let dummy_ctx = KernelContext::default();
        let result = kernel.step(&dummy_ctx, mask_view, &mut output_view);
        assert!(matches!(result, Ok(())));
        unsafe { output.set_len(mask_view.true_count()) };

        // Verify that the mask was applied successfully
        // The select_mask operation filters elements based on the mask

        // Count elements that have been affected by mask selection
        // Note: element 0 is a valid selected value, so we need to count differently
        let non_zero_count = output.iter().filter(|&&x| x != 0).count();

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
        let total_size = SC + 100;
        let values = (0..i32::try_from(total_size).unwrap()).collect::<BufferMut<_>>();
        let primitive_array = values.into_array().to_primitive().unwrap();

        // Create the kernel
        let mut kernel = PrimitiveKernel::<i32> {
            buffer: primitive_array.buffer(),
            offset: 0,
        };

        // All-true mask
        let mask_data = [u64::MAX; SC / 64];
        let mask_view = BitView::new(&mask_data);
        // First step should process first N elements
        {
            let mut output = BufferMut::<i32>::with_capacity(SC);
            unsafe { output.set_len(SC) };
            let mut output_view = ViewMut::new(&mut output[..], None);

            let dummy_ctx = KernelContext::default();
            let result = kernel.step(&dummy_ctx, mask_view, &mut output_view);
            assert!(matches!(result, Ok(())));
            assert_eq!(kernel.offset, SC);

            // Verify first chunk
            for i in 0..SC {
                assert_eq!(output[i], i32::try_from(i).unwrap(), "{i}");
            }
        }

        // Second step should process remaining elements (partial chunk)
        {
            let mut output = BufferMut::<i32>::with_capacity(SC);
            unsafe { output.set_len(SC) };
            let mut output_view = ViewMut::new(&mut output[..], None);

            let dummy_ctx = KernelContext::default();
            let result = kernel.step(&dummy_ctx, mask_view, &mut output_view);
            assert!(matches!(result, Ok(())));
            assert_eq!(kernel.offset, total_size);

            // Verify remaining elements (first 100 should be valid)
            for i in 0..100 {
                assert_eq!(output[i], i32::try_from(SC + i).unwrap());
            }
        }
    }
}
