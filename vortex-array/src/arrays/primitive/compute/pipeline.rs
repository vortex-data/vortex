// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::sync::Arc;

use arrow_buffer::BooleanBuffer;
use log;
use vortex_buffer::{Buffer, ByteBuffer};
use vortex_dtype::{NativePType, PType, match_each_native_ptype};
use vortex_error::{VortexExpect, VortexResult};
use vortex_mask::Mask;

use crate::arrays::{PrimitiveArray, PrimitiveVTable};
use crate::pipeline::bits::BitView;
use crate::pipeline::operators::{BindContext, Operator, OperatorRef};
use crate::pipeline::view::ViewMut;
use crate::pipeline::{Element, Kernel, KernelContext, N, PipelineVTable, VType};
use crate::vtable::ValidityHelper;

impl PipelineVTable<PrimitiveVTable> for PrimitiveVTable {
    fn to_operator(array: &PrimitiveArray) -> VortexResult<Option<OperatorRef>> {
        if !array.validity().all_valid() {
            log::debug!(
                "PipelineVTable::to_operator is not supported for arrays with invalid values"
            );
            return Ok(None);
        }
        Ok(Some(Arc::new(PrimitiveOperator::new(
            array.ptype(),
            array.byte_buffer().clone(),
            array.dtype().is_nullable().then(|| array.validity_mask()),
        ))))
    }
}

/// Pipeline operator for primitive arrays that produces values from a byte buffer.
#[derive(Debug, Clone, Hash)]
pub struct PrimitiveOperator {
    ptype: PType,
    byte_buffer: ByteBuffer,
    mask: Option<Mask>,
}

impl PrimitiveOperator {
    pub fn new(ptype: PType, byte_buffer: ByteBuffer, mask: Option<Mask>) -> Self {
        Self {
            ptype,
            byte_buffer,
            mask,
        }
    }
}

impl Operator for PrimitiveOperator {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn vtype(&self) -> VType {
        VType::Primitive(self.ptype)
    }

    fn children(&self) -> &[OperatorRef] {
        &[]
    }

    fn with_children(&self, _children: Vec<OperatorRef>) -> OperatorRef {
        Arc::new(self.clone())
    }

    fn bind(&self, _ctx: &dyn BindContext) -> VortexResult<Box<dyn Kernel>> {
        if let Some(mask) = &self.mask {
            match_each_native_ptype!(self.ptype, |T| {
                Ok(Box::new(NullablePrimitiveKernel::<T> {
                    buffer: Buffer::from_byte_buffer(self.byte_buffer.clone()),
                    // TODO(joe): opt this.
                    mask: mask.to_boolean_buffer(),
                    offset: 0,
                }) as Box<dyn Kernel>)
            })
        } else {
            match_each_native_ptype!(self.ptype, |T| {
                Ok(Box::new(PrimitiveKernel::<T> {
                    buffer: Buffer::from_byte_buffer(self.byte_buffer.clone()),
                    offset: 0,
                }) as Box<dyn Kernel>)
            })
        }
    }
}

/// A kernel that produces primitive values from a byte buffer.
pub struct PrimitiveKernel<T: NativePType> {
    buffer: Buffer<T>,
    offset: usize,
}

impl<T: Element + NativePType> Kernel for PrimitiveKernel<T> {
    fn seek(&mut self, chunk_idx: usize) -> VortexResult<()> {
        self.offset = chunk_idx * N;
        Ok(())
    }

    fn step(&mut self, _ctx: &KernelContext, mask: BitView, out: &mut ViewMut) -> VortexResult<()> {
        let buffer = &self.buffer;
        let remaining = buffer.len() - self.offset;

        let out_slice = out.as_slice_mut::<T>();

        if remaining >= N {
            out_slice.copy_from_slice(&buffer[self.offset..][..N]);
            self.offset += N;
        } else {
            out_slice[..remaining].copy_from_slice(&buffer[self.offset..]);
            self.offset += remaining;
        }

        // TODO(joe): use mask in copy_from_slice, if faster.
        out.select_mask::<T>(&mask);

        Ok(())
    }
}

/// A kernel that produces primitive values from a byte buffer.
pub struct NullablePrimitiveKernel<T: NativePType> {
    buffer: Buffer<T>,
    mask: BooleanBuffer,
    offset: usize,
}

impl<T: Element + NativePType> Kernel for NullablePrimitiveKernel<T> {
    fn seek(&mut self, chunk_idx: usize) -> VortexResult<()> {
        self.offset = chunk_idx * N;
        Ok(())
    }

    fn step(
        &mut self,
        _ctx: &KernelContext,
        _selected: BitView,
        out: &mut ViewMut,
    ) -> VortexResult<()> {
        let buffer = &self.buffer;
        let remaining = buffer.len() - self.offset;

        let out_slice = out.as_slice_mut::<T>();
        let validity = out.validity();
        let validity_slice = validity.as_raw_mut();

        debug_assert_eq!(N % 8, 0);
        debug_assert_eq!(self.offset % 8, 0);

        if remaining >= N {
            out_slice.copy_from_slice(&buffer[self.offset..][..N]);

            let byte_slice = &self.mask.values()[self.offset / 8..][..N / 8];
            let usize_ptr = byte_slice.as_ptr() as *const usize;
            let usize_slice =
                unsafe { std::slice::from_raw_parts(usize_ptr, N / 8 / size_of::<usize>()) };

            validity_slice.copy_from_slice(usize_slice);
            self.offset += N;
        } else {
            out_slice[..remaining].copy_from_slice(&buffer[self.offset..]);

            let byte_slice = &self.mask.values()[self.offset / 8..][..remaining.div_ceil(8)];
            let usize_ptr = byte_slice.as_ptr() as *const usize;
            let usize_slice = unsafe {
                std::slice::from_raw_parts(
                    usize_ptr,
                    remaining.div_ceil(usize::BITS.try_into().vortex_expect("does fit")),
                )
            };

            validity_slice[..remaining.div_ceil(u32::BITS as usize)].copy_from_slice(usize_slice);

            self.offset += remaining;
        }

        // TODO(joe): use mask in copy_from_slice, if faster.
        // out.select_mask::<T>(&_selected);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use vortex_buffer::BufferMut;
    use vortex_mask::Mask;

    use super::*;
    use crate::pipeline::export_canonical_pipeline;
    use crate::validity::Validity;
    use crate::{IntoArray, ToCanonical};

    #[test]
    fn test_primitive_kernel_basic_operation() {
        // Create a primitive array with values 0..16
        let size = 16;
        let values = (0..i32::try_from(size).unwrap()).collect::<BufferMut<_>>();
        let primitive_array = values.into_array().to_primitive();

        // Create the kernel
        let mut kernel = PrimitiveKernel::<i32> {
            buffer: primitive_array.buffer(),
            offset: 0,
        };

        let out = export_canonical_pipeline(
            primitive_array.dtype(),
            size,
            &mut kernel,
            &Mask::AllTrue(size),
        )
        .unwrap()
        .into_primitive();

        let output = out.as_slice::<i32>();

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
        let primitive_array = (0i32..i32::try_from(size).unwrap()).collect::<PrimitiveArray>();

        // Create the kernel
        let mut kernel = PrimitiveKernel::<i32> {
            buffer: primitive_array.buffer(),
            offset: 0,
        };

        // Create a mask with alternating bits (every other element selected)
        let mask = Mask::from_indices(size, (0..size).step_by(2).collect_vec());
        let out = export_canonical_pipeline(primitive_array.dtype(), size, &mut kernel, &mask)
            .unwrap()
            .into_primitive();

        let output = out.as_slice::<i32>();

        // Verify that element 0 was selected (first bit in mask is 1)
        assert_eq!(output[0], 0, "First element should be 0 since bit 0 is set");

        // The exact number of selected elements should match our true_count
        assert_eq!(
            out.len(),
            size / 2,
            "Selected element count should match true_count"
        )
    }

    #[test]
    fn test_nullable_primitive_kernel() {
        // Create a primitive array with values 0..16
        let size = 16;
        let primitive_array = PrimitiveArray::new(
            (0..i32::try_from(size).unwrap()).collect::<Buffer<i32>>(),
            Validity::from_iter([
                true, false, true, true, false, true, true, false, true, true, false, true, true,
                false, true, false,
            ]),
        );

        // Create the kernel
        let mut kernel = NullablePrimitiveKernel::<i32> {
            buffer: primitive_array.buffer(),
            mask: primitive_array.validity_mask().to_boolean_buffer(),
            offset: 0,
        };

        // Create a mask with alternating bits (every other element selected)
        let mask = Mask::AllTrue(size);
        let out = export_canonical_pipeline(primitive_array.dtype(), size, &mut kernel, &mask)
            .unwrap()
            .into_primitive();

        let output = out.as_slice::<i32>();
        println!("out val {:?}", out.validity.to_mask(size).true_count());
        println!("out {:?}", output);

        // Verify that element 0 was selected (first bit in mask is 1)
        // assert_eq!(output[0], 0, "First element should be 0 since bit 0 is set");

        // The exact number of selected elements should match our true_count
        // assert_eq!(
        //     out.len(),
        //     size / 2,
        //     "Selected element count should match true_count"
        // )
    }
}
