// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::sync::Arc;

use log;
use vortex_buffer::{Buffer, ByteBuffer};
use vortex_dtype::{NativePType, PType, match_each_native_ptype};
use vortex_error::VortexResult;

use crate::arrays::{PrimitiveArray, PrimitiveVTable};
use crate::pipeline::bits::BitView;
use crate::pipeline::operators::{BindContext, Operator, OperatorRef};
use crate::pipeline::view::ViewMut;
use crate::pipeline::{Element, Kernel, KernelContext, N, PipelineVTable, VType};
use crate::vtable::ValidityHelper;

impl PipelineVTable<PrimitiveVTable> for PrimitiveVTable {
    fn to_operator(array: &PrimitiveArray) -> VortexResult<Option<OperatorRef>> {
        if !array.validity().all_valid(array.len()) {
            log::debug!(
                "PipelineVTable::to_operator is not supported for arrays with invalid values"
            );
            return Ok(None);
        }
        Ok(Some(Arc::new(PrimitiveOperator::new(
            array.ptype(),
            array.byte_buffer().clone(),
        ))))
    }
}

/// Pipeline operator for primitive arrays that produces values from a byte buffer.
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

    fn children(&self) -> &[OperatorRef] {
        &[]
    }

    fn with_children(&self, _children: Vec<OperatorRef>) -> OperatorRef {
        Arc::new(self.clone())
    }

    fn bind(&self, _ctx: &dyn BindContext) -> VortexResult<Box<dyn Kernel>> {
        match_each_native_ptype!(self.ptype, |T| {
            Ok(Box::new(PrimitiveKernel::<T> {
                buffer: Buffer::from_byte_buffer(self.byte_buffer.clone()),
                offset: 0,
            }) as Box<dyn Kernel>)
        })
    }
}

/// A kernel that produces primitive values from a byte buffer.
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

        if remaining > N {
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

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use vortex_buffer::BufferMut;
    use vortex_mask::Mask;

    use super::*;
    use crate::pipeline::export_canonical_pipeline;
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
}
