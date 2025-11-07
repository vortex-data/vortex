// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::{MaskedVTable, PrimitiveArray, PrimitiveVTable};
use crate::execution::{kernel, BatchKernelRef, BindCtx};
use crate::pipeline::bit_view::BitView;
use crate::pipeline::{BindContext, KernelContext, PipelinedSource, SourceKernel, N};
use crate::vtable::{OperatorVTable, ValidityHelper};
use crate::{ArrayRef, IntoArray};
use vortex_buffer::Buffer;
use vortex_compute::filter::Filter;
use vortex_dtype::{match_each_native_ptype, NativePType, PTypeDowncastExt};
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_vector::primitive::PVector;
use vortex_vector::VectorMut;

impl OperatorVTable<PrimitiveVTable> for PrimitiveVTable {
    fn as_pipelined_source(array: &PrimitiveArray) -> Option<&dyn PipelinedSource> {
        Some(array)
    }

    fn bind(
        array: &PrimitiveArray,
        selection: Option<&ArrayRef>,
        ctx: &mut dyn BindCtx,
    ) -> VortexResult<BatchKernelRef> {
        let mask = ctx.bind_selection(array.len(), selection)?;
        let validity = ctx.bind_validity(array.validity(), array.len(), selection)?;

        match_each_native_ptype!(array.ptype(), |P| {
            let elements = array.buffer::<P>();
            Ok(kernel(move || {
                let mask = mask.execute()?;
                let validity = validity.execute()?;

                // Note that validity already has the mask applied so we only need to apply it to
                // the elements.
                let elements = elements.filter(&mask);

                Ok(PVector::<P>::try_new(elements, validity)?.into())
            }))
        })
    }

    fn reduce_parent(
        array: &PrimitiveArray,
        parent: &ArrayRef,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        // Push-down masking of `validity` from the parent `MaskedArray`.
        if let Some(masked) = parent.as_opt::<MaskedVTable>() {
            let masked_array = match_each_native_ptype!(array.ptype(), |T| {
                // SAFETY: Since we are only flipping some bits in the validity, all invariants that
                // were upheld are still upheld.
                unsafe {
                    PrimitiveArray::new_unchecked(
                        Buffer::<T>::from_byte_buffer(array.byte_buffer().clone()),
                        array.validity().clone().and(masked.validity().clone()),
                    )
                }
                .into_array()
            });

            return Ok(Some(masked_array));
        }

        Ok(None)
    }
}

impl PipelinedSource for PrimitiveArray {
    fn bind_source(&self, _ctx: &mut dyn BindContext) -> VortexResult<Box<dyn SourceKernel>> {
        match_each_native_ptype!(self.ptype(), |T| {
            let primitive_kernel = PrimitiveKernel {
                buffer: self.buffer::<T>().clone(),
                validity: self.validity_mask(),
                offset: 0,
            };
            Ok(Box::new(primitive_kernel))
        })
    }
}

struct PrimitiveKernel<T: NativePType> {
    buffer: Buffer<T>,
    validity: Mask,
    offset: usize,
}

impl<T: NativePType> SourceKernel for PrimitiveKernel<T> {
    fn skip(&mut self, n: usize) {
        self.offset += n * N;
    }

    fn step(
        &mut self,
        _ctx: &KernelContext,
        selection: &BitView,
        out: &mut VectorMut,
    ) -> VortexResult<()> {
        let out = out.as_primitive_mut().downcast::<T>();

        // SAFETY: we know the output has sufficient capacity. We just have to append nulls
        //  separately from copying over the elements.
        unsafe {
            out.validity_mut().append_n(true, selection.true_count());
            out.elements_mut().set_len(selection.true_count());
        }

        let source = &self.buffer.as_slice()[self.offset..];

        let mut out_pos = 0;
        selection.iter_slices(|(start, end)| {
            print!("Slicing {} to {}\n", start, end);
            let len = end - start;
            out.as_mut()[out_pos..][..len].copy_from_slice(&source[start..end]);
            out_pos += len;
        });

        Ok(())
    }
}
