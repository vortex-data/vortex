// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::{BitBuffer, Buffer};
use vortex_compute::filter::Filter;
use vortex_dtype::{NativePType, PTypeDowncastExt, match_each_native_ptype};
use vortex_error::VortexResult;
use vortex_vector::primitive::PVector;
use vortex_vector::{VectorMut, VectorMutOps};

use crate::arrays::{MaskedVTable, PrimitiveArray, PrimitiveVTable};
use crate::execution::{BatchKernelRef, BindCtx, kernel};
use crate::pipeline::bit_view::{BitSlice, BitView};
use crate::pipeline::{
    AllNullSourceKernel, BindContext, KernelContext, N, PipelinedSource, SourceKernel,
};
use crate::validity::Validity;
use crate::vtable::{OperatorVTable, ValidityHelper};
use crate::{ArrayRef, IntoArray};

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
    fn bind_source(&self, ctx: &mut dyn BindContext) -> VortexResult<Box<dyn SourceKernel>> {
        match self.validity() {
            Validity::NonNullable | Validity::AllValid => {
                match_each_native_ptype!(self.ptype(), |T| {
                    let primitive_kernel = NonNullablePrimitiveKernel {
                        buffer: self.buffer::<T>(),
                        offset: 0,
                    };
                    Ok(Box::new(primitive_kernel))
                })
            }
            Validity::AllInvalid => Ok(Box::new(AllNullSourceKernel)),
            Validity::Array(_) => {
                let validity = ctx.batch_input(0).into_bool();
                // Validity is non-nullable, so we extract the inner bit buffer.
                let (validity, _) = validity.into_parts();

                match_each_native_ptype!(self.ptype(), |T| {
                    let primitive_kernel = NullablePrimitiveKernel {
                        buffer: self.buffer::<T>(),
                        validity,
                        offset: 0,
                    };
                    Ok(Box::new(primitive_kernel))
                })
            }
        }
    }
}

struct NonNullablePrimitiveKernel<T: NativePType> {
    buffer: Buffer<T>,
    offset: usize,
}

impl<T: NativePType> SourceKernel for NonNullablePrimitiveKernel<T> {
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

        // SAFETY: we know the output has sufficient capacity.
        unsafe {
            out.validity_mut().append_n(true, selection.true_count());
            let prev_len = out.len();
            out.elements_mut()
                .set_len(prev_len + selection.true_count());
        }

        let source = &self.buffer.as_slice()[self.offset..];
        let mut out_pos = 0;
        selection.iter_slices(|BitSlice { start, len }| {
            out.as_mut()[out_pos..][..len].copy_from_slice(&source[start..][..len]);
            out_pos += len;
        });

        Ok(())
    }
}

struct NullablePrimitiveKernel<T: NativePType> {
    buffer: Buffer<T>,
    #[allow(dead_code)] // TODO(ngates): implement appending validity bits
    validity: BitBuffer,
    offset: usize,
}

impl<T: NativePType> SourceKernel for NullablePrimitiveKernel<T> {
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
            let prev_len = out.len();
            out.elements_mut()
                .set_len(prev_len + selection.true_count());
        }

        let source = &self.buffer.as_slice()[self.offset..];

        let mut out_pos = 0;
        selection.iter_slices(|BitSlice { start, len }| {
            // Copy over the elements.
            out.as_mut()[out_pos..][..len].copy_from_slice(&source[start..][..len]);
            out_pos += len;

            // Append the validity bits.
            let _validity = unsafe { out.validity_mut() };
            todo!("Append validity bits correctly and optimally!");
        });

        Ok(())
    }
}
