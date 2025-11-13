// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::marker::PhantomData;
use vortex_array::pipeline::bit_view::BitView;
use vortex_array::pipeline::{
    BindContext, Kernel, KernelCtx, PipelineInputs, PipelinedNode, VectorId, N,
};
use vortex_array::vtable::OperatorVTable;
use vortex_dtype::PTypeDowncastExt;
use vortex_error::{vortex_bail, VortexResult};
use vortex_vector::{VectorMut, VectorMutOps};

use crate::{match_each_alp_float_ptype, ALPArray, ALPFloat, ALPVTable, Exponents};

impl OperatorVTable<ALPVTable> for ALPVTable {
    fn pipeline_node(array: &ALPArray) -> Option<&dyn PipelinedNode> {
        Some(array)
    }
}

impl PipelinedNode for ALPArray {
    fn inputs(&self) -> PipelineInputs {
        PipelineInputs::Transform {
            pipelined_inputs: vec![0],
        }
    }

    fn bind(&self, ctx: &dyn BindContext) -> VortexResult<Box<dyn Kernel>> {
        let encoded_vector_id = ctx.pipelined_input(0);
        match self.patches() {
            Some(_) => vortex_bail!("patched ALP kernel not implemented",),
            None => {
                if !self.all_valid() {
                    vortex_bail!("ALP kernel does not yet handle nulls",);
                }
                match_each_alp_float_ptype!(self.ptype(), |A| {
                    Ok(Box::new(UnpatchedALPKernel {
                        encoded_vector_id,
                        exponents: self.exponents(),
                        _phantom: PhantomData::<A>,
                    }))
                })
            }
        }
    }
}

struct UnpatchedALPKernel<A> {
    encoded_vector_id: VectorId,
    exponents: Exponents,
    _phantom: PhantomData<A>,
}

impl<A: ALPFloat> Kernel for UnpatchedALPKernel<A> {
    fn step(
        &mut self,
        ctx: &KernelCtx,
        selection: &BitView,
        out: &mut VectorMut,
    ) -> VortexResult<()> {
        if selection.true_count() == 0 {
            // Nothing to do, and no kernel state to update
            return Ok(());
        }

        let encoded_vec = ctx.input(self.encoded_vector_id);
        let encoded = encoded_vec
            .as_primitive()
            .downcast::<A::ALPInt>()
            .elements()
            .as_slice();

        let decoded_vec = out.as_primitive_mut().downcast::<A>();

        // If our input is in-place, and we have only a few selected elements, then iterate only
        // the selected elements and write them to the output.
        if encoded.len() == N && selection.true_count() < (N / 8) {
            // Reserve capacity for the true_count elements.
            decoded_vec.reserve(
                selection
                    .true_count()
                    .saturating_sub(decoded_vec.capacity()),
            );

            // SAFETY: we set_len and append_validity ensuring elements len matches validity len.
            unsafe { decoded_vec.validity_mut() }.append_n(true, selection.true_count());
            unsafe { decoded_vec.elements_mut().set_len(selection.true_count()) };

            // SAFETY: we reserved capacity above.
            let decoded = unsafe { decoded_vec.elements_mut() };

            let mut out_pos = 0;
            selection.iter_ones(|idx| {
                let encoded = unsafe { encoded.get_unchecked(idx) };
                let element = A::decode_single(*encoded, self.exponents);
                unsafe { *decoded.get_unchecked_mut(out_pos) = element };
                out_pos += 1;
            });

            debug_assert_eq!(decoded_vec.validity().len(), decoded_vec.elements().len());
            return Ok(());
        }

        // Otherwise, iterate the entire input.
        decoded_vec.reserve(N.saturating_sub(decoded_vec.capacity()));
        unsafe { decoded_vec.validity_mut().append_n(true, N) };
        unsafe { decoded_vec.elements_mut().set_len(N) };
        let decoded = unsafe { decoded_vec.elements_mut().as_mut_slice() };

        // By extracting these outside the loop, we auto-vectorize the decoding.
        // I wonder if the regular ALP is actually vectorized?
        let f = A::F10[self.exponents.f as usize];
        let ie = A::IF10[self.exponents.e as usize];
        for idx in 0..N {
            unsafe {
                *decoded.get_unchecked_mut(idx) = A::from_int(*encoded.get_unchecked(idx)) * f * ie;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::alp_encode;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::validity::Validity;
    use vortex_array::{ArrayOperatorExt, IntoArray};
    use vortex_buffer::buffer;
    use vortex_dtype::PTypeDowncastExt;

    #[test]
    fn test_alp_kernel() {
        let buffer = buffer![42.125f32; 10_000];
        let array = PrimitiveArray::new(buffer.clone(), Validity::NonNullable);
        let encoded = alp_encode(&array, None).unwrap().into_array();

        let decoded = encoded
            .execute()
            .unwrap()
            .into_primitive()
            .downcast::<f32>();

        assert_eq!(decoded.elements(), &buffer);
    }
}
