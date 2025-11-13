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
use vortex_vector::primitive::PVectorMut;
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

        // Downcast our input/output vectors
        let encoded_vec = ctx
            .input(self.encoded_vector_id)
            .as_primitive()
            .downcast::<A::ALPInt>();
        let decoded_vec = out.as_primitive_mut().downcast::<A>();

        // If our input is in-place, and we have only a few selected elements, then iterate only
        // the selected elements and write them to the output.
        if encoded_vec.len() == N && selection.true_count() < (N / 8) {
            sparse_alp(
                decoded_vec,
                encoded_vec.elements().as_slice(),
                self.exponents,
                selection,
            )
        }

        // Otherwise, we have to decode the entire vector.
        decoded_vec.reserve(encoded_vec.len());

        // Copy over the validity from the input vector.
        unsafe {
            decoded_vec
                .validity_mut()
                .append_mask_mut(encoded_vec.validity())
        };
        // And set_len on the elements to match.
        unsafe { decoded_vec.elements_mut().set_len(encoded_vec.len()) };

        let enc = encoded_vec.elements();
        let dec = unsafe { decoded_vec.elements_mut() };
        for i in 0..encoded_vec.len() {
            let encoded = unsafe { enc.get_unchecked(i) };
            let decoded = unsafe { dec.get_unchecked_mut(i) };
            *decoded = A::decode_single(*encoded, self.exponents)
        }

        Ok(())
    }
}

#[inline(never)]
fn sparse_alp<A: ALPFloat>(
    decoded_vec: &mut PVectorMut<A>,
    encoded: &[A::ALPInt],
    exponents: Exponents,
    selection: &BitView,
) {
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
        let element = A::decode_single(*encoded, exponents);
        unsafe { *decoded.get_unchecked_mut(out_pos) = element };
        out_pos += 1;
    });

    debug_assert_eq!(decoded_vec.validity().len(), decoded_vec.elements().len());
}

#[cfg(test)]
mod test {
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::validity::Validity;
    use vortex_array::IntoArray;
    use vortex_buffer::buffer;
    use vortex_dtype::PTypeDowncastExt;

    use crate::alp_encode;

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
