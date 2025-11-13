// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::marker::PhantomData;

use vortex_array::pipeline::bit_view::BitView;
use vortex_array::pipeline::{
    BindContext, Kernel, KernelCtx, N, PipelineInputs, PipelinedNode, VectorId,
};
use vortex_array::vtable::OperatorVTable;
use vortex_dtype::PTypeDowncastExt;
use vortex_error::{VortexResult, vortex_bail};
use vortex_vector::primitive::PVectorMut;
use vortex_vector::{VectorMut, VectorMutOps};

use crate::{ALPArray, ALPFloat, ALPVTable, Exponents, match_each_alp_float_ptype};

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
            .elements();

        let decoded_vec = out.as_primitive_mut().downcast::<A>();

        // If our input is in-place, and we have only a few selected elements, then iterate only
        // the selected elements and write them to the output.
        if encoded.len() == N && selection.true_count() < (N / 8) {
            sparse_alp(decoded_vec, encoded.as_slice(), self.exponents, selection)
        }

        // If the input is smaller than N, we have to do a traditional loop (vs known loop over N)
        if encoded.len() < N {
            sparse_alp(decoded_vec, encoded.as_slice(), self.exponents, selection)
        }

        debug_assert_eq!(encoded.len(), N);
        debug_assert_eq!(decoded_vec.len(), N);

        // Otherwise, iterate the entire input.
        decoded_vec.reserve(N.saturating_sub(decoded_vec.capacity()));
        unsafe { decoded_vec.validity_mut().append_n(true, N) };
        unsafe { decoded_vec.elements_mut().set_len(N) };

        // Unsafe cast to array (avoiding the cost of constructing slices...)
        let decoded: &mut [A; N] =
            unsafe { &mut *(decoded_vec.elements_mut().as_mut_ptr() as *mut [A; N]) };
        let encoded: &[A::ALPInt; N] = unsafe { &*(encoded.as_ptr() as *const [A::ALPInt; N]) };

        // By extracting these outside the loop, we auto-vectorize the decoding.
        // I wonder if the regular ALP is actually vectorized?
        let f = A::F10[self.exponents.f as usize];
        let ie = A::IF10[self.exponents.e as usize];

        for i in 0..N {
            decoded[i] = A::from_int(encoded[i]) * f * ie;
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
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::validity::Validity;
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
