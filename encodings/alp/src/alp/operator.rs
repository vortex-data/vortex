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

        let encoded = ctx.input(self.encoded_vector_id);
        let encoded_vec = encoded.as_primitive().downcast::<A::ALPInt>();
        let encoded_buf = encoded_vec.elements().as_slice();

        let decoded = out.as_primitive_mut().downcast::<A>();

        // If our input is in-place, and we have only a few selected elements, then iterate only
        // the selected elements and write them to the output.
        if encoded_buf.len() == N && selection.true_count() < (N / 8) {
            // Reserve capacity for the true_count elements.
            decoded.reserve(selection.true_count().saturating_sub(decoded.capacity()));

            // SAFETY: we set_len and append_validity ensuring elements len matches validity len.
            unsafe { decoded.validity_mut() }.append_n(true, selection.true_count());
            unsafe { decoded.elements_mut().set_len(selection.true_count()) };

            // SAFETY: we reserved capacity above.
            let elements = unsafe { decoded.elements_mut() };

            let mut out_pos = 0;
            selection.iter_ones(|idx| {
                let encoded = unsafe { encoded_buf.get_unchecked(idx) };
                let decoded_value = A::decode_single(*encoded, self.exponents);
                unsafe { *elements.get_unchecked_mut(out_pos) = decoded_value };
                out_pos += 1;
            });

            debug_assert_eq!(decoded.validity().len(), decoded.elements().len());
            return Ok(());
        }

        // Otherwise, iterate the entire input.
        decoded.extend(
            encoded_buf
                .iter()
                .map(|e| A::decode_single(*e, self.exponents)),
        );
        Ok(())
    }
}
