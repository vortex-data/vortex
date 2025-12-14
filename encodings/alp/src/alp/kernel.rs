// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::marker::PhantomData;

use vortex_array::kernel::Kernel;
use vortex_array::kernel::KernelRef;
use vortex_array::kernel::PushDownResult;
use vortex_dtype::PTypeDowncastExt;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_vector::Vector;

use crate::ALPFloat;
use crate::Exponents;
use crate::alp::decompress::decompress_into_vector;

#[derive(Debug)]
pub(super) struct ALPKernel<A: ALPFloat> {
    exponents: Exponents,
    encoded: KernelRef,
    patches: Option<PatchKernels>,
    patches_offset: usize,
    _phantom: PhantomData<A>,
}

impl<A: ALPFloat> ALPKernel<A> {
    pub(super) fn new(
        exponents: Exponents,
        encoded: KernelRef,
        patches: Option<PatchKernels>,
        patches_offset: usize,
    ) -> Self {
        Self {
            exponents,
            encoded,
            patches,
            patches_offset,
            _phantom: PhantomData,
        }
    }
}

#[derive(Debug)]
pub(super) struct PatchKernels {
    pub(super) indices: KernelRef,
    pub(super) values: KernelRef,
    pub(super) chunk_offsets: Option<KernelRef>,
}

impl<A: ALPFloat> Kernel for ALPKernel<A> {
    fn execute(self: Box<Self>) -> VortexResult<Vector> {
        let encoded = self
            .encoded
            .execute()?
            .into_primitive()
            .downcast::<A::ALPInt>();

        let patches_vectors = match self.patches {
            None => None,
            Some(PatchKernels {
                indices,
                values,
                chunk_offsets,
            }) => {
                let indices = indices.execute()?.into_primitive();
                let values = values.execute()?.into_primitive();
                let chunk_offsets = match chunk_offsets {
                    None => None,
                    Some(co) => Some(co.execute()?.into_primitive()),
                };
                Some((indices, values, chunk_offsets))
            }
        };

        decompress_into_vector::<A>(
            encoded,
            self.exponents,
            patches_vectors,
            self.patches_offset,
        )
    }

    fn push_down_filter(self: Box<Self>, _selection: &Mask) -> VortexResult<PushDownResult> {
        Ok(PushDownResult::NotPushed(self))
    }
}
