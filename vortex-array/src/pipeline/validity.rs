use vortex_buffer::BitBuffer;
use vortex_compute::filter::{Filter, MaskIndices};
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_vector::{VectorMut, VectorMutOps};

use crate::pipeline::{BitView, Kernel, KernelCtx};

/// `ValidityKernel` wraps a child kernel, passing a validity mask through to the output.
pub enum ValidityKernel<K> {
    AllValid(AllValidKernel<K>),
    AllInvalid(AllInvalidKernel<K>),
    Array(ArrayKernel<K>),
    // child: K,
    // validity: BitBuffer,
    // position: usize,
}

impl<K> ValidityKernel<K> {
    pub fn new(inner: K, validity: Mask) -> Self {
        match validity {
            Mask::AllTrue(_) => Self::AllValid(AllValidKernel { child: inner }),
            Mask::AllFalse(_) => Self::AllInvalid(AllInvalidKernel { child: inner }),
            Mask::Values(values) => Self::Array(ArrayKernel {
                child: inner,
                bits: values.bit_buffer().clone(),
                position: 0,
            }),
        }
    }
}

struct AllValidKernel<K> {
    child: K,
}

impl<K: Kernel> Kernel for AllValidKernel<K> {
    fn step(
        &mut self,
        ctx: &KernelCtx,
        selection: &BitView,
        out: &mut VectorMut,
    ) -> VortexResult<()> {
        // execute child kernel
        self.child.step(ctx, selection, out)?;

        let n_true = out.len();
        unsafe { out.validity_mut().append_n(true, n_true) };

        Ok(())
    }
}

struct AllInvalidKernel<K> {
    child: K,
}

impl<K: Kernel> Kernel for AllInvalidKernel<K> {
    fn step(
        &mut self,
        ctx: &KernelCtx,
        selection: &BitView,
        out: &mut VectorMut,
    ) -> VortexResult<()> {
        // execute child kernel
        self.child.step(ctx, selection, out)?;

        let n_true = out.len();
        unsafe { out.validity_mut().append_n(false, n_true) };

        Ok(())
    }
}

struct ArrayKernel<K> {
    child: K,
    bits: BitBuffer,
    position: usize,
}

impl<K: Kernel> Kernel for ArrayKernel<K> {
    fn step(
        &mut self,
        ctx: &KernelCtx,
        selection: &BitView,
        out: &mut VectorMut,
    ) -> VortexResult<()> {
        // execute the child kernel
        self.child.step(ctx, selection, out)?;

        debug_assert_eq!(
            out.validity().len(),
            self.position,
            "child kernel should not step validity when wrapped with ValidityKernel"
        );

        let new_position = self.position + out.len();

        // TODO: plug this in once Filter<BitView> gets added
        // let slice = self.validity.slice(self.position..new_position).filter(selection);
        let slice: Mask = todo!();

        // SAFETY: the child kernel must extend elements in its step function.
        unsafe { out.validity_mut().append_mask(&slice) };

        // Advance the position in the kernel here.
        self.position = new_position;

        Ok(())
    }
}
