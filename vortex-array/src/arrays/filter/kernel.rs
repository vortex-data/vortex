// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_compute::filter::Filter;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_vector::Vector;

use crate::kernel::Kernel;
use crate::kernel::KernelRef;
use crate::kernel::PushDownResult;

#[derive(Debug)]
pub struct FilterKernel {
    child: KernelRef,
    mask: Mask,
}

impl FilterKernel {
    pub fn new(child: KernelRef, mask: Mask) -> Self {
        Self { child, mask }
    }
}

impl Kernel for FilterKernel {
    fn execute(self: Box<Self>) -> VortexResult<Vector> {
        Ok(Filter::filter(&self.child.execute()?, &self.mask))
    }

    fn push_down_filter(self: Box<Self>, selection: &Mask) -> VortexResult<PushDownResult> {
        let new_mask = self.mask.intersect_by_rank(selection);
        Ok(match self.child.push_down_filter(&new_mask)? {
            PushDownResult::NotPushed(k) => PushDownResult::NotPushed(Box::new(FilterKernel {
                child: k,
                mask: new_mask,
            })),
            PushDownResult::Pushed(new_k) => PushDownResult::Pushed(new_k),
        })
    }
}
