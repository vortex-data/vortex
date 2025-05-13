use vortex_array::compute::{NaNCountKernel, NaNCountKernelAdapter, nan_count};
use vortex_array::register_kernel;
use vortex_error::VortexResult;

use crate::{ALPArray, ALPVTable};

impl NaNCountKernel for ALPVTable {
    fn nan_count(&self, array: &ALPArray) -> VortexResult<usize> {
        // NANs can only be in patches
        if let Some(patches) = array.patches() {
            nan_count(patches.values())
        } else {
            Ok(0)
        }
    }
}

register_kernel!(NaNCountKernelAdapter(ALPVTable).lift());
