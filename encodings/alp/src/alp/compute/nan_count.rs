use vortex_array::compute::{NaNCountFn, nan_count};
use vortex_error::VortexResult;

use crate::{ALPArray, ALPEncoding};

impl NaNCountFn<&ALPArray> for ALPEncoding {
    fn nan_count(&self, array: &ALPArray) -> VortexResult<Option<usize>> {
        // NANs can only be in patches
        if let Some(patches) = array.patches() {
            nan_count(patches.values())
        } else {
            Ok(Some(0))
        }
    }
}
