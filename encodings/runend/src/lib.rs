// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub use array::*;
pub use iter::trimmed_ends_iter;

mod array;
#[cfg(feature = "arrow")]
mod arrow;
pub mod compress;
mod compute;
mod iter;
mod ops;

#[doc(hidden)]
pub mod _benchmarking {
    pub use compute::filter::filter_run_end;
    pub use compute::take::take_indices_unchecked;

    use super::*;
}

use vortex_array::vtable::{EncodeVTable, VisitorVTable};
use vortex_array::{ArrayBufferVisitor, ArrayChildVisitor, Canonical};
use vortex_error::VortexResult;

use crate::compress::runend_encode;

impl EncodeVTable<RunEndVTable> for RunEndVTable {
    fn encode(
        _vtable: &RunEndVTable,
        canonical: &Canonical,
        _like: Option<&RunEndArray>,
    ) -> VortexResult<Option<RunEndArray>> {
        let parray = canonical.clone().into_primitive();
        let (ends, values) = runend_encode(&parray);
        // SAFETY: runend_decode implementation must return valid RunEndArray
        //  components.
        unsafe {
            Ok(Some(RunEndArray::new_unchecked(
                ends.to_array(),
                values,
                0,
                parray.len(),
            )))
        }
    }
}

impl VisitorVTable<RunEndVTable> for RunEndVTable {
    fn visit_buffers(_array: &RunEndArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn visit_children(array: &RunEndArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("ends", array.ends());
        visitor.visit_child("values", array.values());
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::ProstMetadata;
    use vortex_array::test_harness::check_metadata;
    use vortex_dtype::PType;

    use crate::RunEndMetadata;

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_runend_metadata() {
        check_metadata(
            "runend.metadata",
            ProstMetadata(RunEndMetadata {
                ends_ptype: PType::U64 as i32,
                num_runs: u64::MAX,
                offset: u64::MAX,
            }),
        );
    }
}
