use serde::{Deserialize, Serialize};
use vortex_array::{Array, ArrayChildVisitor, ArrayVisitorImpl, SerdeMetadata};
use vortex_dtype::PType;
use vortex_error::VortexExpect;

use crate::RunEndArray;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunEndMetadata {
    pub(crate) ends_ptype: PType,
    pub(crate) num_runs: usize,
    pub(crate) offset: usize,
}

impl ArrayVisitorImpl<SerdeMetadata<RunEndMetadata>> for RunEndArray {
    fn _children(&self, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("ends", self.ends());
        visitor.visit_child("values", self.values());
    }

    fn _metadata(&self) -> SerdeMetadata<RunEndMetadata> {
        SerdeMetadata(RunEndMetadata {
            ends_ptype: PType::try_from(self.ends().dtype()).vortex_expect("Must be a valid PType"),
            num_runs: self.ends().len(),
            offset: self.offset(),
        })
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::SerdeMetadata;
    use vortex_array::test_harness::check_metadata;
    use vortex_dtype::PType;

    use super::*;

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_runend_metadata() {
        check_metadata(
            "runend.metadata",
            SerdeMetadata(RunEndMetadata {
                offset: usize::MAX,
                ends_ptype: PType::U64,
                num_runs: usize::MAX,
            }),
        );
    }
}
