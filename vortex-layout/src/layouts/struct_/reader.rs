use vortex_array::ContextRef;
use vortex_error::{vortex_panic, VortexResult};

use crate::layouts::struct_::StructLayout;
use crate::{LayoutData, LayoutEncoding, LayoutReader};

#[derive(Debug)]
pub struct StructScan {
    layout: LayoutData,
}

impl StructScan {
    pub(super) fn try_new(layout: LayoutData, _ctx: ContextRef) -> VortexResult<Self> {
        if layout.encoding().id() != StructLayout.id() {
            vortex_panic!("Mismatched layout ID")
        }

        // This is where we need to do some complex things with the scan in order to split it into
        // different scans for different fields.
        Ok(Self { layout })
    }
}

impl LayoutReader for StructScan {
    fn layout(&self) -> &LayoutData {
        &self.layout
    }
}
