use vortex_array::ContextRef;
use vortex_error::{vortex_panic, VortexResult};

use crate::layouts::struct_::StructLayout;
use crate::scanner::{LayoutScan, Poll, Scan, Scanner};
use crate::segments::SegmentReader;
use crate::{LayoutData, LayoutEncoding, RowMask};

pub struct StructScan {
    layout: LayoutData,
    scan: Scan,
}

impl StructScan {
    pub(super) fn new(layout: LayoutData, scan: Scan, _ctx: ContextRef) -> Self {
        if layout.encoding().id() != StructLayout.id() {
            vortex_panic!("Mismatched layout ID")
        }
        // This is where we need to do some complex things with the scan in order to split it into
        // different scans for different fields.
        Self { layout, scan }
    }
}
impl LayoutScan for StructScan {
    fn scanner(&self, mask: RowMask) -> VortexResult<Box<dyn Scanner>> {
        Ok(Box::new(StructScanner {
            layout: self.layout.clone(),
            scan: self.scan.clone(),
            mask,
            state: State::Initial,
        }) as _)
    }
}

#[derive(Clone)]
enum State {
    Initial,
}

struct StructScanner {
    layout: LayoutData,
    scan: Scan,
    mask: RowMask,
    state: State,
}

impl Scanner for StructScanner {
    fn poll(&mut self, _segments: &dyn SegmentReader) -> VortexResult<Poll> {
        todo!()
    }
}
