use vortex_array::ContextRef;
use vortex_dtype::DType;
use vortex_error::{vortex_panic, VortexResult};

use crate::layouts::struct_::StructLayout;
use crate::scanner::{LayoutScan, Poll, Scan, Scanner};
use crate::segments::SegmentReader;
use crate::{LayoutData, LayoutEncoding, RowMask};

#[derive(Debug)]
pub struct StructScan {
    layout: LayoutData,
    scan: Scan,
    dtype: DType,
}

impl StructScan {
    pub(super) fn try_new(layout: LayoutData, scan: Scan, _ctx: ContextRef) -> VortexResult<Self> {
        if layout.encoding().id() != StructLayout.id() {
            vortex_panic!("Mismatched layout ID")
        }

        let dtype = scan.result_dtype(layout.dtype())?;

        // This is where we need to do some complex things with the scan in order to split it into
        // different scans for different fields.
        Ok(Self {
            layout,
            scan,
            dtype,
        })
    }
}
impl LayoutScan for StructScan {
    fn layout(&self) -> &LayoutData {
        &self.layout
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn create_scanner(&self, mask: RowMask) -> VortexResult<Box<dyn Scanner>> {
        Ok(Box::new(StructScanner {
            layout: self.layout.clone(),
            scan: self.scan.clone(),
            mask,
            state: State::Initial,
        }) as _)
    }
}

#[derive(Clone, Debug)]
enum State {
    Initial,
}

#[derive(Debug)]
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
