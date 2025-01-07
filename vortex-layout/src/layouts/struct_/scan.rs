use std::sync::Arc;

use vortex_array::{ArrayData, ContextRef};
use vortex_dtype::DType;
use vortex_error::{vortex_panic, VortexResult};

use crate::layouts::struct_::StructLayout;
use crate::operations::{Operation, Poll};
use crate::scanner::{LayoutScan, Scan, ScanOp};
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

    fn create_scanner(self: Arc<Self>, mask: RowMask) -> VortexResult<ScanOp> {
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
#[allow(dead_code)]
struct StructScanner {
    layout: LayoutData,
    scan: Scan,
    mask: RowMask,
    state: State,
}

impl Operation for StructScanner {
    type Output = ArrayData;

    fn poll(&mut self, _segments: &dyn SegmentReader) -> VortexResult<Poll<Self::Output>> {
        todo!()
    }
}
