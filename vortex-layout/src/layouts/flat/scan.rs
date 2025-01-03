use vortex_array::compute::{fill_null, filter, FilterMask};
use vortex_array::ContextRef;
use vortex_dtype::DType;
use vortex_error::{vortex_bail, vortex_err, vortex_panic, VortexResult};
use vortex_ipc::messages::{BufMessageReader, DecoderMessage};

use crate::layouts::flat::FlatLayout;
use crate::scanner::{LayoutScan, Poll, Scan, Scanner};
use crate::segments::{SegmentId, SegmentReader};
use crate::{LayoutData, LayoutEncoding, RowMask};

#[derive(Clone, Eq, PartialEq)]
enum State {
    Initial,
}

pub struct FlatScan {
    layout: LayoutData,
    scan: Scan,
    ctx: ContextRef,
    state: State,
}

impl FlatScan {
    pub(super) fn new(layout: LayoutData, scan: Scan, ctx: ContextRef) -> Self {
        if layout.encoding().id() != FlatLayout.id() {
            vortex_panic!("Mismatched layout ID")
        }
        Self {
            layout,
            scan,
            ctx,
            state: State::Initial,
        }
    }
}

impl LayoutScan for FlatScan {
    fn layout(&self) -> &LayoutData {
        &self.layout
    }

    fn masked_scanner(&self, mask: RowMask) -> VortexResult<Box<dyn Scanner>> {
        let segment_id = self
            .layout
            .segment_id(0)
            .ok_or_else(|| vortex_err!("FlatLayout missing SegmentID"))?;

        // Convert the row-mask to a filter mask
        let filter_mask = mask.into_filter_mask()?;

        Ok(Box::new(FlatScanner {
            segment_id,
            dtype: self.layout.dtype().clone(),
            scan: self.scan.clone(),
            ctx: self.ctx.clone(),
            mask: filter_mask,
        }) as _)
    }
}

struct FlatScanner {
    segment_id: SegmentId,
    dtype: DType,
    scan: Scan,
    ctx: ContextRef,
    mask: FilterMask,
}

impl Scanner for FlatScanner {
    fn poll(&mut self, segments: &dyn SegmentReader) -> VortexResult<Poll> {
        match segments.get(self.segment_id) {
            None => Ok(Poll::NeedMore(vec![self.segment_id])),
            Some(bytes) => {
                // Decode the ArrayParts from the message bytes.
                // TODO(ngates): ArrayParts should probably live in vortex-array, and not required
                //  IPC message framing to read or write.
                let mut msg_reader = BufMessageReader::new(bytes);
                let array = if let DecoderMessage::Array(parts) = msg_reader
                    .next()
                    .ok_or_else(|| vortex_err!("Flat message body missing"))??
                {
                    parts.into_array_data(self.ctx.clone(), self.dtype.clone())
                } else {
                    vortex_bail!("Flat message is not ArrayParts")
                }?;

                // TODO(ngates): I think we can pull out a "Scanner" object that encapsulates
                //  clever logic for figuring out the best order to apply the filter, projection,
                //  and filter mask. This can then be re-used across chunks so the selectivity
                //  stats are preserved.

                // Now we can apply the scan to the array.
                // NOTE(ngates): there's not a clear answer for order to apply the filter
                // expression, projection and filter mask. We should experiment.
                let mut array = filter(&array, self.mask.clone())?;
                if let Some(expr) = &self.scan.filter {
                    array = expr.evaluate(&array)?;
                    array = fill_null(array, false.into())?;
                }
                array = self.scan.projection.evaluate(&array)?;

                Ok(Poll::Some(array))
            }
        }
    }
}

#[cfg(test)]
mod test {
    use vortex_array::array::PrimitiveArray;
    use vortex_array::validity::Validity;
    use vortex_array::{ArrayDType, IntoArrayVariant, ToArrayData};
    use vortex_buffer::buffer;

    use crate::layouts::flat::writer::FlatLayoutWriter;
    use crate::scanner::Scan;
    use crate::segments::test::TestSegments;
    use crate::strategies::LayoutWriterExt;

    #[test]
    fn flat_scan() {
        let mut segments = TestSegments::default();
        let array = PrimitiveArray::new(buffer![1, 2, 3, 4, 5], Validity::AllValid);
        let layout = FlatLayoutWriter::new(array.dtype().clone())
            .push_one(&mut segments, array.to_array())
            .unwrap();

        let result = segments
            .do_scan(layout.new_scan(Scan::all(), Default::default()).as_ref())
            .into_primitive()
            .unwrap();

        assert_eq!(array.as_slice::<i32>(), result.as_slice::<i32>());
    }
}
