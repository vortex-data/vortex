use std::sync::Arc;

use vortex_array::compute::{fill_null, filter, FilterMask};
use vortex_array::{ArrayData, ContextRef};
use vortex_dtype::DType;
use vortex_error::{vortex_bail, vortex_err, vortex_panic, VortexResult};
use vortex_ipc::messages::{BufMessageReader, DecoderMessage};

use crate::layouts::flat::FlatLayout;
use crate::scanner::{LayoutScan, Poll, Scan, Scanner};
use crate::segments::{SegmentId, SegmentReader};
use crate::{LayoutData, LayoutEncoding, RowMask};

#[derive(Debug)]
pub struct FlatScan {
    layout: LayoutData,
    scan: Scan,
    dtype: DType,
    ctx: ContextRef,
}

impl FlatScan {
    pub(super) fn try_new(layout: LayoutData, scan: Scan, ctx: ContextRef) -> VortexResult<Self> {
        if layout.encoding().id() != FlatLayout.id() {
            vortex_panic!("Mismatched layout ID")
        }
        let dtype = scan.result_dtype(layout.dtype())?;
        Ok(Self {
            layout,
            scan,
            dtype,
            ctx,
        })
    }
}

impl LayoutScan for FlatScan {
    fn layout(&self) -> &LayoutData {
        &self.layout
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn create_scanner(self: Arc<Self>, mask: RowMask) -> VortexResult<Box<dyn Scanner>> {
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
            chunk: None,
        }) as _)
    }
}

// TODO(ngates): this needs to move into a shared Scanner inside the Scan. Then each scanner can
//  share work.
#[derive(Debug)]
struct FlatScanner {
    segment_id: SegmentId,
    dtype: DType,
    scan: Scan,
    ctx: ContextRef,
    mask: FilterMask,
    /// Cache of the resolved chunk that we can continue to return.
    chunk: Option<ArrayData>,
}

impl Scanner for FlatScanner {
    fn poll(&mut self, segments: &dyn SegmentReader) -> VortexResult<Poll> {
        // If we have a cached chunk, return it.
        if let Some(chunk) = &self.chunk {
            return Ok(Poll::Some(chunk.clone()));
        }

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
                    let mask = expr.evaluate(&array)?;
                    let mask = fill_null(&mask, false.into())?;
                    let mask = FilterMask::try_from(mask)?;
                    array = filter(&array, mask)?;
                }
                array = self.scan.projection.evaluate(&array)?;

                // Cache the chunk and return it.
                self.chunk.replace(array.clone());
                Ok(Poll::Some(array))
            }
        }
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use vortex_array::array::PrimitiveArray;
    use vortex_array::validity::Validity;
    use vortex_array::{ArrayDType, IntoArrayVariant, ToArrayData};
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, Nullability};
    use vortex_expr::{lit, BinaryExpr, Identity, Operator};

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
            .do_scan(layout.new_scan(Scan::all(), Default::default()).unwrap())
            .into_primitive()
            .unwrap();

        assert_eq!(array.as_slice::<i32>(), result.as_slice::<i32>());
    }

    #[test]
    fn flat_scan_filter() {
        let mut segments = TestSegments::default();
        let array = PrimitiveArray::new(buffer![1, 2, 3, 4, 5], Validity::AllValid);
        let layout = FlatLayoutWriter::new(array.dtype().clone())
            .push_one(&mut segments, array.to_array())
            .unwrap();

        let scan = Scan {
            projection: Identity::new_expr(),
            filter: Some(BinaryExpr::new_expr(
                Arc::new(Identity),
                Operator::Gt,
                lit(3i32),
            )),
        };

        let result = segments
            .do_scan(layout.new_scan(scan, Default::default()).unwrap())
            .into_primitive()
            .unwrap();

        assert_eq!(&[4, 5], result.as_slice::<i32>());
    }

    #[test]
    fn flat_scan_filter_project() {
        let mut segments = TestSegments::default();
        let array = PrimitiveArray::new(buffer![1, 2, 3, 4, 5], Validity::AllValid);
        let layout = FlatLayoutWriter::new(array.dtype().clone())
            .push_one(&mut segments, array.to_array())
            .unwrap();

        let scan = Scan {
            // The projection function here changes the scan's DType to boolean
            projection: BinaryExpr::new_expr(Arc::new(Identity), Operator::Lt, lit(5)),
            filter: Some(BinaryExpr::new_expr(
                Arc::new(Identity),
                Operator::Gt,
                lit(3),
            )),
        };

        let scan = layout.new_scan(scan, Default::default()).unwrap();
        assert_eq!(scan.dtype(), &DType::Bool(Nullability::Nullable));

        let result = segments.do_scan(scan).into_bool().unwrap();
        assert!(result.boolean_buffer().value(0));
        assert!(!result.boolean_buffer().value(1));
    }
}
