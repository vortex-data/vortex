use std::sync::Arc;

use vortex_array::compute::{filter, FilterMask};
use vortex_array::ArrayData;
use vortex_error::{vortex_bail, vortex_err, VortexExpect, VortexResult};
use vortex_expr::ExprRef;
use vortex_ipc::messages::{BufMessageReader, DecoderMessage};

use crate::layouts::flat::reader::FlatReader;
use crate::operations::{Operation, Poll};
use crate::reader::LayoutScanExt;
use crate::segments::SegmentReader;

#[derive(Debug)]
pub(crate) struct FlatEvaluator {
    reader: Arc<FlatReader>,
    filter_mask: Option<FilterMask>,
    expr: ExprRef,
}

impl FlatEvaluator {
    pub(crate) fn new(reader: Arc<FlatReader>, filter_mask: FilterMask, expr: ExprRef) -> Self {
        Self {
            reader,
            filter_mask: Some(filter_mask),
            expr,
        }
    }
}

impl Operation for FlatEvaluator {
    type Output = ArrayData;

    fn poll(&mut self, segments: &dyn SegmentReader) -> VortexResult<Poll<Self::Output>> {
        // Grab the byte buffer for the segment.
        let Some(bytes) = segments.get(self.reader.segment_id()) else {
            return Ok(Poll::NeedMore(vec![self.reader.segment_id()]));
        };

        // Decode the ArrayParts from the message bytes.
        // TODO(ngates): ArrayParts should probably live in vortex-array, and not required
        //  IPC message framing to read or write.
        let mut msg_reader = BufMessageReader::new(bytes);
        let array = if let DecoderMessage::Array(parts) = msg_reader
            .next()
            .ok_or_else(|| vortex_err!("Flat message body missing"))??
        {
            parts.into_array_data(self.reader.ctx(), self.reader.dtype().clone())
        } else {
            vortex_bail!("Flat message is not ArrayParts")
        }?;

        // TODO(ngates): what's the best order to apply the filter mask / expression?
        let array = self.expr.evaluate(&array)?;

        // If we clone the filter mask, then it eagerly resolves indices. Instead, we use the
        // same technique as futures map to ensure this operation can only be polled once.
        let filter_mask = self
            .filter_mask
            .take()
            .vortex_expect("FlatEvaluator polled multiple times");
        let array = filter(&array, filter_mask)?;

        Ok(Poll::Some(array))
    }
}

#[cfg(test)]
mod test {
    use arrow_buffer::BooleanBuffer;
    use vortex_array::array::PrimitiveArray;
    use vortex_array::validity::Validity;
    use vortex_array::{ArrayDType, IntoArrayVariant, ToArrayData};
    use vortex_buffer::buffer;
    use vortex_expr::{gt, lit, Identity};

    use crate::layouts::flat::writer::FlatLayoutWriter;
    use crate::segments::test::TestSegments;
    use crate::strategies::LayoutWriterExt;

    #[test]
    fn flat_identity() {
        let mut segments = TestSegments::default();
        let array = PrimitiveArray::new(buffer![1, 2, 3, 4, 5], Validity::AllValid);
        let layout = FlatLayoutWriter::new(array.dtype().clone())
            .push_one(&mut segments, array.to_array())
            .unwrap();

        let result = segments
            .evaluate(
                layout.reader(Default::default()).unwrap(),
                Identity::new_expr(),
            )
            .into_primitive()
            .unwrap();

        assert_eq!(array.as_slice::<i32>(), result.as_slice::<i32>());
    }

    #[test]
    fn flat_expr() {
        let mut segments = TestSegments::default();
        let array = PrimitiveArray::new(buffer![1, 2, 3, 4, 5], Validity::AllValid);
        let layout = FlatLayoutWriter::new(array.dtype().clone())
            .push_one(&mut segments, array.to_array())
            .unwrap();

        let expr = gt(Identity::new_expr(), lit(3i32));
        let result = segments
            .evaluate(layout.reader(Default::default()).unwrap(), expr)
            .into_bool()
            .unwrap();

        assert_eq!(
            BooleanBuffer::from_iter([false, false, false, true, true]),
            result.boolean_buffer()
        );
    }
}
