use async_trait::async_trait;
use vortex_array::compute::filter;
use vortex_array::ArrayData;
use vortex_error::{vortex_bail, vortex_err, VortexResult};
use vortex_expr::ExprRef;
use vortex_ipc::messages::{BufMessageReader, DecoderMessage};
use vortex_scan::{AsyncEvaluator, RowMask};

use crate::layouts::flat::reader::FlatReader;
use crate::reader::LayoutScanExt;

#[async_trait(?Send)]
impl AsyncEvaluator for FlatReader {
    async fn evaluate(self: &Self, row_mask: RowMask, expr: ExprRef) -> VortexResult<ArrayData> {
        // Grab the byte buffer for the segment.
        let bytes = self.segments().get(self.segment_id()).await?;

        // Decode the ArrayParts from the message bytes.
        // TODO(ngates): ArrayParts should probably live in vortex-array, and not required
        //  IPC message framing to read or write.
        let mut msg_reader = BufMessageReader::new(bytes);
        let array = if let DecoderMessage::Array(parts) = msg_reader
            .next()
            .ok_or_else(|| vortex_err!("Flat message body missing"))??
        {
            parts.decode(self.ctx(), self.dtype().clone())
        } else {
            vortex_bail!("Flat message is not ArrayParts")
        }?;

        // TODO(ngates): what's the best order to apply the filter mask / expression?
        let array = expr.evaluate(&array)?;
        filter(&array, row_mask.into_filter_mask()?)
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use arrow_buffer::BooleanBuffer;
    use futures::executor::block_on;
    use vortex_array::array::PrimitiveArray;
    use vortex_array::validity::Validity;
    use vortex_array::{ArrayDType, IntoArrayVariant, ToArrayData};
    use vortex_buffer::buffer;
    use vortex_expr::{gt, lit, Identity};
    use vortex_scan::RowMask;

    use crate::layouts::flat::writer::FlatLayoutWriter;
    use crate::segments::test::TestSegments;
    use crate::strategies::LayoutWriterExt;

    #[test]
    fn flat_identity() {
        block_on(async {
            let mut segments = TestSegments::default();
            let array = PrimitiveArray::new(buffer![1, 2, 3, 4, 5], Validity::AllValid);
            let layout = FlatLayoutWriter::new(array.dtype().clone())
                .push_one(&mut segments, array.to_array())
                .unwrap();

            let result = layout
                .reader(Arc::new(segments), Default::default())
                .unwrap()
                .evaluate(
                    RowMask::new_valid_between(0, layout.row_count()),
                    Identity::new_expr(),
                )
                .await
                .unwrap()
                .into_primitive()
                .unwrap();

            assert_eq!(array.as_slice::<i32>(), result.as_slice::<i32>());
        })
    }

    #[test]
    fn flat_expr() {
        block_on(async {
            let mut segments = TestSegments::default();
            let array = PrimitiveArray::new(buffer![1, 2, 3, 4, 5], Validity::AllValid);
            let layout = FlatLayoutWriter::new(array.dtype().clone())
                .push_one(&mut segments, array.to_array())
                .unwrap();

            let expr = gt(Identity::new_expr(), lit(3i32));
            let result = layout
                .reader(Arc::new(segments), Default::default())
                .unwrap()
                .evaluate(RowMask::new_valid_between(0, layout.row_count()), expr)
                .await
                .unwrap()
                .into_bool()
                .unwrap();

            assert_eq!(
                BooleanBuffer::from_iter([false, false, false, true, true]),
                result.boolean_buffer()
            );
        })
    }
}
