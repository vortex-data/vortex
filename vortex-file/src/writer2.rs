// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use futures::{StreamExt, pin_mut};
use vortex_array::ArrayContext;
use vortex_array::stream::ArrayStream;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_io::VortexWrite;
use vortex_io::kanal_ext::KanalExt;
use vortex_io::runtime::Handle;
use vortex_layout::layouts::chunked::writer::ChunkedWriter;
use vortex_layout::layouts::compressed::CompressedWriter;
use vortex_layout::layouts::flat::writer::FlatWriter;
use vortex_layout::layouts::struct_::writer::StructWriter;
use vortex_layout::segments::SegmentSinkRef;
use vortex_layout::sequence::SequenceId;
use vortex_layout::{LayoutRef, Writer};

use crate::segments::writer::BufferedSegmentSink;

/// Write to a file, returning the layout ref written instead.
pub async fn write_file<W: VortexWrite, S: ArrayStream + Send + 'static>(
    mut writer: W,
    stream: S,
    handle: Handle,
) -> VortexResult<(LayoutRef, W)> {
    let dtype = stream.dtype().clone();

    let (mut ptr, eof) = SequenceId::root().split();

    // Create a channel to send buffers from the segment sink to the output stream.
    let (send, recv) = kanal::bounded_async(1);
    let mut position = 0;
    let segments = Arc::new(BufferedSegmentSink::new(send, position));
    let mut pipeline = struct_writer(handle.clone(), &dtype, segments);

    pipeline.init(eof);

    // Spawn the writer task in the background
    let layout_task = handle.spawn(async move {
        pin_mut!(stream);

        // Push each chunk. We await to allow for it to provide backpressure.
        while let Some(chunk) = stream.next().await.transpose()? {
            pipeline.push_chunk(chunk, ptr.advance()).await?;
        }

        pipeline.finish().await
    });

    // Flush buffers as they arrive
    let recv_stream = recv.into_stream();
    pin_mut!(recv_stream);
    while let Some(buffer) = recv_stream.next().await {
        if buffer.is_empty() {
            continue;
        }
        position += buffer.len() as u64;
        writer.write_all(buffer).await?;
    }

    // flush the output
    writer.flush().await?;

    // Return the result of the layout future.
    let layout = layout_task.await?;
    Ok((layout, writer))
}

fn struct_writer(handle: Handle, schema: &DType, sink: SegmentSinkRef) -> Box<dyn Writer> {
    let _handle = handle.clone();
    let _dtype = schema.clone();
    let ctx = ArrayContext::empty();

    let field_writers = schema
        .as_struct_fields()
        .fields()
        .map(|field_dtype| field_writer(field_dtype, &handle, &sink, &ctx))
        .collect();

    Box::new(StructWriter::new(schema.clone(), field_writers))
}

fn field_writer(
    dtype: DType,
    handle: &Handle,
    sink: &SegmentSinkRef,
    ctx: &ArrayContext,
) -> Box<dyn Writer> {
    let _handle = handle.clone();
    let _dtype = dtype.clone();
    let _sink = sink.clone();
    let _ctx = ctx.clone();
    let make_writer: Box<dyn Fn() -> Box<dyn Writer> + Send + Sync + 'static> =
        Box::new(move || {
            let result: Box<dyn Writer + 'static> = Box::new(CompressedWriter::new_btrblocks(
                _handle.clone(),
                Box::new(FlatWriter::new(
                    _dtype.clone(),
                    true,
                    _sink.clone(),
                    _ctx.clone(),
                )),
                true,
            ));

            result
        });

    Box::new(ChunkedWriter::new(dtype, make_writer))
}

#[cfg(test)]
mod tests {
    use vortex_array::arrays::{ChunkedArray, StructArray, VarBinViewArray};
    use vortex_array::validity::Validity;
    use vortex_array::{Array, ArrayRef, IntoArray};
    use vortex_buffer::Buffer;
    use vortex_dtype::{DType, FieldNames, Nullability};
    use vortex_io::runtime::Handle;

    use crate::writer2::write_file;

    fn make_chunk(xs: &[i32], ys: &[f32], zs: &[&str]) -> ArrayRef {
        let len = xs.len();
        let xs = xs
            .into_iter()
            .copied()
            .collect::<Buffer<i32>>()
            .into_array();
        let ys = ys
            .into_iter()
            .copied()
            .collect::<Buffer<f32>>()
            .into_array();
        let zs = VarBinViewArray::from_iter(
            zs.into_iter().map(|x| Some(x)),
            DType::Utf8(Nullability::NonNullable),
        )
        .into_array();

        StructArray::new(
            FieldNames::from(vec!["xs", "ys", "zs"]),
            vec![xs, ys, zs],
            len,
            Validity::NonNullable,
        )
        .into_array()
    }

    #[tokio::test]
    async fn test_write_structs() {
        let output = Vec::with_capacity(16 * 1024 * 1024);

        let handle = Handle::find().unwrap();

        let chunk_type = make_chunk(&[], &[], &[]).dtype().clone();

        let chunks = ChunkedArray::try_new(
            vec![
                make_chunk(
                    &[1, 2, 3, 4],
                    &[1.0, 2.0, 3.0, 4.0],
                    &["one", "two", "three", "four"],
                ),
                make_chunk(
                    &[5, 6, 7, 8],
                    &[5.0, 6.0, 7.0, 8.0],
                    &["five", "six", "seven", "eight"],
                ),
                make_chunk(
                    &[9, 10, 11, 12],
                    &[9.0, 10.0, 11.0, 12.0],
                    &["nine", "ten", "eleven", "twelve"],
                ),
            ],
            chunk_type,
        )
        .unwrap();

        let (layout, output) = write_file(output, chunks.to_array_stream(), handle)
            .await
            .unwrap();

        println!("WRITE LAYOUT: {}", layout.display_tree_verbose(true));
        println!("WROTE {} bytes", output.len());
    }
}
