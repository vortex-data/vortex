use futures::stream::once;
use futures::StreamExt;
use std::future::ready;
use std::sync::Arc;
use vortex_array::arrays::NullArray;
use vortex_array::{ArrayContext, IntoArray};
use vortex_dtype::DType;
use vortex_error::{vortex_bail, VortexResult};
use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex_layout::segments::SequenceWriter;
use vortex_layout::{LayoutRef, LayoutStrategy, SendableSequentialStream, SequentialStream, SequentialStreamAdapter, SequentialStreamExt, TaskExecutor};

pub async fn write_file(
    ctx: &ArrayContext,
    mut source: SendableSequentialStream,
    sink: SequenceWriter,
    task_executor: &Arc<dyn TaskExecutor>,
) -> VortexResult<LayoutRef> {
    // Top-level type must be struct
    let dtype = source.dtype().clone();
    let DType::Struct(fields, _) = dtype else {
        vortex_bail!("File DType must be struct, was {}", source.dtype());
    };

    for dtype in fields.fields() {
        match dtype {
            DType::Null => {
                // Null arrays should not be chunked, since they contain no data. Instead, write
                // a single FlatLayout encoding the length of the column.
                let mut rows = 0;
                while let Some(chunk) = source.next().await {
                    let (_, chunk) = chunk?;
                    rows += chunk.len();
                }

                // Assign a new sequence ID for this type instead.
                let single_null_chunk = NullArray::new(rows).into_array();
                let stream = SequentialStreamAdapter::new(dtype, once(ready(single_null_chunk))).sendable();

                // Write a FlatStrategy into the output node instead.
                let flat_writer = FlatLayoutStrategy::default();

                flat_writer.write_stream(
                    ctx,
                    sink.clone(),
                    stream,
                )
            }
            DType::Bool(_) |
            DType::Primitive(_, _) |
            DType::Decimal(_, _) |
            DType::Extension(_) => {
                // Extension we assume is fixed-size if it is not otherwise one of the others.
                // Write the fixed-size values out instead
                Box::pin(write_fixed_width(
                    ctx, source, sink.clone(), task_executor,
                ))
            }
            // Collect a shared dictionary, extract that chunk out instead.
            // They always do some work before writing to their child instead.
            // So the dict strategy is going to identify dictionaries.
            // If Dict is not selected for a colum, it will apply View + FSST instead.
            DType::Utf8(_) => {}
            DType::Binary(_) => {}
            DType::List(_, _) => {}
            // If it's another top-level struct, we should do
            DType::Struct(_, _) => {}
        }
    }
    let written = fields.fields()
        .map(|field|)
        .collect();
}

/// Vortex file writer for fixed-width columns. These are types such as Primitive, Bool,
/// Decimal and the like.
///
/// Fixed-width types are easy to layout because they do not have large or small "outlier values"
/// that exist in the variable-sized types. Because of this, serialization takes on a fairly simple
/// format:
///
/// 1. We repartition to **zones** of 8,192 values, over which we calculate statistics
/// 2. We collect the **zones** back together into **column chunks** which are at least 1MB and
///    some multiple of 8,192 values
/// 3. We compress each column chunks and emit it as a **flat layout**, i.e. just a single array
pub async fn write_fixed_width(
    ctx: &ArrayContext,
    source: SendableSequentialStream,
    sink: SequenceWriter,
    task_executor: &Arc<dyn TaskExecutor>,
) -> VortexResult<LayoutRef> {
    // We make a compressed copy of this instead.
}

async fn write_flat(
    ctx: &ArrayContext,
    source: SendableSequentialStream,
    sink: SequenceWriter,
) -> VortexResult<LayoutRef> {
    let strategy: Box<dyn LayoutStrategy> = Box::new(move || {});

    todo!()
}

pub struct Writer {

}

/// Writer for direct file access instead of other things here...I think there is too much
/// pluggability in some of these positions.
///
/// Even just having it all in Rust makes a big difference. We want to be able to plugin new
/// WASM encodings to opt for this.
impl Writer {

}

#[cfg(test)]
mod tests {
    #[test]
    fn test_flat() {
        // Flat layout writer.
        // I think there's a lot going on here instead
    }
}