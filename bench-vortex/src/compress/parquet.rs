use std::io::Cursor;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::Schema;
use bytes::Bytes;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

#[inline(never)]
pub fn parquet_compress_write(
    batches: Vec<RecordBatch>,
    schema: Arc<Schema>,
    compression: Compression,
    buf: &mut Vec<u8>,
) -> usize {
    let mut buf = Cursor::new(buf);
    let writer_properties = WriterProperties::builder()
        .set_compression(compression)
        .build();
    let mut writer = ArrowWriter::try_new(&mut buf, schema, Some(writer_properties)).unwrap();
    for batch in batches {
        writer.write(&batch).unwrap();
    }
    writer.flush().unwrap();
    let n_bytes = writer.bytes_written();
    writer.close().unwrap();
    n_bytes
}

#[inline(never)]
pub fn parquet_decompress_read(buf: Bytes) -> usize {
    let builder = ParquetRecordBatchReaderBuilder::try_new(buf).unwrap();
    let reader = builder.build().unwrap();
    let mut nbytes = 0;
    for batch in reader {
        nbytes += batch.unwrap().get_array_memory_size()
    }
    nbytes
}
