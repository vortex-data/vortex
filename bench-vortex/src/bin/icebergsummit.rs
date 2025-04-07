#![allow(unused, dead_code)]

use std::sync::Arc;

use arrow_array::{Array, RecordBatch, StructArray};
use arrow_schema::{DataType, Fields, Schema};
use futures::StreamExt;
use indicatif::ProgressBar;
use object_store::azure::MicrosoftAzureBuilder;
use object_store::path::Path;
use object_store::{ObjectMeta, ObjectStore};
use parquet::arrow::ParquetRecordBatchStreamBuilder;
use parquet::arrow::async_reader::ParquetObjectReader;
use tokio::sync::Semaphore;
use vortex::TryIntoArray;
use vortex::dtype::DType;
use vortex::dtype::arrow::FromArrowType;
use vortex::error::{VortexError, VortexExpect};
use vortex::file::VortexWriteOptions;
use vortex::io::ObjectStoreWriter;
use vortex::stream::ArrayStreamAdapter;

#[tokio::main(flavor = "multi_thread", worker_threads = 10)]
pub async fn main() {
    let read_store = MicrosoftAzureBuilder::from_env()
        .build()
        .expect("build read store");
    let write_store = MicrosoftAzureBuilder::from_env()
        .with_container_name("vortex")
        .build()
        .expect("build write store");

    let read_store: Arc<dyn ObjectStore> = Arc::new(read_store);
    let write_store: Arc<dyn ObjectStore> = Arc::new(write_store);

    // Only allow 16 concurrent uploads.
    let semaphore = Arc::new(Semaphore::new(16));

    let mut filtered = read_store.list(None);

    let mut tasks = vec![];
    while let Some(next) = filtered.next().await {
        let blob = next.unwrap();
        if !blob.location.as_ref().ends_with(".parquet") {
            continue;
        }
        // Otherwise extract the location info
        println!("discovered parquet file: {:?}", blob.location.as_ref());

        let _read_store = read_store.clone();
        let _write_store = write_store.clone();
        let _sem = semaphore.clone();
        let task = tokio::spawn(async move {
            let _perm = _sem.acquire().await.unwrap();
            exec_convert(_read_store, _write_store, blob).await;
        });
        tasks.push(task);
    }

    for task in ProgressBar::new(tasks.len() as u64).wrap_iter(tasks.into_iter()) {
        task.await.unwrap();
    }
}

const BATCH_SIZE: usize = 65_536;

async fn exec_convert(
    read_store: Arc<dyn ObjectStore>,
    write_store: Arc<dyn ObjectStore>,
    meta: ObjectMeta,
) {
    eprintln!("Converting input Parquet file: {}", meta.location);

    let path = meta.location.clone();

    let obj_reader = ParquetObjectReader::new(read_store.clone(), meta);
    let parquet = ParquetRecordBatchStreamBuilder::new(obj_reader)
        .await
        .expect("parquet reader build")
        .with_batch_size(BATCH_SIZE);
    let num_rows = parquet.metadata().file_metadata().num_rows();

    let schema_no_decimal = Schema::new(cast_decimal_fields(&parquet.schema().fields()));
    let dtype = DType::from_arrow(&schema_no_decimal);
    let mut vortex_stream = parquet
        .build()
        .expect("build reader stream")
        .map(|record_batch| {
            record_batch
                .map_err(VortexError::from)
                .and_then(|rb| cast_away_decimal_batch(rb).try_into_array())
        })
        .boxed();

    // Parquet reader returns batches, rather than row groups. So make sure we correctly
    // configure the progress bar.
    let nbatches = u64::try_from(num_rows)
        .vortex_expect("negative row count?")
        .div_ceil(BATCH_SIZE as u64);

    let output_path = rename_vortex(path);
    let writer = ObjectStoreWriter::new(write_store, output_path.clone())
        .await
        .expect("make writer");
    VortexWriteOptions::default()
        .write(writer, ArrayStreamAdapter::new(dtype, vortex_stream))
        .await
        .expect("write vortex");
    println!("complete writing {output_path}");
}

fn rename_vortex(path: Path) -> Path {
    let no_ext = path.as_ref().strip_suffix(".parquet").unwrap();
    format!("{no_ext}.vortex").into()
}

fn cast_decimal_fields(fields: &Fields) -> Fields {
    fields
        .iter()
        .map(|field| match field.data_type() {
            DataType::Decimal128(..) | DataType::Decimal256(..) => {
                let new_field = (**field).clone().with_data_type(DataType::Float64);
                Arc::new(new_field)
            }
            _ => field.clone(),
        })
        .collect()
}

fn cast_away_decimal_batch(batch: RecordBatch) -> RecordBatch {
    let mut new_fields = vec![];
    let batch = StructArray::from(batch);

    for field in batch.columns() {
        new_fields.push(cast_away_decimal(field));
    }

    StructArray::new(
        cast_decimal_fields(batch.fields()),
        new_fields,
        batch.nulls().cloned(),
    )
    .into()
}

fn cast_away_decimal(array: &arrow_array::ArrayRef) -> arrow_array::ArrayRef {
    match array.data_type() {
        DataType::Decimal128(..) | DataType::Decimal256(..) => {
            arrow_cast::cast(array.as_ref(), &DataType::Float64).unwrap()
        }
        _ => array.clone(),
    }
}
