use std::fmt::Display;
use std::fs::File;
use std::future::Future;
use std::io::{Read, Write};
use std::path::PathBuf;

use arrow_array::RecordBatchReader;
use bzip2::read::BzDecoder;
use futures::StreamExt;
use log::info;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use tokio::runtime::Runtime;
use vortex::array::ChunkedArray;
use vortex::arrow::FromArrowType;
use vortex::dtype::DType;
use vortex::error::{VortexError, VortexResult};
use vortex::io::{TokioAdapter, VortexWrite};
use vortex::ipc::stream::ArrayStreamIPC;
use vortex::{ArrayData, IntoArrayData};

use crate::idempotent;
use crate::reader::BATCH_SIZE;

pub fn download_data(fname: PathBuf, data_url: &str) -> PathBuf {
    idempotent(&fname, |path| {
        info!("Downloading {} from {}", fname.to_str().unwrap(), data_url);
        let mut file = File::create(path).unwrap();
        let mut response = reqwest::blocking::get(data_url).unwrap();
        if !response.status().is_success() {
            panic!("Failed to download data from {}", data_url);
        }
        response.copy_to(&mut file)
    })
    .unwrap()
}

pub fn data_vortex_uncompressed(fname_out: &str, downloaded_data: PathBuf) -> PathBuf {
    idempotent(fname_out, |path| {
        let taxi_pq = File::open(downloaded_data).unwrap();
        let builder = ParquetRecordBatchReaderBuilder::try_new(taxi_pq).unwrap();

        // FIXME(ngates): #157 the compressor should handle batch size.
        let reader = builder.with_batch_size(BATCH_SIZE).build().unwrap();

        // TODO(ngates): create an ArrayStream from an ArrayIterator.
        let dtype = DType::from_arrow(reader.schema());
        let array = ChunkedArray::try_new(
            reader
                .into_iter()
                .map(|batch_result| ArrayData::try_from(batch_result.unwrap()).unwrap())
                .collect(),
            dtype,
        )
        .unwrap()
        .into_array();

        Runtime::new()
            .unwrap()
            .block_on(async move {
                let mut write = TokioAdapter(tokio::fs::File::create(path).await.unwrap());
                let mut bytes = array.into_array_stream().into_ipc();
                while let Some(buffer) = bytes.next().await {
                    write.write_all(buffer.unwrap()).await.unwrap();
                }
                Ok::<(), VortexError>(())
            })
            .unwrap();

        Ok::<(), VortexError>(())
    })
    .unwrap()
}

pub fn decompress_bz2(input_path: PathBuf, output_path: PathBuf) -> PathBuf {
    idempotent(&output_path, |path| {
        info!(
            "Decompressing bzip from {} to {}",
            input_path.to_str().unwrap(),
            output_path.to_str().unwrap()
        );
        let input_file = File::open(input_path).unwrap();
        let mut decoder = BzDecoder::new(input_file);

        let mut buffer = Vec::new();
        decoder.read_to_end(&mut buffer).unwrap();

        let mut output_file = File::create(path).unwrap();
        output_file.write_all(&buffer).unwrap();
        Ok::<PathBuf, VortexError>(output_path.clone())
    })
    .unwrap()
}

pub trait BenchmarkDataset {
    fn as_uncompressed(&self);
    fn to_vortex_array(&self) -> VortexResult<ArrayData>;
    fn compress_to_vortex(&self) -> VortexResult<()>;
    fn write_as_parquet(&self);
    fn write_as_vortex(&self) -> impl Future<Output = ()>;
    fn list_files(&self, file_type: FileType) -> Vec<PathBuf>;
    fn directory_location(&self) -> PathBuf;
}

#[derive(Clone, Copy, Debug)]
pub enum FileType {
    Csv,
    Parquet,
    Vortex,
}

impl Display for FileType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let str = match self {
            Self::Csv => "csv".to_string(),
            Self::Parquet => "parquet".to_string(),
            Self::Vortex => "vortex".to_string(),
        };
        write!(f, "{}", str)
    }
}
