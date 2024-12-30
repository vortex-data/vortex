use std::future::{ready, Future};
use std::io::Write;
use std::path::PathBuf;

use futures::executor::block_on;
use vortex::error::VortexError;
use vortex::io::{IoBuf, VortexWrite};

use crate::data_downloads::{data_vortex_uncompressed, download_data};
use crate::reader::rewrite_parquet_as_vortex;
use crate::{idempotent, IdempotentPath};

fn download_taxi_data() -> PathBuf {
    let taxi_parquet_fpath = "yellow-tripdata-2023-11.parquet".to_data_path();
    let taxi_data_url =
        "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-11.parquet";
    download_data(taxi_parquet_fpath, taxi_data_url)
}

pub fn taxi_data_parquet() -> PathBuf {
    download_taxi_data()
}

pub fn taxi_data_vortex_uncompressed() -> PathBuf {
    data_vortex_uncompressed("taxi-uncompressed.vortex", download_taxi_data())
}

pub fn taxi_data_vortex() -> PathBuf {
    idempotent("taxi.vortex", |output_fname| {
        block_on(async {
            let output_file = std::fs::File::create(output_fname)?;
            let output_file = StdFile(output_file);
            rewrite_parquet_as_vortex(taxi_data_parquet(), output_file).await?;
            Ok::<PathBuf, VortexError>(output_fname.to_path_buf())
        })
    })
    .unwrap()
}

//
// Test code uses futures_executor with a local pool, and nothing in VortexWrite ties us to Tokio,
// so this is a simple bridge to allow us to use a `std::fs::File` as a `VortexWrite`.
//

struct StdFile(std::fs::File);

impl VortexWrite for StdFile {
    async fn write_all<B: IoBuf>(&mut self, buffer: B) -> std::io::Result<B> {
        self.0.write_all(buffer.as_slice())?;
        Ok(buffer)
    }

    async fn flush(&mut self) -> std::io::Result<()> {
        self.0.flush()?;
        Ok(())
    }

    fn shutdown(&mut self) -> impl Future<Output = std::io::Result<()>> {
        ready(Ok(()))
    }
}
