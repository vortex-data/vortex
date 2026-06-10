// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`Dataset`] definition for the [JSONBench] dataset.
//!
//! The dataset has up to 1000 files, each with 1 million JSON lines. This setup only runs it for a single file.
//!
//! [JSONBench]: https://jsonbench.com/

use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_array::StringArray;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Schema;
use async_trait::async_trait;
use flate2::read::GzDecoder;
use parquet::arrow::ArrowWriter;
use tokio::fs::File as TokioFile;
use vortex::array::ArrayRef;
use vortex::array::EmptyMetadata;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::ChunkedArray;
use vortex::array::arrays::ExtensionArray;
use vortex::array::arrays::Struct;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::chunked::ChunkedArrayExt;
use vortex::array::arrays::struct_::StructArrayExt;
use vortex::array::stream::ArrayStreamExt;
use vortex::array::validity::Validity;
use vortex::dtype::FieldNames;
use vortex::dtype::extension::ExtDType;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::WriteOptionsSessionExt;
use vortex::io::VortexWrite;
use vortex_json::Json;

use crate::IdempotentPath;
use crate::SESSION;
use crate::conversions::parquet_to_vortex_chunks;
use crate::datasets::Dataset;
use crate::datasets::data_downloads::download_data;
use crate::idempotent_async;

const JSONBENCH_URL: &str =
    "https://clickhouse-public-datasets.s3.amazonaws.com/bluesky/file_0001.json.gz";
const JSONBENCH_SOURCE_PATH: &str = "json_bench/data.json.gz";
const JSONBENCH_PARQUET_PATH: &str = "json_bench/data.parquet";

pub struct JsonBench;

#[async_trait]
impl Dataset for JsonBench {
    fn name(&self) -> &str {
        "jsonbench"
    }

    async fn to_vortex_array(&self, _ctx: &mut ExecutionCtx) -> anyhow::Result<ArrayRef> {
        let vortex_path = idempotent_async("json_bench/data.vortex", |temp_path| async move {
            let mut output_file = TokioFile::create(&temp_path).await?;
            let parquet_path = self.to_parquet_path().await?;
            let mut ctx = SESSION.create_execution_ctx();
            let data = self
                .to_vortex_compression_array(&mut ctx, &parquet_path)
                .await?;

            SESSION
                .write_options()
                .write(&mut output_file, data.to_array_stream())
                .await?;
            output_file.flush().await?;
            Ok(temp_path)
        })
        .await?;

        Ok(SESSION
            .open_options()
            .open_path(vortex_path)
            .await?
            .scan()?
            .into_array_stream()?
            .read_all()
            .await?)
    }

    async fn to_parquet_path(&self) -> anyhow::Result<PathBuf> {
        let json_data = download_data(JSONBENCH_SOURCE_PATH.to_data_path(), JSONBENCH_URL).await?;

        idempotent_async(JSONBENCH_PARQUET_PATH, |parquet_path| async move {
            write_json_lines_as_parquet(&json_data, &parquet_path).await
        })
        .await
    }

    async fn to_vortex_compression_array(
        &self,
        _ctx: &mut ExecutionCtx,
        parquet_path: &Path,
    ) -> anyhow::Result<ArrayRef> {
        json_extension_array_from_parquet(parquet_path).await
    }
}

async fn json_extension_array_from_parquet(parquet_path: &Path) -> anyhow::Result<ArrayRef> {
    let data = parquet_to_vortex_chunks(parquet_path.to_path_buf()).await?;
    let chunks = data
        .iter_chunks()
        .map(|chunk| {
            let chunk = chunk.as_::<Struct>();
            let storage = chunk.unmasked_field_by_name("data")?.clone();
            let ext_dtype =
                ExtDType::<Json>::try_new(EmptyMetadata, storage.dtype().clone())?.erased();
            let data = ExtensionArray::new(ext_dtype, storage).into_array();
            Ok(StructArray::try_new(
                FieldNames::from(["data"]),
                vec![data],
                chunk.len(),
                Validity::NonNullable,
            )?
            .into_array())
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    Ok(ChunkedArray::from_iter(chunks).into_array())
}

async fn write_json_lines_as_parquet(json_path: &Path, parquet_path: &Path) -> anyhow::Result<()> {
    let compressed_json = tokio::fs::read(json_path).await?;
    let mut json = String::new();
    GzDecoder::new(compressed_json.as_slice()).read_to_string(&mut json)?;

    let schema = Arc::new(Schema::new(vec![Field::new("data", DataType::Utf8, false)]));
    let mut writer = ArrowWriter::try_new(File::create(parquet_path)?, Arc::clone(&schema), None)?;
    let rows = json.lines().collect::<Vec<_>>();
    let data = StringArray::from_iter_values(rows);
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(data)])?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Write;

    use arrow_array::RecordBatchReader;
    use arrow_array::StringArray;
    use arrow_schema::DataType;
    use flate2::Compression;
    use flate2::write::GzEncoder;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use vortex::array::arrays::Chunked;

    use super::*;
    use crate::temp_download_filepath;

    #[tokio::test]
    async fn writes_json_lines_as_single_string_column() -> anyhow::Result<()> {
        let json_path = temp_download_filepath();
        let parquet_path = temp_download_filepath();
        let mut encoder = GzEncoder::new(File::create(&json_path)?, Compression::default());
        encoder
            .write_all(b"{\"id\":1,\"message\":\"hello\"}\n{\"id\":2,\"message\":\"world\"}\n")?;
        encoder.finish()?;

        write_json_lines_as_parquet(&json_path, &parquet_path).await?;

        let reader =
            ParquetRecordBatchReaderBuilder::try_new(File::open(&parquet_path)?)?.build()?;
        let schema = reader.schema();
        assert_eq!(schema.fields().len(), 1);
        assert_eq!(schema.field(0).name(), "data");
        assert_eq!(schema.field(0).data_type(), &DataType::Utf8);
        assert!(!schema.field(0).is_nullable());

        let batches = reader.collect::<Result<Vec<_>, _>>()?;
        assert_eq!(batches.len(), 1);
        let data = batches[0].column(0).as_any().downcast_ref::<StringArray>();
        assert_eq!(
            data.map(|array| array.value(0)),
            Some("{\"id\":1,\"message\":\"hello\"}")
        );
        assert_eq!(
            data.map(|array| array.value(1)),
            Some("{\"id\":2,\"message\":\"world\"}")
        );

        std::fs::remove_file(json_path)?;
        std::fs::remove_file(parquet_path)?;
        Ok(())
    }

    #[tokio::test]
    async fn compression_array_uses_json_extension_dtype() -> anyhow::Result<()> {
        let json_path = temp_download_filepath();
        let parquet_path = temp_download_filepath();
        let mut encoder = GzEncoder::new(File::create(&json_path)?, Compression::default());
        encoder.write_all(b"{\"id\":1}\n")?;
        encoder.finish()?;
        write_json_lines_as_parquet(&json_path, &parquet_path).await?;

        let mut ctx = SESSION.create_execution_ctx();
        let array = JsonBench
            .to_vortex_compression_array(&mut ctx, &parquet_path)
            .await?;
        let chunked = array.as_::<Chunked>();
        let chunk = chunked.chunk(0).as_::<Struct>();
        let ext_dtype = chunk
            .unmasked_field_by_name("data")?
            .dtype()
            .as_extension_opt()
            .ok_or_else(|| anyhow::anyhow!("expected JSON extension dtype"))?
            .clone();
        ext_dtype
            .try_downcast::<Json>()
            .map_err(|_| anyhow::anyhow!("expected vortex_json::Json extension dtype"))?;

        std::fs::remove_file(json_path)?;
        std::fs::remove_file(parquet_path)?;
        Ok(())
    }
}
