// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use datafusion::arrow::array::AsArray;
use datafusion::arrow::datatypes::BinaryType;
use datafusion::parquet::arrow::ProjectionMask;
use datafusion::parquet::arrow::arrow_reader::ArrowReaderBuilder;
use futures::StreamExt;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;

//! This was a one-off file to test out to see if building a pre-trained dictionary on WKB helped
//! us compress more. The answer seems to be no, probably because our files are large enough that it
//! doesn't end up mattering.

#[tokio::main]
pub async fn main() {
    let f = File::open(
        "/Users/aduffy/Downloads/BuildingsParquet/custom_download_20251204_095222.parquet",
    )
        .await
        .unwrap();

    let mut reader = ArrowReaderBuilder::new(f).await.unwrap();

    let schema = reader.parquet_schema();
    let projection_mask = ProjectionMask::roots(&schema, [7]);

    reader = reader.with_projection(projection_mask);
    let mut reader = reader.build().unwrap();

    let mut packed = File::create("/Users/aduffy/Downloads/wkb_all.bin")
        .await
        .unwrap();

    let mut index = 0;
    while let Some(next) = reader.next().await {
        let next = next.expect("read error");
        let bytes = next.column(0).as_bytes::<BinaryType>();
        let (_, buffer, _) = bytes.clone().into_parts();
        std::fs::write(
            format!("/Users/aduffy/Downloads/wkb/{index}.bin"),
            buffer.as_slice(),
        )
            .unwrap();
        packed.write_all(&buffer).await.unwrap();
        index += 1;
    }
}
