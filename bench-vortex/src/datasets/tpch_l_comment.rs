// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use async_trait::async_trait;
use glob::glob;
use vortex::arrays::ChunkedArray;
use vortex::dtype::Nullability::NonNullable;
use vortex::expr::{col, pack};
use vortex::file::VortexOpenOptions;
use vortex::{Array, ArrayRef, IntoArray, ToCanonical};

use crate::datasets::Dataset;
use crate::tpch::tpchgen::{TpchGenOptions, generate_tpch_tables};
use crate::{Format, IdempotentPath};

pub struct TPCHLCommentChunked;

#[async_trait]
impl Dataset for TPCHLCommentChunked {
    fn name(&self) -> &str {
        "TPC-H l_comment chunked"
    }

    async fn to_vortex_array(&self) -> ArrayRef {
        let base_path = "tpch".to_data_path();
        let scale_factor_dir = base_path.join("1.0");
        let data_dir = scale_factor_dir.join(Format::OnDiskVortex.name());

        // Generate TPC-H CSV data if it doesn't exist
        if !data_dir.exists() {
            // Use blocking call like TPC-H benchmark does
            let options = TpchGenOptions::new("1.0".to_string(), scale_factor_dir)
                .with_format(Format::OnDiskVortex);

            futures::executor::block_on(generate_tpch_tables(options))
                .expect("Failed to generate TPC-H data");
        }

        let mut chunks: Vec<ArrayRef> = vec![];
        for path in glob(
            data_dir
                .join("lineitem_*.vortex")
                .to_string_lossy()
                .as_ref(),
        )
        .unwrap()
        {
            let file = VortexOpenOptions::file()
                .open(path.unwrap())
                .await
                .expect("cannot open lineitem.vortex");

            chunks.extend(
                file.scan()
                    .expect("cannot scan lineitem.vortex")
                    .with_projection(pack(vec![("l_comment", col("l_comment"))], NonNullable))
                    .into_array_iter()
                    .unwrap()
                    .map(|a| a.unwrap().to_canonical().into_array()),
            )
        }

        ChunkedArray::from_iter(chunks).into_array()
    }
}

pub struct TPCHLCommentCanonical;

#[async_trait]
impl Dataset for TPCHLCommentCanonical {
    fn name(&self) -> &str {
        "TPC-H l_comment canonical"
    }

    async fn to_vortex_array(&self) -> ArrayRef {
        let comments_canonical = TPCHLCommentChunked
            .to_vortex_array()
            .await
            .to_struct()
            .into_array();
        ChunkedArray::from_iter([comments_canonical]).into_array()
    }
}
