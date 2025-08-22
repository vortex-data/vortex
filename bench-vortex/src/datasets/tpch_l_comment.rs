// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::datasets::Dataset;
use crate::tpch::tpchgen::{generate_tpch_tables, TpchGenOptions};
use crate::{Format, IdempotentPath};
use async_trait::async_trait;
use glob::glob;
use vortex::arrays::ChunkedArray;
use vortex::dtype::Nullability::NonNullable;
use vortex::expr::{col, pack};
use vortex::file::VortexOpenOptions;
use vortex::{Array, ArrayRef, IntoArray, ToCanonical};

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

            generate_tpch_tables(options)
                .await
                .expect("Failed to generate TPC-H data");
        }

        let mut chunks: Vec<ArrayRef> = vec![];
        let paths = glob(
            data_dir
                .join("lineitem_*.vortex")
                .to_string_lossy()
                .as_ref(),
        )
        .unwrap()
        .map(|x| x.unwrap())
        .collect::<Vec<_>>();

        for path in paths {
            let file = VortexOpenOptions::file().open_blocking(path).unwrap();

            chunks.extend(
                file.scan()
                    .expect("cannot scan lineitem.vortex")
                    .with_projection(pack(vec![("l_comment", col("l_comment"))], NonNullable))
                    .into_array_iter()
                    .unwrap()
                    .map(|a| a.unwrap().to_canonical().unwrap().into_array()),
            );
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
            .unwrap()
            .into_array();
        ChunkedArray::from_iter([comments_canonical]).into_array()
    }
}
