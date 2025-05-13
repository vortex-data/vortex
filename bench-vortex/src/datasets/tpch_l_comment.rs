use async_trait::async_trait;
use vortex::arrays::{ChunkedArray, ChunkedVTable};
use vortex::dtype::FieldName;
use vortex::{ArrayExt, ArrayRef, IntoArray, ToCanonical};

use crate::datasets::Dataset;
use crate::ddb::{build_vortex_duckdb, get_executable_path};
use crate::tpch::duckdb::{DuckdbTpcOptions, TpcDataset, generate_tpc};
use crate::{Format, IdempotentPath, tpch};

pub struct TPCHLCommentChunked;

#[async_trait]
impl Dataset for TPCHLCommentChunked {
    fn name(&self) -> &str {
        "TPC-H l_comment chunked"
    }

    async fn to_vortex_array(&self) -> ArrayRef {
        // TODO(joe): replace with a duckdb binary
        build_vortex_duckdb();
        let duckdb_resolved_path = get_executable_path(&None);
        let opts = DuckdbTpcOptions::new("tpch".to_data_path(), TpcDataset::TpcH, Format::Csv)
            .with_duckdb_path(duckdb_resolved_path.clone());
        let data_dir = generate_tpc(opts).expect("gen tpch");

        let lineitem_vortex = tpch::load_table(data_dir, "lineitem", &tpch::schema::LINEITEM).await;

        let lineitem_chunked = lineitem_vortex.as_::<ChunkedVTable>();
        let comment_chunks = lineitem_chunked.chunks().iter().map(|chunk| {
            chunk
                .to_struct()
                .unwrap()
                .project(&[FieldName::from("l_comment")])
                .unwrap()
                .into_array()
        });
        ChunkedArray::from_iter(comment_chunks).into_array()
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
