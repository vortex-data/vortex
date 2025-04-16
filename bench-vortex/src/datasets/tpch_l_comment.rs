use async_trait::async_trait;
use vortex::arrays::ChunkedArray;
use vortex::dtype::FieldName;
use vortex::{Array, ArrayExt, ArrayRef, ToCanonical};

use crate::datasets::Dataset;
use crate::tpch;
use crate::tpch::dbgen::{DBGen, DBGenOptions};

pub struct TPCHLCommentChunked;

#[async_trait]
impl Dataset for TPCHLCommentChunked {
    fn name(&self) -> &str {
        "TPC-H l_comment chunked"
    }

    async fn to_vortex_array(&self) -> ArrayRef {
        let data_dir = DBGen::new(DBGenOptions::default()).generate().unwrap();
        let lineitem_vortex = tpch::load_table(data_dir, "lineitem", &tpch::schema::LINEITEM).await;

        let lineitem_chunked = lineitem_vortex.as_::<ChunkedArray>();
        let comment_chunks = lineitem_chunked.chunks().iter().map(|chunk| {
            chunk
                .as_struct_typed()
                .unwrap()
                .project(&[FieldName::from("l_comment")])
                .unwrap()
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
