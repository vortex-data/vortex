use std::sync::Arc;

use datafusion::datasource::listing::{
    ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl,
};
use datafusion::prelude::SessionContext;
use object_store::ObjectStore;
use object_store::local::LocalFileSystem;
use tempfile::tempdir;
use tokio::fs::OpenOptions;
use url::Url;
use vortex_array::arrays::{ChunkedArray, StructArray, VarBinArray};
use vortex_array::stream::ArrayStreamArrayExt;
use vortex_array::validity::Validity;
use vortex_array::{Array, IntoArray};
use vortex_buffer::buffer;
use vortex_datafusion::persistent::VortexFormat;
use vortex_error::vortex_err;
use vortex_file::VortexWriteOptions;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let temp_dir = tempdir()?;
    let strings = ChunkedArray::from_iter([
        VarBinArray::from(vec!["ab", "foo", "bar", "baz"]).into_array(),
        VarBinArray::from(vec!["ab", "foo", "bar", "baz"]).into_array(),
    ])
    .into_array();

    let numbers = ChunkedArray::from_iter([
        buffer![1u32, 2, 3, 4].into_array(),
        buffer![5u32, 6, 7, 8].into_array(),
    ])
    .into_array();

    let st = StructArray::try_new(
        ["strings".into(), "numbers".into()].into(),
        vec![strings, numbers],
        8,
        Validity::NonNullable,
    )?;

    let filepath = temp_dir.path().join("a.vtx");

    let f = OpenOptions::new()
        .write(true)
        .truncate(true)
        .create(true)
        .open(&filepath)
        .await?;

    VortexWriteOptions::default()
        .write(f, st.to_array_stream())
        .await?;

    let ctx = SessionContext::new();

    let object_store: Arc<dyn ObjectStore> = Arc::new(LocalFileSystem::new());
    let url: Url = Url::try_from("file://")?;
    ctx.register_object_store(&url, object_store);

    let format = Arc::new(VortexFormat::default());
    let table_url = ListingTableUrl::parse(
        filepath
            .to_str()
            .ok_or_else(|| vortex_err!("Path is not valid UTF-8"))?,
    )?;
    let config = ListingTableConfig::new(table_url)
        .with_listing_options(ListingOptions::new(format as _))
        .infer_schema(&ctx.state())
        .await?;

    let listing_table = Arc::new(ListingTable::try_new(config)?);

    ctx.register_table("vortex_tbl", listing_table as _)?;

    run_query(&ctx, "SELECT * FROM vortex_tbl").await?;

    Ok(())
}

async fn run_query(ctx: &SessionContext, query_string: impl AsRef<str>) -> anyhow::Result<()> {
    let query_string = query_string.as_ref();

    ctx.sql(&format!("EXPLAIN {query_string}"))
        .await?
        .show()
        .await?;

    ctx.sql(query_string).await?.show().await?;

    Ok(())
}
