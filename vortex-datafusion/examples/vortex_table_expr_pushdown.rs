use std::sync::{Arc, LazyLock};

use datafusion::datasource::listing::{
    ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl,
};
use datafusion::datasource::DefaultTableSource;
use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::SessionContext;
use datafusion_expr::{col, LogicalPlanBuilder};
use object_store::local::LocalFileSystem;
use object_store::ObjectStore;
use url::Url;
use vortex_alp::{ALPEncoding, ALPRDEncoding};
use vortex_array::array::{
    PrimitiveEncoding, SparseEncoding, StructEncoding, VarBinEncoding, VarBinViewEncoding,
};
use vortex_array::encoding::EncodingRef;
use vortex_array::Context;
use vortex_datafusion::expr::list_mean;
use vortex_datafusion::persistent::format::VortexFormat;
use vortex_datafusion::persistent::optimizer::VortexExecProjectionPushdown;
use vortex_dict::DictEncoding;
use vortex_fastlanes::{BitPackedEncoding, DeltaEncoding, FoREncoding};
use vortex_fsst::FSSTEncoding;
use vortex_runend::RunEndEncoding;

pub static ALL_ENCODINGS_CONTEXT: LazyLock<Arc<Context>> = LazyLock::new(|| {
    Arc::new(Context::default().with_encodings([
        &ALPEncoding as EncodingRef,
        &ALPRDEncoding,
        &DictEncoding,
        &BitPackedEncoding,
        &DeltaEncoding,
        &FoREncoding,
        &FSSTEncoding,
        &PrimitiveEncoding,
        &RunEndEncoding,
        &SparseEncoding,
        &StructEncoding,
        &VarBinEncoding,
        &VarBinViewEncoding,
    ]))
});

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let ctx = SessionContext::new();

    let object_store: Arc<dyn ObjectStore> = Arc::new(LocalFileSystem::new());
    let url: Url = Url::try_from("file://")?;
    ctx.register_object_store(&url, object_store);

    let format = Arc::new(VortexFormat::new(&ALL_ENCODINGS_CONTEXT.clone()));
    let table_url = ListingTableUrl::parse(
        "/Users/mbakovic/git/vortex/vortex-genetics/100_000-no-lists-of-lists.vcf.vortex",
    )?;
    let config = ListingTableConfig::new(table_url)
        .with_listing_options(ListingOptions::new(format as _))
        .infer_schema(&ctx.state())
        .await?;

    let listing_table = Arc::new(ListingTable::try_new(config)?);

    let logical_plan = LogicalPlanBuilder::scan(
        "vortex_tbl",
        Arc::new(DefaultTableSource::new(listing_table as _)),
        None,
    )?
    .build()?;
    let ctx = SessionContext::new_with_state(
        SessionStateBuilder::new()
            .with_physical_optimizer_rule(Arc::new(VortexExecProjectionPushdown::new()))
            .build(),
    );
    let df = ctx.execute_logical_plan(logical_plan).await?;

    df.select(vec![list_mean(col("vortex_tbl.\"GT\""))])?.show_limit(20).await?;

    Ok(())
}
