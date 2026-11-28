// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A DataFusion table over a vortex-table [`Manifest`] of Vortex files.
//!
//! A scan drops the files whose manifest stats rule out the query's filters, then hands the rest
//! to a listing table, so it scans exactly as the file integration does, over fewer files.

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::catalog::Session;
use datafusion::common::DFSchema;
use datafusion::common::Result as DFResult;
use datafusion::datasource::TableProvider;
use datafusion::datasource::listing::ListingOptions;
use datafusion::datasource::listing::ListingTable;
use datafusion::datasource::listing::ListingTableConfig;
use datafusion::datasource::listing::ListingTableUrl;
use datafusion::error::DataFusionError;
use datafusion::execution::cache::cache_manager::FileStatisticsCache;
use datafusion::logical_expr::Expr;
use datafusion::logical_expr::TableProviderFilterPushDown;
use datafusion::logical_expr::TableType;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::empty::EmptyExec;
use datafusion::prelude::SessionContext;
use futures::TryStreamExt;
use tracing::debug;
use vortex::expr::and_collect;
use vortex::io::VortexReadAt;
use vortex::io::filesystem::FileSystemRef;
use vortex::io::object_store::ObjectStoreFileSystem;
use vortex::io::session::RuntimeSessionExt;
use vortex_bench::SESSION;
use vortex_datafusion::convert::DefaultExpressionConverter;
use vortex_datafusion::convert::ExpressionConverter;
use vortex_table::Manifest;

/// A table over the Vortex files a manifest lists, with their stats.
pub struct ManifestTable {
    manifest: Manifest,
    /// The URL of each manifest entry, in manifest order.
    urls: Vec<ListingTableUrl>,
    schema: SchemaRef,
    options: ListingOptions,
    cache: Option<Arc<FileStatisticsCache>>,
    /// A listing table over every file, which answers what the scans of fewer files would.
    all: ListingTable,
}

impl fmt::Debug for ManifestTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ManifestTable")
            .field("files", &self.urls.len())
            .finish()
    }
}

impl ManifestTable {
    /// Lists the files of `table_url` and builds their manifest, reading every footer.
    pub async fn try_new(
        session: &SessionContext,
        table_url: ListingTableUrl,
        options: ListingOptions,
        schema: SchemaRef,
    ) -> anyhow::Result<Self> {
        let state = session.state();
        let store = state.runtime_env().object_store(table_url.object_store())?;
        let mut metas: Vec<_> = table_url
            .list_all_files(&state, store.as_ref(), &options.file_extension)
            .await?
            .try_collect()
            .await?;
        metas.sort_by(|a, b| a.location.cmp(&b.location));

        let fs: FileSystemRef = Arc::new(ObjectStoreFileSystem::new(store, SESSION.handle()));
        let mut files = Vec::with_capacity(metas.len());
        let mut urls = Vec::with_capacity(metas.len());
        for meta in &metas {
            let path = meta.location.to_string();
            let read: Arc<dyn VortexReadAt> = fs.open_read(&path).await?;
            let mut url = table_url.get_url().clone();
            url.set_path(&format!("/{path}"));
            urls.push(ListingTableUrl::parse(url.as_str())?);
            files.push((path, read));
        }
        let manifest = Manifest::build(&SESSION, files).await?;

        let cache = state.runtime_env().cache_manager.get_file_statistic_cache();
        let all = listing_table(urls.clone(), &options, &schema, cache.as_ref())?;
        Ok(Self {
            manifest,
            urls,
            schema,
            options,
            cache,
            all,
        })
    }

    /// The URLs of the files `filters` may match on their manifest stats.
    fn kept_files(&self, state: &dyn Session, filters: &[Expr]) -> DFResult<Vec<ListingTableUrl>> {
        let (Some(stats), Some(filter)) =
            (&self.manifest.stats, self.vortex_filter(state, filters)?)
        else {
            return Ok(self.urls.clone());
        };
        let row_counts: Vec<u64> = self.manifest.entries.iter().map(|e| e.row_count).collect();
        let may_match = stats
            .may_match(&filter, &row_counts, &SESSION)
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        let kept: Vec<_> = self
            .urls
            .iter()
            .enumerate()
            .filter(|(i, _)| may_match.value(*i))
            .map(|(_, url)| url.clone())
            .collect();
        debug!(
            kept = kept.len(),
            files = self.urls.len(),
            "manifest stats pruning"
        );
        Ok(kept)
    }

    /// The conjunction of the filters that convert to Vortex. Dropping a filter only keeps more
    /// files.
    fn vortex_filter(
        &self,
        state: &dyn Session,
        filters: &[Expr],
    ) -> DFResult<Option<vortex::expr::Expression>> {
        let df_schema = DFSchema::try_from(Arc::clone(&self.schema))?;
        let converter = DefaultExpressionConverter::default();
        let mut converted = Vec::new();
        for filter in filters {
            let physical = state.create_physical_expr(filter.clone(), &df_schema)?;
            converted.extend(converter.try_convert(&physical, &self.schema)?);
        }
        Ok(and_collect(converted))
    }
}

fn listing_table(
    urls: Vec<ListingTableUrl>,
    options: &ListingOptions,
    schema: &SchemaRef,
    cache: Option<&Arc<FileStatisticsCache>>,
) -> DFResult<ListingTable> {
    Ok(ListingTable::try_new(
        ListingTableConfig::new_with_multi_paths(urls)
            .with_listing_options(options.clone())
            .with_schema(Arc::clone(schema)),
    )?
    .with_cache(cache.cloned()))
}

#[async_trait]
impl TableProvider for ManifestTable {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DFResult<Vec<TableProviderFilterPushDown>> {
        self.all.supports_filters_pushdown(filters)
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&[usize]>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        let kept = self.kept_files(state, filters)?;
        if kept.is_empty() {
            let schema = match projection {
                Some(projection) => Arc::new(self.schema.project(projection)?),
                None => Arc::clone(&self.schema),
            };
            return Ok(Arc::new(EmptyExec::new(schema)));
        }
        listing_table(kept, &self.options, &self.schema, self.cache.as_ref())?
            .scan(state, projection, filters, limit)
            .await
    }
}
