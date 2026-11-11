// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A DataFusion table over a [`vortex_table::Manifest`].
//!
//! The manifest names every data file with its size and footer, so planning a scan needs no
//! listing and no footer reads: [`VortexManifestTable::scan`] prunes files on the statistics in
//! their footers, seeds the session's file metadata cache with the footers of the files it keeps,
//! and hands those files to the same [`VortexFormat`] scan a listing table builds.

use std::fmt;
use std::sync::Arc;

use arrow_schema::SchemaRef;
use async_trait::async_trait;
use datafusion_catalog::Session;
use datafusion_catalog::TableProvider;
use datafusion_common::DFSchema;
use datafusion_common::DataFusionError;
use datafusion_common::Result as DFResult;
use datafusion_common::Statistics;
use datafusion_common::project_schema;
use datafusion_datasource::PartitionedFile;
use datafusion_datasource::TableSchema;
use datafusion_datasource::compute_all_files_statistics;
use datafusion_datasource::file_format::FileFormat;
use datafusion_datasource::file_groups::FileGroup;
use datafusion_datasource::file_scan_config::FileScanConfigBuilder;
use datafusion_execution::cache::cache_manager::CachedFileMetadataEntry;
use datafusion_execution::object_store::ObjectStoreUrl;
use datafusion_expr::Expr;
use datafusion_expr::TableProviderFilterPushDown;
use datafusion_expr::TableType;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::empty::EmptyExec;
use object_store::ObjectMeta;
use object_store::path::Path;
use tokio::sync::OnceCell;
use vortex::error::VortexError;
use vortex::error::VortexResult;
use vortex::error::vortex_ensure_eq;
use vortex::error::vortex_err;
use vortex::expr::Expression;
use vortex::expr::and;
use vortex::file::FileStatsPruner;
use vortex::file::Footer;
use vortex::session::VortexSession;
use vortex_arrow::ArrowSessionExt;
use vortex_table::Manifest;
use vortex_table::footer::footer_hash;
use vortex_table::footer::parse_footer;

use super::cache::CachedVortexMetadata;
use super::format::VortexFormat;
use crate::convert::DefaultExpressionConverter;
use crate::convert::ExpressionConverter;

/// One data file of a manifest table.
struct ManifestFile {
    meta: ObjectMeta,
    footer: Footer,
}

/// A DataFusion [`TableProvider`] over the files a [`Manifest`] names.
///
/// Every manifest entry must inline its footer. Filters are reported as
/// [`Inexact`](TableProviderFilterPushDown::Inexact): the manifest only prunes whole files, and
/// the scan's own filter pushdown handles the rows of the files it keeps.
pub struct VortexManifestTable {
    format: Arc<VortexFormat>,
    session: VortexSession,
    object_store_url: ObjectStoreUrl,
    schema: SchemaRef,
    files: Vec<ManifestFile>,
    /// Per-file statistics, computed from the footers on the first scan.
    statistics: OnceCell<Vec<Arc<Statistics>>>,
}

impl fmt::Debug for VortexManifestTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VortexManifestTable")
            .field("object_store_url", &self.object_store_url)
            .field("files", &self.files.len())
            .finish_non_exhaustive()
    }
}

impl VortexManifestTable {
    /// A table over `manifest`, whose paths are relative to `prefix` in the object store at
    /// `object_store_url`.
    pub fn try_new(
        format: Arc<VortexFormat>,
        session: VortexSession,
        object_store_url: ObjectStoreUrl,
        prefix: &Path,
        manifest: &Manifest,
    ) -> VortexResult<Self> {
        let files = manifest
            .entries()
            .iter()
            .map(|entry| {
                let region = entry.footer.clone().ok_or_else(|| {
                    vortex_err!("{} has no inlined footer in the manifest", entry.path)
                })?;
                vortex_ensure_eq!(
                    footer_hash(&region),
                    entry.footer_hash,
                    "the inlined footer of {} does not match its hash",
                    entry.path
                );
                let footer = parse_footer(region, entry.byte_size, &session)?;
                let location = entry
                    .path
                    .split('/')
                    .fold(prefix.clone(), |path, part| path.join(part));
                let meta = ObjectMeta {
                    location,
                    // The manifest pins each file by its footer hash, so the cache entries this
                    // table seeds key on size alone, with a fixed modification time.
                    last_modified: Default::default(),
                    size: entry.byte_size,
                    e_tag: None,
                    version: None,
                };
                Ok(ManifestFile { meta, footer })
            })
            .collect::<VortexResult<Vec<_>>>()?;
        let first = files
            .first()
            .ok_or_else(|| vortex_err!("the manifest names no files"))?;
        for file in &files[1..] {
            vortex_ensure_eq!(
                file.footer.dtype(),
                first.footer.dtype(),
                "{} has a different dtype from the table's first file",
                file.meta.location
            );
        }
        let schema = Arc::new(session.arrow().to_arrow_schema(first.footer.dtype())?);
        Ok(Self {
            format,
            session,
            object_store_url,
            schema,
            files,
            statistics: OnceCell::new(),
        })
    }

    /// Puts every file's footer in the session's file metadata cache, so neither planning nor the
    /// scan reads a footer.
    fn seed_footer_cache(&self, state: &dyn Session) {
        let cache = state.runtime_env().cache_manager.get_file_metadata_cache();
        for file in &self.files {
            let cached = cache
                .get(&file.meta.location)
                .is_some_and(|entry| entry.is_valid_for(&file.meta));
            if !cached {
                cache.put(
                    &file.meta.location,
                    CachedFileMetadataEntry::new(
                        file.meta.clone(),
                        Arc::new(CachedVortexMetadata::from_footer(file.footer.clone())),
                    ),
                );
            }
        }
    }

    async fn file_statistics(&self, state: &dyn Session) -> DFResult<&[Arc<Statistics>]> {
        let statistics = self
            .statistics
            .get_or_try_init(|| async {
                let store = state.runtime_env().object_store(&self.object_store_url)?;
                let mut statistics = Vec::with_capacity(self.files.len());
                for file in &self.files {
                    let stats = self
                        .format
                        .infer_stats(state, &store, Arc::clone(&self.schema), &file.meta)
                        .await?;
                    statistics.push(Arc::new(stats));
                }
                Ok::<_, DataFusionError>(statistics)
            })
            .await?;
        Ok(statistics)
    }

    /// The conjunction of the filters Vortex can evaluate, if any.
    fn pruning_predicate(
        &self,
        state: &dyn Session,
        filters: &[Expr],
    ) -> DFResult<Option<Expression>> {
        let df_schema = DFSchema::try_from(Arc::clone(&self.schema))?;
        let converter = DefaultExpressionConverter::new(self.session.clone());
        let mut predicate = None;
        for filter in filters {
            let physical = state.create_physical_expr(filter.clone(), &df_schema)?;
            if let Some(expr) = converter.try_convert(&physical, &self.schema)? {
                predicate = Some(match predicate {
                    None => expr,
                    Some(lhs) => and(lhs, expr),
                });
            }
        }
        Ok(predicate)
    }
}

#[async_trait]
impl TableProvider for VortexManifestTable {
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
        Ok(vec![TableProviderFilterPushDown::Inexact; filters.len()])
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        let start = std::time::Instant::now();
        self.seed_footer_cache(state);
        let seeded = start.elapsed();
        let statistics = self.file_statistics(state).await?;
        let stats_done = start.elapsed();
        let predicate = self.pruning_predicate(state, filters)?;

        let to_df = |e: VortexError| DataFusionError::External(Box::new(e));
        // Every file shares the first file's dtype, so the filter is rewritten for pruning once.
        let pruner = predicate
            .as_ref()
            .map(|predicate| {
                FileStatsPruner::try_new(predicate, self.files[0].footer.dtype(), &self.session)
            })
            .transpose()
            .map_err(to_df)?;
        let mut kept = Vec::with_capacity(self.files.len());
        for (file, stats) in self.files.iter().zip(statistics) {
            let pruned = match &pruner {
                Some(pruner) => pruner.can_prune(&file.footer).map_err(to_df)?,
                None => false,
            };
            if !pruned {
                kept.push(
                    PartitionedFile::from(file.meta.clone()).with_statistics(Arc::clone(stats)),
                );
            }
        }
        tracing::debug!(
            files = self.files.len(),
            kept = kept.len(),
            pruned = self.files.len() - kept.len(),
            predicate = ?predicate.as_ref().map(|p| p.to_string()),
            seed_us = seeded.as_micros(),
            stats_us = (stats_done - seeded).as_micros(),
            prune_us = (start.elapsed() - stats_done).as_micros(),
            "planned manifest scan"
        );

        if kept.is_empty() {
            let projected = project_schema(&self.schema, projection)?;
            return Ok(Arc::new(EmptyExec::new(projected)));
        }

        let group_count = state.config().target_partitions().clamp(1, kept.len());
        let mut groups = vec![Vec::new(); group_count];
        for (i, file) in kept.into_iter().enumerate() {
            groups[i % group_count].push(file);
        }
        let groups = groups.into_iter().map(FileGroup::new).collect();
        let (groups, statistics) =
            compute_all_files_statistics(groups, Arc::clone(&self.schema), true, false)?;

        let source = self
            .format
            .file_source(TableSchema::from(Arc::clone(&self.schema)));
        let config = FileScanConfigBuilder::new(self.object_store_url.clone(), source)
            .with_file_groups(groups)
            .with_statistics(statistics)
            .with_projection_indices(projection.cloned())?
            .with_limit(limit)
            .build();
        self.format.create_physical_plan(state, config).await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion::arrow::util::pretty::pretty_format_batches;
    use datafusion_datasource::file_scan_config::FileScanConfig;
    use datafusion_datasource::source::DataSourceExec;
    use datafusion_execution::object_store::ObjectStoreUrl;
    use datafusion_expr::col;
    use datafusion_expr::lit;
    use datafusion_physical_plan::ExecutionPlan;
    use datafusion_physical_plan::empty::EmptyExec;
    use insta::assert_snapshot;
    use object_store::path::Path;
    use rstest::rstest;
    use vortex::VortexSessionDefault;
    use vortex::array::IntoArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::arrays::StructArray;
    use vortex::array::validity::Validity;
    use vortex::file::WriteOptionsSessionExt;
    use vortex::io::VortexWrite;
    use vortex::io::object_store::ObjectStoreReadAt;
    use vortex::io::object_store::ObjectStoreWrite;
    use vortex::io::session::RuntimeSessionExt;
    use vortex::session::VortexSession;
    use vortex_table::ManifestBuilder;

    use super::VortexManifestTable;
    use crate::VortexFormat;
    use crate::common_tests::TestSessionContext;

    /// A table of three files whose `id` columns hold 0..10, 10..20 and 20..30.
    async fn table(ctx: &TestSessionContext) -> anyhow::Result<VortexManifestTable> {
        let session = VortexSession::default();
        let prefix = Path::from("table");
        let mut builder = ManifestBuilder::new(session.clone());
        for file in 0..3i32 {
            let ids = PrimitiveArray::from_iter(file * 10..file * 10 + 10).into_array();
            let array = StructArray::try_new(["id"].into(), vec![ids], 10, Validity::NonNullable)?;
            let path = prefix.clone().join(format!("{file}.vortex"));
            let mut writer = ObjectStoreWrite::new(Arc::clone(&ctx.store), &path).await?;
            session
                .write_options()
                .write(&mut writer, array.into_array().to_array_stream())
                .await?;
            writer.shutdown().await?;
            let read = ObjectStoreReadAt::new(Arc::clone(&ctx.store), path, session.handle());
            builder.add(format!("{file}.vortex"), &read).await?;
        }
        let format = Arc::new(VortexFormat::new(session.clone()));
        Ok(VortexManifestTable::try_new(
            format,
            session,
            ObjectStoreUrl::local_filesystem(),
            &prefix,
            &builder.finish(),
        )?)
    }

    fn scanned_files(plan: &dyn ExecutionPlan) -> usize {
        if plan.downcast_ref::<EmptyExec>().is_some() {
            return 0;
        }
        plan.downcast_ref::<DataSourceExec>()
            .and_then(|exec| exec.data_source().downcast_ref::<FileScanConfig>())
            .map(|config| config.file_groups.iter().map(|group| group.len()).sum())
            .unwrap_or_else(|| panic!("expected a file scan, got {plan:?}"))
    }

    #[rstest]
    #[case::no_filter(None, 3)]
    #[case::one_file(Some(col("id").gt_eq(lit(25i32))), 1)]
    #[case::two_files(Some(col("id").lt(lit(15i32))), 2)]
    #[case::no_file(Some(col("id").gt(lit(100i32))), 0)]
    #[tokio::test]
    async fn scan_prunes_files_on_manifest_statistics(
        #[case] filter: Option<datafusion_expr::Expr>,
        #[case] files: usize,
    ) -> anyhow::Result<()> {
        let ctx = TestSessionContext::default();
        let table = table(&ctx).await?;
        let state = ctx.session.state();
        let filters: Vec<_> = filter.into_iter().collect();
        let plan =
            datafusion_catalog::TableProvider::scan(&table, &state, None, &filters, None).await?;
        assert_eq!(scanned_files(plan.as_ref()), files);
        Ok(())
    }

    #[tokio::test]
    async fn query_reads_only_the_unpruned_files() -> anyhow::Result<()> {
        let ctx = TestSessionContext::default();
        ctx.session
            .register_table("t", Arc::new(table(&ctx).await?))?;
        let batches = ctx
            .session
            .sql("SELECT count(*) AS n, min(id) AS lo, max(id) AS hi FROM t WHERE id >= 25")
            .await?
            .collect()
            .await?;
        assert_snapshot!(pretty_format_batches(&batches)?, @r"
        +---+----+----+
        | n | lo | hi |
        +---+----+----+
        | 5 | 25 | 29 |
        +---+----+----+
        ");
        Ok(())
    }
}
