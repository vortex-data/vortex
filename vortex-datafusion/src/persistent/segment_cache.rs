// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A segment cache that V2 scans of a table share across queries.

use std::sync::Arc;

use async_trait::async_trait;
use object_store::ObjectMeta;
use vortex::buffer::ByteBuffer;
use vortex::error::VortexResult;
use vortex::layout::LayoutRef;
use vortex::layout::layouts::dict::Dict;
use vortex::layout::layouts::zoned::LegacyStats;
use vortex::layout::layouts::zoned::Zoned;
use vortex::layout::segments::FileSegmentCache;
use vortex::layout::segments::MokaSegmentCache;
use vortex::layout::segments::SegmentCache;
use vortex::layout::segments::SegmentCacheSourceAdapter;
use vortex::layout::segments::SegmentEviction;
use vortex::layout::segments::SegmentFuture;
use vortex::layout::segments::SegmentId;
use vortex::layout::segments::SegmentSource;
use vortex::layout::segments::SegmentSourceId;
use vortex::metrics::Counter;
use vortex::metrics::Label;
use vortex::metrics::MetricBuilder;
use vortex::metrics::MetricsRegistry;
use vortex_utils::aliases::hash_set::HashSet;

/// The default capacity of a table's [`AuxiliarySegmentCache`]: 256MB.
pub(crate) const DEFAULT_SEGMENT_CACHE_SIZE_BYTES: usize = 256 << 20;

/// Counts the auxiliary segments a scan found in the shared cache.
pub(crate) const SEGMENT_CACHE_HITS: &str = "vortex.datafusion.segment_cache.hits";
/// Counts the auxiliary segments a scan did not find in the shared cache and read from the file.
pub(crate) const SEGMENT_CACHE_MISSES: &str = "vortex.datafusion.segment_cache.misses";
/// Counts the auxiliary segments a scan stored in the shared cache.
pub(crate) const SEGMENT_CACHE_STORES: &str = "vortex.datafusion.segment_cache.stores";

/// A bounded cache of the auxiliary segments of a table's files, shared by every V2 scan of the
/// table so that later queries skip reading them again.
///
/// # Policy
///
/// Only the segments of zone maps (the zones of a zoned layout) and of dictionary values (the
/// values of a dict layout) are cached. They are small next to the data, every query that prunes
/// with or decodes the column reads them, and a scan otherwise reads them from the file each
/// time. Every other segment bypasses the cache, including the other auxiliary children of a
/// layout, such as list elements or struct validity, which are data: caching data would make a
/// warm scan a scan of memory, and keeping the table's data is the operating system's job.
///
/// The cache is capped by the total bytes of the cached segments and evicts the least recently
/// used. A file's segments are keyed by its path, size, modification time and ETag, so a
/// rewritten file never reads the segments of its previous version.
#[derive(Clone)]
pub(crate) struct AuxiliarySegmentCache {
    cache: MokaSegmentCache,
}

impl AuxiliarySegmentCache {
    /// A cache of at most `capacity_bytes` bytes, or `None` if the capacity is zero.
    pub(crate) fn try_new(capacity_bytes: usize) -> Option<Self> {
        (capacity_bytes > 0).then(|| Self {
            cache: MokaSegmentCache::new(capacity_bytes as u64, SegmentEviction::Lru),
        })
    }

    /// Wraps `source`, the segment source of the file `object_meta` with root `layout`, so that
    /// its auxiliary segments resolve through this cache.
    ///
    /// Hits, misses and stores are counted in `metrics_registry` with `labels`.
    pub(crate) fn cached_source(
        &self,
        object_meta: &ObjectMeta,
        layout: &LayoutRef,
        source: Arc<dyn SegmentSource>,
        metrics_registry: &dyn MetricsRegistry,
        labels: Vec<Label>,
    ) -> VortexResult<Arc<dyn SegmentSource>> {
        let auxiliary = auxiliary_segments(layout)?;
        if auxiliary.is_empty() {
            return Ok(source);
        }
        let cache = CountingSegmentCache {
            cache: self.cache.for_file(source_id(object_meta)),
            hits: MetricBuilder::new(metrics_registry)
                .add_labels(labels.clone())
                .counter(SEGMENT_CACHE_HITS),
            misses: MetricBuilder::new(metrics_registry)
                .add_labels(labels.clone())
                .counter(SEGMENT_CACHE_MISSES),
            stores: MetricBuilder::new(metrics_registry)
                .add_labels(labels)
                .counter(SEGMENT_CACHE_STORES),
        };
        Ok(Arc::new(AuxiliaryCachedSource {
            auxiliary,
            cached: SegmentCacheSourceAdapter::new(Arc::new(cache), Arc::clone(&source)),
            source,
        }))
    }
}

/// Identifies the contents of a file, so that a rewritten file gets a new ID.
fn source_id(object_meta: &ObjectMeta) -> SegmentSourceId {
    SegmentSourceId::new(format!(
        "{}\0{}\0{}\0{}",
        object_meta.location,
        object_meta.size,
        object_meta
            .last_modified
            .timestamp_nanos_opt()
            .unwrap_or_default(),
        object_meta.e_tag.as_deref().unwrap_or_default(),
    ))
}

/// The segments of every layout under a zone map or dictionary values child of `layout`.
fn auxiliary_segments(layout: &LayoutRef) -> VortexResult<HashSet<SegmentId>> {
    let mut segments = HashSet::default();
    let mut stack = vec![(Arc::clone(layout), false)];
    while let Some((layout, auxiliary)) = stack.pop() {
        if auxiliary {
            segments.extend(layout.segment_ids());
        }
        for slot in 0..layout.nslots() {
            let Some(child) = layout.slot(slot)? else {
                continue;
            };
            let child_auxiliary = auxiliary || is_metadata_child(&layout, slot);
            stack.push((child, child_auxiliary));
        }
    }
    Ok(segments)
}

/// Whether the child in `slot` of `layout` holds its zone map or its dictionary values.
fn is_metadata_child(layout: &LayoutRef, slot: usize) -> bool {
    // The slots of `Zoned::child_type`, `LegacyStats::child_type` and `Dict::child_type`.
    ((layout.is::<Zoned>() || layout.is::<LegacyStats>()) && slot == 1)
        || (layout.is::<Dict>() && slot == 0)
}

/// A segment source that resolves auxiliary segments through the shared cache and every other
/// segment directly, without a cache lookup.
struct AuxiliaryCachedSource {
    auxiliary: HashSet<SegmentId>,
    cached: SegmentCacheSourceAdapter,
    source: Arc<dyn SegmentSource>,
}

impl SegmentSource for AuxiliaryCachedSource {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        if self.auxiliary.contains(&id) {
            self.cached.request(id)
        } else {
            self.source.request(id)
        }
    }
}

/// One file's view of the shared cache, counting hits, misses and stores.
struct CountingSegmentCache {
    cache: FileSegmentCache,
    hits: Counter,
    misses: Counter,
    stores: Counter,
}

#[async_trait]
impl SegmentCache for CountingSegmentCache {
    async fn get(&self, id: SegmentId) -> VortexResult<Option<ByteBuffer>> {
        let segment = self.cache.get(id).await?;
        if segment.is_some() {
            self.hits.add(1);
        } else {
            self.misses.add(1);
        }
        Ok(segment)
    }

    async fn put(&self, id: SegmentId, buffer: ByteBuffer) -> VortexResult<()> {
        self.cache.put(id, buffer).await?;
        self.stores.add(1);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use chrono::DateTime;
    use rstest::rstest;
    use vortex::VortexSessionDefault;
    use vortex::array::IntoArray;
    use vortex::array::arrays::StructArray;
    use vortex::array::arrays::VarBinArray;
    use vortex::array::validity::Validity;
    use vortex::buffer::Buffer;
    use vortex::file::OpenOptionsSessionExt;
    use vortex::file::VortexFile;
    use vortex::file::WriteOptionsSessionExt;
    use vortex::metrics::DefaultMetricsRegistry;
    use vortex::metrics::MetricValue;
    use vortex::session::VortexSession;

    use super::*;

    async fn test_file() -> VortexResult<VortexFile> {
        let rows = 40_000_usize;
        let names = ["alpha", "beta", "gamma", "delta"];
        let table = StructArray::try_new(
            ["value", "name"].into(),
            vec![
                Buffer::from_iter(0..i32::try_from(rows)?).into_array(),
                VarBinArray::from((0..rows).map(|i| names[i % 4]).collect::<Vec<_>>()).into_array(),
            ],
            rows,
            Validity::NonNullable,
        )?;
        let session = VortexSession::default();
        let mut bytes = Vec::new();
        session
            .write_options()
            .write(&mut bytes, table.into_array().to_array_stream())
            .await?;
        session.open_options().open_buffer(bytes)
    }

    fn object_meta(size: u64, nanos: i64, e_tag: Option<&str>) -> ObjectMeta {
        ObjectMeta {
            location: "data.vortex".into(),
            last_modified: DateTime::from_timestamp_nanos(nanos),
            size,
            e_tag: e_tag.map(String::from),
            version: None,
        }
    }

    fn counter(registry: &DefaultMetricsRegistry, name: &str) -> u64 {
        registry
            .snapshot()
            .iter()
            .filter(|metric| metric.name() == name)
            .map(|metric| match metric.value() {
                MetricValue::Counter(counter) => counter.value(),
                _ => 0,
            })
            .sum()
    }

    #[test]
    fn test_try_new_disabled_at_zero() {
        assert!(AuxiliarySegmentCache::try_new(0).is_none());
        assert!(AuxiliarySegmentCache::try_new(1).is_some());
    }

    #[rstest]
    #[case::size(object_meta(11, 1, Some("a")))]
    #[case::modified(object_meta(10, 2, Some("a")))]
    #[case::e_tag(object_meta(10, 1, Some("b")))]
    #[case::no_e_tag(object_meta(10, 1, None))]
    fn test_source_id_changes_with_file(#[case] changed: ObjectMeta) {
        assert_ne!(
            source_id(&object_meta(10, 1, Some("a"))),
            source_id(&changed)
        );
    }

    /// Auxiliary segments resolve through the cache, data segments bypass it.
    #[tokio::test]
    async fn test_cached_source_caches_only_auxiliary_segments() -> VortexResult<()> {
        let file = test_file().await?;
        let layout = file.footer().layout();
        let auxiliary = auxiliary_segments(layout)?;
        let data = (0..file.footer().segment_map().len())
            .map(|id| SegmentId::from(u32::try_from(id).unwrap_or(u32::MAX)))
            .find(|id| !auxiliary.contains(id));
        let Some(data) = data else {
            vortex::error::vortex_bail!("expected a data segment");
        };
        let Some(&zone_map) = auxiliary.iter().next() else {
            vortex::error::vortex_bail!("expected an auxiliary segment");
        };

        let cache = AuxiliarySegmentCache::try_new(DEFAULT_SEGMENT_CACHE_SIZE_BYTES)
            .ok_or_else(|| vortex::error::vortex_err!("cache enabled"))?;
        let registry = DefaultMetricsRegistry::default();
        let meta = object_meta(10, 1, None);
        let source =
            cache.cached_source(&meta, layout, file.segment_source(), &registry, Vec::new())?;

        source.request(data).await?;
        cache.cache.run_pending_tasks().await;
        assert_eq!(cache.cache.entry_count(), 0);
        assert_eq!(counter(&registry, SEGMENT_CACHE_MISSES), 0);

        let cold = source.request(zone_map).await?;
        assert_eq!(counter(&registry, SEGMENT_CACHE_MISSES), 1);
        assert_eq!(counter(&registry, SEGMENT_CACHE_STORES), 1);

        // A later scan of the same file finds the segment in the cache.
        let warm = cache
            .cached_source(&meta, layout, file.segment_source(), &registry, Vec::new())?
            .request(zone_map)
            .await?;
        assert_eq!(counter(&registry, SEGMENT_CACHE_HITS), 1);
        assert_eq!(warm.as_host().as_slice(), cold.as_host().as_slice());
        Ok(())
    }
}
