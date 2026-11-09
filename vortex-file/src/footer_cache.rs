// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Footers and zone maps of immutable files, kept across opens.

use std::hash::Hash;
use std::sync::Arc;

use async_trait::async_trait;
use moka::future::Cache;
use moka::policy::EvictionPolicy;
use parking_lot::Mutex;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_io::VortexReadAt;
use vortex_layout::layouts::zoned::zone_map_segments;
use vortex_layout::segments::SegmentCache;
use vortex_layout::segments::SegmentId;
use vortex_utils::aliases::hash_map::HashMap;
use vortex_utils::aliases::hash_set::HashSet;

use crate::Footer;
use crate::SegmentSpec;
use crate::VortexFile;
use crate::VortexOpenOptions;

/// The bytes the first read of a file takes from its end, by default.
const DEFAULT_INITIAL_READ_BYTES: usize = 64 * 1024;

/// The footers of files opened before, each with the zone maps read for it, within a byte
/// budget that evicts the least recently used first.
///
/// An entry holds a file's layout, segment map and statistics, and copies of its zone-map
/// segments: never a segment of row values, nor a buffer that shares memory with one. The
/// caller names each file with a key, which must identify the file's bytes exactly, for example
/// by its location and an immutable version. A kept footer is served without a read, so a
/// caller that must check access before each open puts what it checked in the key.
///
/// Beside each footer, the cache keeps the file opened from it, so later opens share one file
/// and the readers it builds on its first scan. Such a file holds no row values: it reads every
/// segment but the kept zone maps from its reader.
pub struct FooterCache<K> {
    footers: Cache<K, Arc<KeptFooter>>,
    files: Cache<K, Arc<VortexFile>>,
    initial_read_bytes: usize,
}

impl<K> Clone for FooterCache<K> {
    fn clone(&self) -> Self {
        Self {
            footers: self.footers.clone(),
            files: self.files.clone(),
            initial_read_bytes: self.initial_read_bytes,
        }
    }
}

impl<K> FooterCache<K>
where
    K: Hash + Eq + Clone + Send + Sync + 'static,
{
    /// A cache of footers and zone maps of about `capacity_bytes` at most, and of as many files
    /// opened from them as their footers would weigh again.
    pub fn new(capacity_bytes: u64) -> Self {
        Self {
            footers: Cache::builder()
                .max_capacity(capacity_bytes)
                .weigher(|_, footer: &Arc<KeptFooter>| footer.weight())
                .eviction_policy(EvictionPolicy::lru())
                .build(),
            files: Cache::builder()
                .max_capacity(capacity_bytes)
                .weigher(|_, file: &Arc<VortexFile>| footer_weight(file.footer()))
                .eviction_policy(EvictionPolicy::lru())
                .build(),
            initial_read_bytes: DEFAULT_INITIAL_READ_BYTES,
        }
    }

    /// Read the last `bytes` of a file in the open that reads its footer. A file no larger than
    /// this arrives whole in that read, so the cache keeps nothing for it. The default is 64 KiB.
    pub fn with_initial_read_bytes(mut self, bytes: usize) -> Self {
        self.initial_read_bytes = bytes;
        self
    }

    /// Opens the file `key` of `file_size` bytes with `options` over `reader`, from its kept
    /// footer when there is one.
    ///
    /// The first open of a file reads its footer as any open does, and keeps it with the zone
    /// maps that read covered. The file it opens holds that read, so it serves only its caller.
    /// Later opens read no footer and share one file opened from the kept footer.
    pub async fn open(
        &self,
        key: K,
        file_size: u64,
        options: VortexOpenOptions,
        reader: Arc<dyn VortexReadAt>,
    ) -> VortexResult<Arc<VortexFile>> {
        let options = options.with_file_size(file_size);
        if file_size <= self.initial_read_bytes as u64 {
            return options.open(reader).await.map(Arc::new);
        }

        if let Some(file) = self.files.get(&key).await {
            return Ok(file);
        }

        // The file of the open that read the footer, for the caller that ran it.
        let loaded = Mutex::new(None);
        let footer = self
            .footers
            .try_get_with(key.clone(), async {
                let file = options
                    .clone()
                    .with_initial_read_size(self.initial_read_bytes)
                    .open(Arc::clone(&reader))
                    .await?;
                let footer =
                    KeptFooter::from_opened(&file, file_size, self.initial_read_bytes).await?;
                *loaded.lock() = Some(file);
                Ok::<_, VortexError>(Arc::new(footer))
            })
            .await
            .map_err(|error| vortex_err!("Opening a file to keep its footer failed: {error}"))?;
        if let Some(file) = loaded.into_inner() {
            return Ok(Arc::new(file));
        }

        self.files
            .try_get_with(key.clone(), async {
                let zone_maps = KeptZoneMaps {
                    footers: self.clone(),
                    key,
                    footer: Arc::clone(&footer),
                };
                let file = options
                    .with_footer(footer.footer.clone())
                    .with_segment_cache(Arc::new(zone_maps))
                    .open(reader)
                    .await?;
                Ok::<_, VortexError>(Arc::new(file))
            })
            .await
            .map_err(|error| vortex_err!("Opening a file from its kept footer failed: {error}"))
    }

    /// Whether the cache holds the footer of `key`.
    pub fn contains_footer(&self, key: &K) -> bool {
        self.footers.contains_key(key)
    }

    /// The zone-map segments kept for `key`, when the cache holds its footer.
    #[cfg(test)]
    async fn kept_zone_maps(&self, key: &K) -> Option<HashMap<SegmentId, ByteBuffer>> {
        self.footers
            .get(key)
            .await
            .map(|footer| footer.zone_maps.clone())
    }
}

/// One file's footer and the zone maps read for it. An entry never changes: one that gains a
/// zone map is replaced, so the cache weighs each entry as it is.
#[derive(Clone)]
struct KeptFooter {
    footer: Footer,
    /// Every segment of the file's zone maps.
    zone_map_ids: Arc<HashSet<SegmentId>>,
    /// Copies of the zone-map segments read so far.
    zone_maps: HashMap<SegmentId, ByteBuffer>,
}

impl KeptFooter {
    /// The footer of `file`, opened with a first read of its last `initial_read_bytes`, and the
    /// zone maps within that read.
    async fn from_opened(
        file: &VortexFile,
        file_size: u64,
        initial_read_bytes: usize,
    ) -> VortexResult<Self> {
        let footer = file.footer().clone();
        let zone_map_ids = zone_map_segments(footer.layout())?;
        let first_read = file_size.saturating_sub(initial_read_bytes as u64);
        let source = file.segment_source();

        let mut zone_maps = HashMap::default();
        for id in &zone_map_ids {
            let spec = segment_spec(&footer, *id)?;
            // The open's first read holds these, so requesting them reads nothing.
            if spec.offset >= first_read {
                let segment = source.request(*id).await?.try_to_host_sync()?;
                zone_maps.insert(*id, Self::detached(&segment, spec));
            }
        }

        Ok(Self {
            footer,
            zone_map_ids: Arc::new(zone_map_ids),
            zone_maps,
        })
    }

    /// Whether `id` is a zone-map segment that this entry does not hold yet.
    fn lacks_zone_map(&self, id: SegmentId) -> bool {
        self.zone_map_ids.contains(&id) && !self.zone_maps.contains_key(&id)
    }

    /// This entry with the zone-map segment `id` as well, when it lacks it.
    fn with_zone_map(&self, id: SegmentId, segment: &ByteBuffer) -> VortexResult<Option<Self>> {
        if !self.lacks_zone_map(id) {
            return Ok(None);
        }

        let spec = segment_spec(&self.footer, id)?;
        let mut grown = self.clone();
        grown.zone_maps.insert(id, Self::detached(segment, spec));
        Ok(Some(grown))
    }

    /// A copy of `segment` in memory of its own. A segment read with others can share their
    /// buffer, which would keep their rows alive with it.
    fn detached(segment: &ByteBuffer, spec: &SegmentSpec) -> ByteBuffer {
        ByteBuffer::copy_from_aligned(segment, spec.alignment)
    }

    /// The bytes this entry holds, for the cache's budget.
    fn weight(&self) -> u32 {
        let zone_maps = self.zone_maps.values().map(ByteBuffer::len).sum::<usize>();
        footer_weight(&self.footer).saturating_add(u32::try_from(zone_maps).unwrap_or(u32::MAX))
    }
}

fn segment_spec(footer: &Footer, id: SegmentId) -> VortexResult<&SegmentSpec> {
    footer
        .segment_map()
        .get(*id as usize)
        .ok_or_else(|| vortex_err!("The footer names no {id}"))
}

/// About the bytes `footer` holds. A file opened from it builds readers in proportion.
fn footer_weight(footer: &Footer) -> u32 {
    let bytes = footer
        .approx_byte_size()
        .unwrap_or_else(|| footer.segment_map().len() * size_of::<SegmentSpec>());
    u32::try_from(bytes).unwrap_or(u32::MAX)
}

/// The segment cache of a file opened from its kept footer: zone maps come from the entry, and
/// zone maps read from the file join the entry. Every other segment passes through, not kept.
struct KeptZoneMaps<K> {
    footers: FooterCache<K>,
    key: K,
    footer: Arc<KeptFooter>,
}

#[async_trait]
impl<K> SegmentCache for KeptZoneMaps<K>
where
    K: Hash + Eq + Clone + Send + Sync + 'static,
{
    async fn get(&self, id: SegmentId) -> VortexResult<Option<ByteBuffer>> {
        Ok(self.footer.zone_maps.get(&id).cloned())
    }

    async fn put(&self, id: SegmentId, buffer: ByteBuffer) -> VortexResult<()> {
        if !self.footer.lacks_zone_map(id) {
            return Ok(());
        }

        // The updates of one key run one at a time, so none drops a zone map another added.
        let opened = Arc::clone(&self.footer);
        let mut failure = None;
        self.footers
            .footers
            .entry(self.key.clone())
            .and_upsert_with(|kept| {
                let kept = kept.map_or(opened, |kept| kept.into_value());
                let grown = match kept.with_zone_map(id, &buffer) {
                    Ok(grown) => grown.map_or(kept, Arc::new),
                    Err(error) => {
                        failure = Some(error);
                        kept
                    }
                };
                std::future::ready(grown)
            })
            .await;
        failure.map_or(Ok(()), Err)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::LazyLock;

    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::stream::ArrayStreamExt;
    use vortex_buffer::ByteBuffer;
    use vortex_buffer::ByteBufferMut;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_io::session::RuntimeSession;
    use vortex_layout::LayoutRef;
    use vortex_layout::layouts::zoned::zone_map_segments;
    use vortex_layout::segments::SegmentId;
    use vortex_layout::session::LayoutSession;
    use vortex_session::VortexSession;
    use vortex_utils::aliases::hash_set::HashSet;

    use super::FooterCache;
    use crate::OpenOptionsSessionExt;
    use crate::VortexFile;
    use crate::WriteOptionsSessionExt;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty()
            .with::<LayoutSession>()
            .with::<RuntimeSession>();
        crate::register_default_encodings(&session);
        crate::enable_all_registered_array_encodings(&session);
        session
    });

    /// A file of `rows` rows of incompressible numbers, so a file of many rows is larger than
    /// its first read.
    async fn file_bytes(rows: u64) -> VortexResult<ByteBuffer> {
        let noise = || {
            PrimitiveArray::from_iter(
                (0..rows).map(|value| value.wrapping_mul(6364136223846793005)),
            )
            .into_array()
        };
        let array = StructArray::from_fields(&[
            ("id", PrimitiveArray::from_iter(0..rows).into_array()),
            ("noise", noise()),
        ])?
        .into_array();

        let mut buffer = ByteBufferMut::empty();
        SESSION
            .write_options()
            .write(&mut buffer, array.to_array_stream())
            .await?;
        Ok(buffer.freeze())
    }

    async fn open(
        footers: &FooterCache<&'static str>,
        key: &'static str,
        bytes: &ByteBuffer,
    ) -> VortexResult<Arc<VortexFile>> {
        footers
            .open(
                key,
                bytes.len() as u64,
                SESSION.open_options(),
                Arc::new(bytes.clone()),
            )
            .await
    }

    /// The segments under a child named `zones` and all other segments of `root`, found by
    /// child name rather than by layout, to check [`zone_map_segments`] independently.
    fn segments_by_child_name(
        root: &LayoutRef,
    ) -> VortexResult<(HashSet<SegmentId>, HashSet<SegmentId>)> {
        let (mut zones, mut rows) = (HashSet::default(), HashSet::default());
        let mut pending = vec![(Arc::clone(root), false)];
        while let Some((layout, in_zones)) = pending.pop() {
            if in_zones {
                zones.extend(layout.segment_ids());
            } else {
                rows.extend(layout.segment_ids());
            }
            for (child, name) in layout.children()?.into_iter().zip(layout.child_names()) {
                pending.push((child, in_zones || name.as_ref() == "zones"));
            }
        }
        Ok((zones, rows))
    }

    #[tokio::test]
    async fn kept_footers_hold_zone_maps_but_no_rows() -> VortexResult<()> {
        let footers = FooterCache::new(1 << 24);
        let bytes = file_bytes(3 * 8192).await?;
        assert!(bytes.len() > 64 * 1024);

        let first = open(&footers, "file", &bytes).await?;
        let kept = footers
            .kept_zone_maps(&"file")
            .await
            .ok_or_else(|| vortex_err!("the first open keeps the footer"))?;

        let (zones, rows) = segments_by_child_name(first.footer().layout())?;
        assert_eq!(zone_map_segments(first.footer().layout())?, zones);

        let held = kept.keys().copied().collect::<HashSet<_>>();
        assert!(!held.is_empty(), "the first read covers the zone maps");
        assert!(held.is_subset(&zones), "only zone maps are kept: {held:?}");
        assert!(held.is_disjoint(&rows), "no row segment is kept: {held:?}");

        // Each kept zone map is a copy of its own, not a view of the read that fetched it with
        // row segments.
        let source = first.segment_source();
        for (id, zone_map) in &kept {
            let read = source.request(*id).await?.try_to_host_sync()?;
            assert_eq!(read.as_slice(), zone_map.as_slice());
            assert_ne!(read.as_ptr(), zone_map.as_ptr());
        }

        // A reopen from the kept footer reads the same rows.
        let again = open(&footers, "file", &bytes).await?;
        assert!(!Arc::ptr_eq(&first, &again));
        let expected = first.scan()?.into_array_stream()?.read_all().await?;
        let actual = again.scan()?.into_array_stream()?.read_all().await?;
        let mut ctx = SESSION.create_execution_ctx();
        assert_arrays_eq!(actual, expected, &mut ctx);
        Ok(())
    }

    #[tokio::test]
    async fn files_within_one_read_keep_nothing() -> VortexResult<()> {
        let footers = FooterCache::new(1 << 24);
        let bytes = file_bytes(100).await?;
        assert!(bytes.len() <= 64 * 1024);

        open(&footers, "small", &bytes).await?;
        assert!(footers.kept_zone_maps(&"small").await.is_none());
        Ok(())
    }

    #[tokio::test]
    async fn later_opens_share_one_file_per_key() -> VortexResult<()> {
        let footers = FooterCache::new(1 << 24);
        let bytes = file_bytes(3 * 8192).await?;

        let reading = open(&footers, "first", &bytes).await?;
        let kept = open(&footers, "first", &bytes).await?;
        assert!(!Arc::ptr_eq(&reading, &kept));
        assert!(Arc::ptr_eq(&kept, &open(&footers, "first", &bytes).await?));

        // Another key never shares the first key's footer or file.
        assert!(footers.contains_footer(&"first"));
        assert!(!footers.contains_footer(&"second"));
        open(&footers, "second", &bytes).await?;
        let other = open(&footers, "second", &bytes).await?;
        assert!(!Arc::ptr_eq(&kept, &other));
        Ok(())
    }
}
