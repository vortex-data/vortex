// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Sharing one opened Vortex file between the partitions of a scan.
//!
//! DataFusion splits a large file into byte ranges and gives each range to its own partition,
//! which calls [`FileOpener::open`](datafusion_datasource::file_stream::FileOpener::open) for
//! it. Opening a [`VortexFile`] builds a new segment source with its own IO driver, so the
//! partitions reading ranges of the same file share one open file instead.

use std::future::Future;
use std::sync::Arc;

use datafusion_common::Result as DFResult;
use datafusion_datasource::PartitionedFile;
use datafusion_datasource::file_groups::FileGroup;
use object_store::ObjectMeta;
use object_store::path::Path;
use tokio::sync::OnceCell;
use vortex::file::VortexFile;
use vortex::layout::LayoutReader;
use vortex_utils::aliases::dash_map::DashMap;
use vortex_utils::aliases::dash_map::Entry;
use vortex_utils::aliases::hash_map::HashMap;

/// An opened Vortex file and the layout reader every partition scans it through.
#[derive(Clone)]
pub(crate) struct OpenedFile {
    pub file: VortexFile,
    pub layout_reader: Arc<dyn LayoutReader>,
}

/// Identifies one version of a file, so a file rewritten under the same path is never shared
/// with a stale open.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct FileKey {
    location: Path,
    size: u64,
    last_modified_micros: i64,
    e_tag: Option<String>,
}

impl From<&ObjectMeta> for FileKey {
    fn from(meta: &ObjectMeta) -> Self {
        Self {
            location: meta.location.clone(),
            size: meta.size,
            last_modified_micros: meta.last_modified.timestamp_micros(),
            e_tag: meta.e_tag.clone(),
        }
    }
}

struct SharedFile {
    file: Arc<OnceCell<OpenedFile>>,
    /// Openers of this file that have not finished opening it yet. The entry is removed when
    /// this reaches zero, so the map holds a file only while some partition still needs it.
    remaining: usize,
}

/// Files opened by one scan, shared by the partitions that read ranges of the same file.
#[derive(Default)]
pub(crate) struct SharedFiles {
    files: DashMap<FileKey, SharedFile>,
}

impl SharedFiles {
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.files.len()
    }

    /// Registers one opener of `file`, which `expected_opens` openers in this scan read.
    ///
    /// The first opener creates the entry and every opener shares it. The entry is dropped
    /// once all `expected_opens` claims are dropped.
    pub(crate) fn claim(self: &Arc<Self>, meta: &ObjectMeta, expected_opens: usize) -> FileClaim {
        let key = FileKey::from(meta);
        let file = Arc::clone(
            &self
                .files
                .entry(key.clone())
                .or_insert_with(|| SharedFile {
                    file: Arc::new(OnceCell::new()),
                    remaining: expected_opens,
                })
                .file,
        );
        FileClaim {
            files: Arc::clone(self),
            key,
            file,
        }
    }
}

/// One opener's share of a [`SharedFiles`] entry.
pub(crate) struct FileClaim {
    files: Arc<SharedFiles>,
    key: FileKey,
    file: Arc<OnceCell<OpenedFile>>,
}

impl FileClaim {
    /// Returns the shared file, opening it with `open` if no other opener has.
    ///
    /// Concurrent callers wait for a single open. A failed open is not cached, so the next
    /// caller retries it.
    pub(crate) async fn get_or_try_open<F, Fut>(&self, open: F) -> DFResult<OpenedFile>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = DFResult<OpenedFile>>,
    {
        self.file.get_or_try_init(open).await.cloned()
    }
}

impl Drop for FileClaim {
    fn drop(&mut self) {
        if let Entry::Occupied(mut entry) = self.files.files.entry(self.key.clone())
            && Arc::ptr_eq(&entry.get().file, &self.file)
        {
            let shared = entry.get_mut();
            shared.remaining = shared.remaining.saturating_sub(1);
            if shared.remaining == 0 {
                entry.remove();
            }
        }
    }
}

/// Counts the openers of each file that DataFusion split into more than one byte range.
///
/// Files missing from the result are read by a single opener and need no sharing.
pub(crate) fn split_file_opens(file_groups: &[FileGroup]) -> HashMap<Path, usize> {
    let mut opens = HashMap::<Path, usize>::default();
    for file in file_groups.iter().flat_map(FileGroup::iter) {
        if is_partial_range(file) {
            *opens.entry(file.object_meta.location.clone()).or_default() += 1;
        }
    }
    opens.retain(|_, count| *count > 1);
    opens
}

fn is_partial_range(file: &PartitionedFile) -> bool {
    file.range.as_ref().is_some_and(|range| {
        range.start != 0 || u64::try_from(range.end).ok() != Some(file.object_meta.size)
    })
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use datafusion_common::exec_datafusion_err;
    use datafusion_datasource::FileRange;
    use vortex::VortexSessionDefault;
    use vortex::array::IntoArray;
    use vortex::buffer::ByteBufferMut;
    use vortex::buffer::buffer;
    use vortex::file::OpenOptionsSessionExt;
    use vortex::file::WriteOptionsSessionExt;
    use vortex::session::VortexSession;

    use super::*;

    fn meta(path: &str, size: u64) -> ObjectMeta {
        ObjectMeta {
            location: Path::from(path),
            last_modified: Default::default(),
            size,
            e_tag: None,
            version: None,
        }
    }

    fn ranged(meta: &ObjectMeta, start: i64, end: i64) -> PartitionedFile {
        let mut file = PartitionedFile::from(meta.clone());
        file.range = Some(FileRange { start, end });
        file
    }

    #[test]
    fn counts_only_split_files() {
        let split = meta("split.vortex", 100);
        let whole = meta("whole.vortex", 100);
        let groups = vec![
            FileGroup::new(vec![
                ranged(&split, 0, 50),
                PartitionedFile::from(whole.clone()),
            ]),
            FileGroup::new(vec![ranged(&split, 50, 100), ranged(&whole, 0, 100)]),
        ];

        let opens = split_file_opens(&groups);

        assert_eq!(opens.len(), 1);
        assert_eq!(opens.get(&split.location), Some(&2));
    }

    async fn open_test_file(opens: &AtomicUsize) -> DFResult<OpenedFile> {
        opens.fetch_add(1, Ordering::SeqCst);
        let session = VortexSession::default();
        let mut buf = ByteBufferMut::empty();
        session
            .write_options()
            .write(&mut buf, buffer![1i32, 2, 3].into_array().to_array_stream())
            .await
            .map_err(|e| exec_datafusion_err!("{e}"))?;
        let file = session
            .open_options()
            .open_buffer(buf.freeze())
            .map_err(|e| exec_datafusion_err!("{e}"))?;
        let layout_reader = file
            .layout_reader()
            .map_err(|e| exec_datafusion_err!("{e}"))?;
        Ok(OpenedFile {
            file,
            layout_reader,
        })
    }

    #[tokio::test]
    async fn concurrent_claims_open_the_file_once() -> DFResult<()> {
        let files = Arc::new(SharedFiles::default());
        let meta = meta("file.vortex", 100);
        let opens = AtomicUsize::new(0);

        let first = files.claim(&meta, 2);
        let second = files.claim(&meta, 2);
        let (a, b) = tokio::join!(
            first.get_or_try_open(|| open_test_file(&opens)),
            second.get_or_try_open(|| open_test_file(&opens)),
        );

        assert_eq!(opens.load(Ordering::SeqCst), 1);
        assert!(Arc::ptr_eq(&a?.layout_reader, &b?.layout_reader));
        Ok(())
    }

    #[test]
    fn entry_is_released_after_the_last_expected_claim() {
        let files = Arc::new(SharedFiles::default());
        let meta = meta("file.vortex", 100);

        // An opener that finishes before the next one claims keeps the file shared.
        drop(files.claim(&meta, 2));
        assert_eq!(files.files.len(), 1);
        drop(files.claim(&meta, 2));
        assert!(files.files.is_empty());
    }

    #[test]
    fn rewritten_file_is_not_shared() {
        let files = Arc::new(SharedFiles::default());
        let old = meta("file.vortex", 100);
        let new = meta("file.vortex", 200);

        let old_claim = files.claim(&old, 2);
        let new_claim = files.claim(&new, 2);

        assert!(!Arc::ptr_eq(&old_claim.file, &new_claim.file));
    }
}
