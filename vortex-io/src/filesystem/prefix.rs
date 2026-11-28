// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use futures::stream::BoxStream;
use vortex_error::VortexResult;

use crate::VortexReadAt;
use crate::filesystem::FileListing;
use crate::filesystem::FileSystem;
use crate::filesystem::FileSystemRef;

/// A [`FileSystem`] decorator that roots all operations under a given prefix.
///
/// Paths returned from [`list`](FileSystem::list) are relative to the prefix, and paths
/// passed to [`open_read`](FileSystem::open_read) are automatically prefixed.
#[derive(Debug)]
pub struct PrefixFileSystem {
    inner: FileSystemRef,
    prefix: String,
}

impl PrefixFileSystem {
    pub fn new(inner: FileSystemRef, prefix: String) -> Self {
        // Normalize non-empty prefixes to have a single trailing slash for clean concatenation,
        // and leave empty prefixes empty so keys are not rewritten with a leading slash.
        let trimmed = prefix.trim_matches('/');
        let prefix = if trimmed.is_empty() {
            String::new()
        } else {
            format!("{trimmed}/")
        };
        Self { inner, prefix }
    }
}

#[async_trait]
impl FileSystem for PrefixFileSystem {
    fn list(&self, prefix: &str) -> BoxStream<'_, VortexResult<FileListing>> {
        let full_prefix = format!("{}{}", self.prefix, prefix.trim_start_matches('/'));

        let strip_prefix = self.prefix.clone();
        self.inner
            .list(&full_prefix)
            .map(move |result| {
                result.map(|mut listing| {
                    listing.path = listing
                        .path
                        .strip_prefix(&strip_prefix)
                        .unwrap_or(&listing.path)
                        .to_string();
                    listing
                })
            })
            .boxed()
    }

    async fn head(&self, path: &str) -> VortexResult<Option<FileListing>> {
        let full_path = format!("{}{}", self.prefix, path.trim_start_matches('/'));
        Ok(self.inner.head(&full_path).await?.map(|mut listing| {
            listing.path = listing
                .path
                .strip_prefix(&self.prefix)
                .unwrap_or(&listing.path)
                .to_string();
            listing
        }))
    }

    async fn open_read(&self, path: &str) -> VortexResult<Arc<dyn VortexReadAt>> {
        self.inner
            .open_read(&format!("{}{}", self.prefix, path.trim_start_matches('/')))
            .await
    }

    async fn delete(&self, path: &str) -> VortexResult<()> {
        self.inner
            .delete(&format!("{}{}", self.prefix, path.trim_start_matches('/')))
            .await
    }
}

impl dyn FileSystem + 'static {
    /// Create a new filesystem that applies the given prefix to all operations on this filesystem.
    pub fn with_prefix(self: Arc<Self>, prefix: String) -> FileSystemRef {
        Arc::new(PrefixFileSystem::new(self, prefix))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use async_trait::async_trait;
    use futures::TryStreamExt;
    use futures::stream;
    use parking_lot::Mutex;
    use rstest::rstest;
    use vortex_buffer::Alignment;
    use vortex_buffer::ByteBuffer;
    use vortex_error::VortexResult;
    use vortex_error::vortex_bail;

    use super::*;
    use crate::test_harness::SimulatedExecutor;

    #[derive(Debug)]
    struct MemoryFileSystem {
        files: Mutex<BTreeMap<String, ByteBuffer>>,
        recorded_prefixes: Mutex<Vec<String>>,
    }

    impl MemoryFileSystem {
        fn new(entries: &[(&str, &[u8])]) -> Arc<Self> {
            let map = entries
                .iter()
                .map(|(k, v)| ((*k).to_owned(), ByteBuffer::from(v.to_vec())))
                .collect();
            Arc::new(Self {
                files: Mutex::new(map),
                recorded_prefixes: Mutex::new(Vec::new()),
            })
        }
    }

    #[async_trait]
    impl FileSystem for MemoryFileSystem {
        fn list(&self, prefix: &str) -> BoxStream<'_, VortexResult<FileListing>> {
            self.recorded_prefixes.lock().push(prefix.to_owned());
            let prefix = prefix.to_owned();
            let items: Vec<_> = self
                .files
                .lock()
                .iter()
                .filter(|(path, _)| path.starts_with(&prefix))
                .map(|(path, data)| {
                    Ok(FileListing {
                        path: path.clone(),
                        size: Some(data.len() as u64),
                    })
                })
                .collect();
            stream::iter(items).boxed()
        }

        async fn head(&self, path: &str) -> VortexResult<Option<FileListing>> {
            Ok(self.files.lock().get(path).map(|data| FileListing {
                path: path.to_owned(),
                size: Some(data.len() as u64),
            }))
        }

        async fn open_read(&self, path: &str) -> VortexResult<Arc<dyn VortexReadAt>> {
            let Some(buf) = self.files.lock().get(path).cloned() else {
                vortex_bail!("not found: {path}");
            };
            Ok(Arc::new(buf))
        }

        async fn delete(&self, path: &str) -> VortexResult<()> {
            if self.files.lock().remove(path).is_none() {
                vortex_bail!("cannot delete missing: {path}");
            }
            Ok(())
        }
    }

    #[rstest]
    #[case::plain("data/v1")]
    #[case::leading_slash("/data/v1")]
    #[case::trailing_slash("data/v1/")]
    #[case::multiple_slashes("///data/v1///")]
    fn test_prefix_normalization_and_isolation(#[case] raw_prefix: &str) -> VortexResult<()> {
        let exec = SimulatedExecutor::new();
        let mem = MemoryFileSystem::new(&[
            ("data/v1/a.vortex", b"aaa"),
            ("data/v1/sub/b.vortex", b"bbbbb"),
            ("data/v10/sibling.vortex", b"sibling"),
            ("other/c.vortex", b"ccc"),
        ]);
        let root: FileSystemRef = Arc::clone(&mem) as FileSystemRef;
        let prefixed = root.with_prefix(raw_prefix.to_owned());

        // Listing with empty prefix must only surface files under `data/v1/` (not `data/v10/`),
        // with paths stripped of `data/v1/`.
        let mut listed = exec.drive(prefixed.list("").try_collect::<Vec<_>>())?;
        listed.sort();
        assert_eq!(
            listed,
            vec![
                FileListing {
                    path: "a.vortex".to_owned(),
                    size: Some(3),
                },
                FileListing {
                    path: "sub/b.vortex".to_owned(),
                    size: Some(5),
                },
            ]
        );
        assert_eq!(mem.recorded_prefixes.lock().as_slice(), ["data/v1/"]);

        // Sub-prefix with leading slash is normalized cleanly.
        let sub = exec.drive(prefixed.list("/sub/").try_collect::<Vec<_>>())?;
        assert_eq!(
            sub,
            vec![FileListing {
                path: "sub/b.vortex".to_owned(),
                size: Some(5),
            }]
        );

        // `head` strips the prefix from returned `FileListing::path` and handles leading slashes.
        assert_eq!(
            exec.drive(prefixed.head("/a.vortex"))?,
            Some(FileListing {
                path: "a.vortex".to_owned(),
                size: Some(3),
            })
        );
        assert_eq!(exec.drive(prefixed.head("sibling.vortex"))?, None);

        // `open_read` reads through the prefix and rejects missing files with full prefixed path.
        let reader = exec.drive(prefixed.open_read("/sub/b.vortex"))?;
        let bytes = exec.drive(reader.read_at(0, 5, Alignment::none()))?;
        assert_eq!(exec.drive(bytes.to_host()).as_ref(), b"bbbbb");

        let missing = exec
            .drive(prefixed.open_read("missing.vortex"))
            .err()
            .expect("missing file must fail");
        assert!(
            missing
                .to_string()
                .contains("not found: data/v1/missing.vortex"),
            "{missing}"
        );

        // `delete` targets only the prefixed file and leaves sibling prefixes untouched.
        exec.drive(prefixed.delete("/a.vortex"))?;
        assert_eq!(exec.drive(prefixed.head("a.vortex"))?, None);
        assert!(exec.drive(mem.head("data/v10/sibling.vortex"))?.is_some());

        let del_err = exec
            .drive(prefixed.delete("a.vortex"))
            .expect_err("deleting missing file must fail");
        assert!(
            del_err
                .to_string()
                .contains("cannot delete missing: data/v1/a.vortex"),
            "{del_err}"
        );
        Ok(())
    }

    #[rstest]
    #[case::empty("")]
    #[case::single_slash("/")]
    #[case::multi_slash("///")]
    fn test_empty_prefix_does_not_prepend_leading_slash(
        #[case] empty_prefix: &str,
    ) -> VortexResult<()> {
        let exec = SimulatedExecutor::new();
        let mem = MemoryFileSystem::new(&[("root.vortex", b"root"), ("dir/x.vortex", b"xx")]);
        let root: FileSystemRef = Arc::clone(&mem) as FileSystemRef;
        let prefixed = root.with_prefix(empty_prefix.to_owned());

        let mut listed = exec.drive(prefixed.list("").try_collect::<Vec<_>>())?;
        listed.sort();
        assert_eq!(listed.len(), 2);
        assert_eq!(mem.recorded_prefixes.lock().as_slice(), [""]);

        assert_eq!(
            exec.drive(prefixed.head("root.vortex"))?,
            Some(FileListing {
                path: "root.vortex".to_owned(),
                size: Some(4),
            })
        );
        let reader = exec.drive(prefixed.open_read("/dir/x.vortex"))?;
        let bytes = exec.drive(reader.read_at(0, 2, Alignment::none()))?;
        assert_eq!(exec.drive(bytes.to_host()).as_ref(), b"xx");
        Ok(())
    }

    #[test]
    fn test_chained_with_prefix_and_special_characters() -> VortexResult<()> {
        let exec = SimulatedExecutor::new();
        let mem = MemoryFileSystem::new(&[("org/team/file%20~[1].vortex", b"special")]);
        let root: FileSystemRef = Arc::clone(&mem) as FileSystemRef;
        let nested = root
            .with_prefix("org".to_owned())
            .with_prefix("/team/".to_owned());

        let listed = exec.drive(nested.list("").try_collect::<Vec<_>>())?;
        assert_eq!(
            listed,
            vec![FileListing {
                path: "file%20~[1].vortex".to_owned(),
                size: Some(7),
            }]
        );
        assert_eq!(
            exec.drive(nested.head("file%20~[1].vortex"))?,
            Some(FileListing {
                path: "file%20~[1].vortex".to_owned(),
                size: Some(7),
            })
        );
        let reader = exec.drive(nested.open_read("file%20~[1].vortex"))?;
        let bytes = exec.drive(reader.read_at(0, 7, Alignment::none()))?;
        assert_eq!(exec.drive(bytes.to_host()).as_ref(), b"special");
        Ok(())
    }
}
