// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Formatter;
use std::fs;
use std::io;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use futures::stream;
use futures::stream::BoxStream;
use vortex_error::VortexResult;

use crate::VortexReadAt;
use crate::filesystem::FileListing;
use crate::filesystem::FileSystem;
use crate::runtime::Handle;
use crate::std_file::FileReadAt;

/// A FileSystem over local filesystem.
pub struct StdFileSystem {
    handle: Handle,
}

impl Debug for StdFileSystem {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StdFileSystem").finish()
    }
}

impl StdFileSystem {
    pub fn new(handle: Handle) -> Self {
        Self { handle }
    }
}

fn walk(dir: &Path, out: &mut Vec<FileListing>) -> io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            walk(&entry.path(), out)?;
        } else {
            let metadata = entry.metadata()?;
            out.push(FileListing {
                path: entry.path().to_string_lossy().into_owned(),
                size: Some(metadata.len()),
            });
        }
    }
    Ok(())
}

#[async_trait]
impl FileSystem for StdFileSystem {
    fn list(&self, prefix: &str) -> BoxStream<'_, VortexResult<FileListing>> {
        let dir = PathBuf::from(prefix);
        let listing = self.handle.spawn_blocking(move || {
            let mut out = Vec::new();
            walk(&dir, &mut out)?;
            Ok::<_, io::Error>(out)
        });
        stream::once(listing)
            .flat_map(|result| match result {
                Ok(listings) => stream::iter(listings.into_iter().map(Ok)).boxed(),
                Err(e) => stream::once(async move { Err(e.into()) }).boxed(),
            })
            .boxed()
    }

    async fn head(&self, path: &str) -> VortexResult<Option<FileListing>> {
        let path = path.to_owned();
        self.handle
            .spawn_blocking(move || match fs::metadata(&path) {
                Ok(metadata) if metadata.is_file() => Ok(Some(FileListing {
                    path,
                    size: Some(metadata.len()),
                })),
                Ok(_) => Ok(None),
                Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
                Err(e) => Err(e.into()),
            })
            .await
    }

    async fn open_read(&self, path: &str) -> VortexResult<Arc<dyn VortexReadAt>> {
        let handle = self.handle.clone();
        let path = path.to_owned();
        let reader = self
            .handle
            .spawn_blocking(move || FileReadAt::open(path, handle))
            .await?;
        Ok(Arc::new(reader))
    }

    async fn delete(&self, path: &str) -> VortexResult<()> {
        let path = path.to_owned();
        self.handle
            .spawn_blocking(move || fs::remove_file(path))
            .await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;
    use std::fs;
    use std::io;
    use std::sync::Arc;

    use futures::TryStreamExt;
    use tempfile::tempdir;
    use vortex_buffer::Alignment;
    use vortex_error::VortexError;
    use vortex_error::VortexResult;

    use super::*;
    use crate::ReadAtRequest;
    use crate::filesystem::FileSystemRef;
    use crate::test_harness::SimulatedExecutor;

    fn io_error_kind(err: &VortexError) -> Option<io::ErrorKind> {
        let mut current: Option<&(dyn Error + 'static)> = Some(err);
        while let Some(source) = current {
            if let Some(io_err) = source.downcast_ref::<io::Error>() {
                return Some(io_err.kind());
            }
            current = source.source();
        }
        None
    }

    #[test]
    fn test_std_fs_list_recursive_head_open_read_and_delete() -> VortexResult<()> {
        let exec = SimulatedExecutor::new();
        let dir = tempdir()?;
        let sub = dir.path().join("nested").join("deep");
        fs::create_dir_all(&sub)?;

        let root_file = dir.path().join("root.vortex");
        let deep_file = sub.join("leaf.vortex");
        fs::write(&root_file, b"hello")?;
        fs::write(&deep_file, b"0123456789")?;

        let fs_impl = StdFileSystem::new(exec.handle());
        assert_eq!(format!("{fs_impl:?}"), "StdFileSystem");

        // Recursive listing finds files in nested directories with exact sizes.
        let mut listed = exec.drive(
            fs_impl
                .list(&dir.path().to_string_lossy())
                .try_collect::<Vec<_>>(),
        )?;
        listed.sort();
        let mut expected = vec![
            FileListing {
                path: root_file.to_string_lossy().into_owned(),
                size: Some(5),
            },
            FileListing {
                path: deep_file.to_string_lossy().into_owned(),
                size: Some(10),
            },
        ];
        expected.sort();
        assert_eq!(listed, expected);

        // `head` returns metadata for files, and `None` for missing paths or directories.
        let root_str = root_file.to_string_lossy().into_owned();
        assert_eq!(
            exec.drive(fs_impl.head(&root_str))?,
            Some(FileListing {
                path: root_str.clone(),
                size: Some(5),
            })
        );
        assert_eq!(
            exec.drive(fs_impl.head(&dir.path().join("missing.vortex").to_string_lossy()))?,
            None
        );
        assert_eq!(exec.drive(fs_impl.head(&sub.to_string_lossy()))?, None);

        // `open_read` reads both `read_at` and `read_ranges`.
        let deep_str = deep_file.to_string_lossy().into_owned();
        let reader = exec.drive(fs_impl.open_read(&deep_str))?;
        assert_eq!(exec.drive(reader.size())?, 10);
        let slice = exec.drive(reader.read_at(2, 4, Alignment::none()))?;
        assert_eq!(exec.drive(slice.to_host()).as_ref(), b"2345");

        let range_results = exec.drive(
            reader
                .read_ranges(Arc::from([
                    ReadAtRequest::new(0, 3, Alignment::none()),
                    ReadAtRequest::new(7, 3, Alignment::none()),
                ]))
                .collect::<Vec<_>>(),
        );
        assert_eq!(range_results.len(), 2);
        for (req, res) in range_results {
            let bytes = exec.drive(res?.to_host());
            match req.offset {
                0 => assert_eq!(bytes.as_ref(), b"012"),
                7 => assert_eq!(bytes.as_ref(), b"789"),
                other => panic!("unexpected offset {other}"),
            }
        }

        // `delete` removes the file and makes subsequent `head` return `None`.
        exec.drive(fs_impl.delete(&root_str))?;
        assert_eq!(exec.drive(fs_impl.head(&root_str))?, None);
        Ok(())
    }

    #[test]
    fn test_std_fs_empty_dir_and_error_edges() -> VortexResult<()> {
        let exec = SimulatedExecutor::new();
        let dir = tempdir()?;
        let fs_impl = StdFileSystem::new(exec.handle());

        // Empty directory lists zero entries.
        let empty = exec.drive(
            fs_impl
                .list(&dir.path().to_string_lossy())
                .try_collect::<Vec<_>>(),
        )?;
        assert!(empty.is_empty());

        // Listing a missing directory yields NotFound.
        let missing = dir.path().join("does-not-exist");
        let list_missing_err = exec
            .drive(
                fs_impl
                    .list(&missing.to_string_lossy())
                    .try_collect::<Vec<_>>(),
            )
            .expect_err("listing missing directory must fail");
        assert_eq!(
            io_error_kind(&list_missing_err),
            Some(io::ErrorKind::NotFound),
            "{list_missing_err}"
        );

        // Listing a regular file path (not a directory) yields NotADirectory.
        let file_path = dir.path().join("not-a-dir.vortex");
        fs::write(&file_path, b"abc")?;
        let list_file_err = exec
            .drive(
                fs_impl
                    .list(&file_path.to_string_lossy())
                    .try_collect::<Vec<_>>(),
            )
            .expect_err("listing a regular file must fail");
        assert_eq!(
            io_error_kind(&list_file_err),
            Some(io::ErrorKind::NotADirectory),
            "{list_file_err}"
        );

        // Opening or deleting a missing file yields NotFound.
        let open_err = exec
            .drive(fs_impl.open_read(&missing.to_string_lossy()))
            .err()
            .expect("opening missing file must fail");
        assert_eq!(
            io_error_kind(&open_err),
            Some(io::ErrorKind::NotFound),
            "{open_err}"
        );

        let delete_missing_err = exec
            .drive(fs_impl.delete(&missing.to_string_lossy()))
            .expect_err("deleting missing file must fail");
        assert_eq!(
            io_error_kind(&delete_missing_err),
            Some(io::ErrorKind::NotFound),
            "{delete_missing_err}"
        );

        // Deleting a directory fails (`remove_file` yields IsADirectory on Linux and
        // PermissionDenied on macOS/Windows).
        let delete_dir_err = exec
            .drive(fs_impl.delete(&dir.path().to_string_lossy()))
            .expect_err("deleting a directory must fail");
        assert!(
            matches!(
                io_error_kind(&delete_dir_err),
                Some(io::ErrorKind::IsADirectory | io::ErrorKind::PermissionDenied)
            ),
            "{delete_dir_err}"
        );
        Ok(())
    }

    #[test]
    fn test_std_fs_glob_exact_and_wildcard() -> VortexResult<()> {
        let exec = SimulatedExecutor::new();
        let dir = tempdir()?;
        let sub = dir.path().join("part");
        fs::create_dir_all(&sub)?;

        let a = sub.join("a.vortex");
        let b = sub.join("b.vortex");
        let ignored = sub.join("notes.txt");
        fs::write(&a, b"1")?;
        fs::write(&b, b"22")?;
        fs::write(&ignored, b"333")?;

        let fs_ref: FileSystemRef = Arc::new(StdFileSystem::new(exec.handle()));

        // Exact path glob uses `head`.
        let exact = exec.drive(fs_ref.glob(&a.to_string_lossy())?.try_collect::<Vec<_>>())?;
        assert_eq!(
            exact,
            vec![FileListing {
                path: a.to_string_lossy().into_owned(),
                size: Some(1),
            }]
        );

        // Wildcard glob filters listed files.
        let pattern = format!("{}/*.vortex", sub.to_string_lossy());
        let mut matched = exec.drive(fs_ref.glob(&pattern)?.try_collect::<Vec<_>>())?;
        matched.sort();
        let mut expected = vec![
            FileListing {
                path: a.to_string_lossy().into_owned(),
                size: Some(1),
            },
            FileListing {
                path: b.to_string_lossy().into_owned(),
                size: Some(2),
            },
        ];
        expected.sort();
        assert_eq!(matched, expected);
        Ok(())
    }
}
