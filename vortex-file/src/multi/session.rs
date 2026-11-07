// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Session extension for multi-file scanning, providing a shared footer cache.

use std::any::Any;
use std::fmt;
use std::fmt::Debug;

use vortex_session::SessionExt;
use vortex_session::SessionGuard;
use vortex_session::SessionVar;

use crate::footer::Footer;

/// Session state for multi-file scanning.
///
/// Provides a shared, in-memory footer cache so that repeated scans over the same files
/// avoid redundant footer I/O. The cache is bounded by the approximate memory its footers hold
/// (see [`Footer::approx_byte_size`]) and lives as long as the
/// [`VortexSession`](vortex_session::VortexSession).
///
/// # Future Work
///
/// Consider generalizing this cache into [`VortexOpenOptions`](crate::VortexOpenOptions) so
/// that single-file opens also benefit from session-level footer caching.
#[derive(Clone)]
pub struct MultiFileSession {
    footer_cache: moka::sync::Cache<String, Footer>,
}

impl MultiFileSession {
    const DEFAULT_FOOTER_CACHE_BYTES: u64 = 100 << 20;

    /// Creates a session whose footer cache holds footers totalling about `max_bytes`.
    pub(crate) fn new(max_bytes: u64) -> Self {
        Self {
            footer_cache: moka::sync::Cache::builder()
                // Capacity and weigher are in KiB
                .max_capacity(max_bytes / 1024)
                .weigher(|_k, footer: &Footer| footer_weight_kib(footer))
                .build(),
        }
    }
}

/// Weighs a footer in KiB, rounded up so that no footer is free to cache.
fn footer_weight_kib(footer: &Footer) -> u32 {
    footer.approx_byte_size().map_or(10, |bytes| {
        u32::try_from(bytes.div_ceil(1024).max(1)).unwrap_or(u32::MAX)
    })
}

impl Default for MultiFileSession {
    fn default() -> Self {
        Self::new(Self::DEFAULT_FOOTER_CACHE_BYTES)
    }
}

impl Debug for MultiFileSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MultiFileSession")
            .field("footer_cache_entry_count", &self.footer_cache.entry_count())
            .finish()
    }
}

impl MultiFileSession {
    /// Retrieve a cached footer for the given file path.
    pub fn get_footer(&self, path: &str) -> Option<Footer> {
        self.footer_cache.get(path)
    }

    /// Store a footer under the given file path.
    pub(crate) fn put_footer(&self, path: &str, footer: Footer) {
        self.footer_cache.insert(path.to_string(), footer);
    }
}

impl SessionVar for MultiFileSession {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Extension trait for accessing the [`MultiFileSession`] from a session.
pub(super) trait MultiFileSessionExt: SessionExt {
    /// Returns a reference to the [`MultiFileSession`] state.
    fn multi_file(&self) -> SessionGuard<'_, MultiFileSession> {
        self.get::<MultiFileSession>()
    }
}

impl<S: SessionExt> MultiFileSessionExt for S {}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::arrays::VarBinArray;
    use vortex_buffer::ByteBuffer;
    use vortex_buffer::ByteBufferMut;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_io::session::RuntimeSession;
    use vortex_layout::session::LayoutSession;
    use vortex_session::VortexSession;

    use super::*;
    use crate::OpenOptionsSessionExt;
    use crate::WriteOptionsSessionExt;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty()
            .with::<LayoutSession>()
            .with::<RuntimeSession>();
        crate::register_default_encodings(&session);
        crate::enable_all_registered_array_encodings(&session);
        session
    });

    /// A struct of `groups` structs of ten columns each.
    fn nested_array(groups: usize) -> VortexResult<ArrayRef> {
        let columns = (0..groups)
            .map(|g| {
                let leaves = (0..10)
                    .map(|l| {
                        let leaf = if l % 2 == 0 {
                            buffer![1i64, 2, 3].into_array()
                        } else {
                            VarBinArray::from(vec!["a", "b", "c"]).into_array()
                        };
                        (format!("leaf_column_{l:02}"), leaf)
                    })
                    .collect::<Vec<_>>();
                Ok((
                    format!("group_{g:03}"),
                    StructArray::from_fields(&leaves)?.into_array(),
                ))
            })
            .collect::<VortexResult<Vec<_>>>()?;
        Ok(StructArray::from_fields(&columns)?.into_array())
    }

    async fn footer(groups: usize) -> VortexResult<Footer> {
        let mut buf = ByteBufferMut::empty();
        SESSION
            .write_options()
            .write(&mut buf, nested_array(groups)?.to_array_stream())
            .await?;
        Ok(SESSION
            .open_options()
            .open_buffer(ByteBuffer::from(buf))?
            .footer()
            .clone())
    }

    #[tokio::test]
    async fn footer_size_scales_with_fields() -> VortexResult<()> {
        let narrow = footer(10).await?;
        let wide = footer(100).await?;
        let narrow_size = narrow.approx_byte_size().unwrap_or_default();
        let wide_size = wide.approx_byte_size().unwrap_or_default();
        assert!(
            wide_size > 8 * narrow_size,
            "a footer with 10x the fields weighs {wide_size} bytes against {narrow_size}"
        );
        // The serialized footer takes about 200 bytes per leaf; the parsed one holds far more.
        assert!(wide_size > 1000 * 1000, "{wide_size}");
        Ok(())
    }

    #[tokio::test]
    async fn footer_cache_evicts_wide_footers_at_its_byte_limit() -> VortexResult<()> {
        let wide = footer(100).await?;
        let session = MultiFileSession::new(16 << 20);
        for i in 0..50 {
            session.put_footer(&format!("file-{i}.vortex"), wide.clone());
        }
        session.footer_cache.run_pending_tasks();

        // Each footer holds over 1 MB, so 16 MiB fits at most 16 of them.
        let entries = session.footer_cache.entry_count();
        assert!((1..=16).contains(&entries), "{entries} footers cached");
        assert!(session.footer_cache.weighted_size() <= 16 << 10);
        Ok(())
    }

    #[tokio::test]
    async fn footer_under_a_kib_still_weighs() -> VortexResult<()> {
        let footer = footer(1).await?;
        for (bytes, kib) in [(0, 1), (1024, 1), (1025, 2)] {
            assert_eq!(
                footer_weight_kib(&footer.clone().with_approx_byte_size(bytes)),
                kib
            );
        }
        Ok(())
    }
}
