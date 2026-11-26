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

const FOOTER_CACHE_CAPACITY_BYTES: u64 = 100 << 20;

/// Session state for multi-file scanning.
///
/// Provides a shared, in-memory footer cache so that repeated scans over the same files
/// avoid redundant footer I/O. The cache is bounded by approximate byte weight and lives as
/// long as the [`VortexSession`](vortex_session::VortexSession).
///
/// # Future Work
///
/// Consider generalizing this cache into [`VortexOpenOptions`](crate::VortexOpenOptions) so
/// that single-file opens also benefit from session-level footer caching.
#[derive(Clone)]
pub struct MultiFileSession {
    footer_cache: moka::sync::Cache<String, Footer>,
}

impl Default for MultiFileSession {
    fn default() -> Self {
        Self {
            footer_cache: new_footer_cache(FOOTER_CACHE_CAPACITY_BYTES),
        }
    }
}

fn new_footer_cache(max_capacity_bytes: u64) -> moka::sync::Cache<String, Footer> {
    moka::sync::Cache::builder()
        .max_capacity(max_capacity_bytes)
        .weigher(|_key, footer: &Footer| moka_weight_from_bytes(footer.cache_weight_bytes()))
        .build()
}

fn moka_weight_from_bytes(weight_bytes: usize) -> u32 {
    u32::try_from(weight_bytes).unwrap_or(u32::MAX)
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
    use std::sync::Arc;

    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::StructFields;
    use vortex_buffer::Alignment;
    use vortex_buffer::ByteBuffer;
    use vortex_buffer::ByteBufferMut;
    use vortex_layout::layouts::flat::FlatLayout;
    use vortex_layout::segments::SegmentId;
    use vortex_layout::session::LayoutSession;
    use vortex_session::VortexSession;
    use vortex_session::registry::ReadContext;

    use crate::footer::DeserializeStep;
    use crate::footer::SegmentSpec;

    use super::*;

    fn footer() -> Footer {
        let layout = FlatLayout::new(
            0,
            DType::Null,
            SegmentId::from(0),
            ReadContext::new([]),
        )
        .into_layout();
        Footer::new(layout, Arc::from([]), None, ReadContext::new([]))
    }

    fn parsed_wide_footer() -> vortex_error::VortexResult<Footer> {
        let dtype = DType::Struct(
            StructFields::from_iter((0..1024).map(|index| {
                (
                    format!("field_{index}"),
                    DType::Primitive(PType::I32, Nullability::NonNullable),
                )
            })),
            Nullability::NonNullable,
        );
        let layout = FlatLayout::new(
            1,
            dtype,
            SegmentId::from(0),
            ReadContext::new([]),
        )
        .into_layout();
        let session = VortexSession::empty().with::<LayoutSession>();
        let parts = Footer::new(
            layout,
            Arc::from([SegmentSpec {
                offset: 0,
                length: 1,
                alignment: Alignment::none(),
            }]),
            None,
            ReadContext::new([]),
        )
        .into_serializer(&session)
        .serialize()?;
        let total_len = parts.iter().map(ByteBuffer::len).sum();
        let mut file = ByteBufferMut::with_capacity(total_len);
        for part in parts {
            file.extend_from_slice(part.as_ref());
        }
        let file = file.freeze();
        let file_size = u64::try_from(file.len())?;
        let mut deserializer = Footer::deserializer(file, session).with_size(file_size);

        match deserializer.deserialize()? {
            DeserializeStep::Done(footer) => Ok(footer),
            _ => Err(vortex_error::vortex_err!(
                "complete serialized footer should deserialize in one step"
            )),
        }
    }

    #[test]
    fn test_default_footer_cache_uses_bytes_and_nonzero_weights() {
        assert_eq!(FOOTER_CACHE_CAPACITY_BYTES, 100 << 20);

        let session = MultiFileSession::default();
        let footer = footer();
        let weight = footer.cache_weight_bytes();
        assert!(weight > 0);
        session.put_footer("file", footer);
        session.footer_cache.run_pending_tasks();

        assert_eq!(
            session.footer_cache.weighted_size(),
            u64::try_from(weight).unwrap_or(u64::MAX)
        );
    }

    #[test]
    fn test_moka_footer_weight_saturates_at_u32_max() {
        if let Ok(weight_bytes) = usize::try_from(u64::from(u32::MAX) + 1) {
            assert_eq!(moka_weight_from_bytes(weight_bytes), u32::MAX);
        }
    }

    #[test]
    fn test_footer_cache_evicts_wide_footers_at_its_byte_limit()
    -> vortex_error::VortexResult<()> {
        let session = MultiFileSession::default();
        let footer = parsed_wide_footer()?;
        let entry_weight = u64::try_from(footer.cache_weight_bytes())?;
        assert!(entry_weight > 0);
        assert!(entry_weight <= FOOTER_CACHE_CAPACITY_BYTES);
        let expected_max_entries = FOOTER_CACHE_CAPACITY_BYTES / entry_weight;
        let total_entries = expected_max_entries + 16;

        for index in 0..total_entries {
            session.put_footer(&format!("file-{index}"), footer.clone());
        }
        session.footer_cache.run_pending_tasks();

        let entry_count = session.footer_cache.entry_count();
        assert!(entry_count > 0);
        assert!(entry_count < total_entries);
        assert!(entry_count <= expected_max_entries);
        assert!(session.footer_cache.weighted_size() <= FOOTER_CACHE_CAPACITY_BYTES);
        Ok(())
    }
}
