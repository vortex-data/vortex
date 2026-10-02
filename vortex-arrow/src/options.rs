// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Per-export options, including options defined by external Arrow export plugins.

use std::any::Any;
use std::any::TypeId;
use std::fmt;
use std::sync::Arc;

/// Typed options propagated to every nested array in an Arrow export.
///
/// Export plugins can define their own option types without modifying this crate. Each type has
/// at most one value; absent options leave the exporter's default behavior unchanged.
///
/// ```
/// use vortex_arrow::{ArrowExportOptions, CompactBuffers};
///
/// let options = ArrowExportOptions::default().with(CompactBuffers(false));
/// assert_eq!(options.get::<CompactBuffers>(), Some(&CompactBuffers(false)));
/// ```
#[derive(Clone, Default)]
pub struct ArrowExportOptions {
    values: Vec<(TypeId, Arc<dyn Any + Send + Sync>)>,
}

impl ArrowExportOptions {
    /// Set an option, replacing any previous value of the same type.
    pub fn with<T: Any + Send + Sync>(mut self, option: T) -> Self {
        let id = TypeId::of::<T>();
        if let Some((_, value)) = self.values.iter_mut().find(|(key, _)| *key == id) {
            *value = Arc::new(option);
        } else {
            self.values.push((id, Arc::new(option)));
        }
        self
    }

    /// Look up an option defined by this crate or an external exporter.
    pub fn get<T: Any + Send + Sync>(&self) -> Option<&T> {
        self.values
            .iter()
            .find(|(key, _)| *key == TypeId::of::<T>())
            .and_then(|(_, value)| value.downcast_ref())
    }
}

impl fmt::Debug for ArrowExportOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ArrowExportOptions")
            .field("option_count", &self.values.len())
            .finish_non_exhaustive()
    }
}

/// Whether to compact string and binary view buffers during Arrow export.
///
/// Enabled by default. Disabling compaction avoids scans and copies, but the exported array may
/// retain backing data outside the selected rows. Values and the Arrow type are unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompactBuffers(pub bool);

impl Default for CompactBuffers {
    fn default() -> Self {
        Self(true)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicU32;
    use std::sync::atomic::Ordering;

    use arrow_schema::DataType;
    use arrow_schema::Field;
    use vortex_array::ArrayRef;
    use vortex_array::ExecutionCtx;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::ListArray;
    use vortex_array::dtype::DType;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::registry::CachedId;
    use vortex_session::registry::Id;

    use super::*;
    use crate::ArrowExport;
    use crate::ArrowExportVTable;
    use crate::ArrowSession;
    use crate::ArrowSessionExt;

    #[test]
    fn typed_options_are_independent() {
        struct PluginOption(u32);
        let original = ArrowExportOptions::default().with(CompactBuffers(false));
        let changed = original
            .clone()
            .with(CompactBuffers(true))
            .with(PluginOption(42));
        assert_eq!(
            original.get::<CompactBuffers>(),
            Some(&CompactBuffers(false))
        );
        assert!(original.get::<PluginOption>().is_none());
        assert_eq!(changed.get::<CompactBuffers>(), Some(&CompactBuffers(true)));
        assert_eq!(changed.get::<PluginOption>().map(|value| value.0), Some(42));
        assert!(
            ArrowExportOptions::default()
                .get::<CompactBuffers>()
                .is_none()
        );
    }

    #[derive(Debug)]
    struct Plugin;

    struct PluginOption(Arc<AtomicU32>);

    static TEST_OPTION_ID: CachedId = CachedId::new("test.options");

    impl ArrowExportVTable for Plugin {
        fn arrow_ext_id(&self) -> Id {
            *TEST_OPTION_ID
        }

        fn vortex_id(&self) -> Id {
            *TEST_OPTION_ID
        }

        fn to_arrow_field(
            &self,
            _name: &str,
            _dtype: &DType,
            _session: &ArrowSession,
        ) -> VortexResult<Option<Field>> {
            Ok(None)
        }

        fn execute_arrow(
            &self,
            array: ArrayRef,
            _target: &Field,
            _ctx: &mut ExecutionCtx,
        ) -> VortexResult<ArrowExport> {
            Ok(ArrowExport::Unsupported(array))
        }

        fn execute_arrow_with_options(
            &self,
            array: ArrayRef,
            _target: &Field,
            options: &ArrowExportOptions,
            _ctx: &mut ExecutionCtx,
        ) -> VortexResult<ArrowExport> {
            options
                .get::<PluginOption>()
                .expect("external option")
                .0
                .store(42, Ordering::Relaxed);
            Ok(ArrowExport::Unsupported(array))
        }
    }

    #[test]
    fn external_options_reach_nested_exporters() -> VortexResult<()> {
        let session = array_session();
        session.arrow().register_exporter(Arc::new(Plugin));
        let mut ctx = session.create_execution_ctx();
        let child = Field::new("item", DataType::Int32, false).with_metadata(
            [("ARROW:extension:name".to_owned(), "test.options".to_owned())]
                .into_iter()
                .collect(),
        );
        let field = Field::new("list", DataType::List(Arc::new(child)), false);
        let array = ListArray::try_new(
            buffer![1i32, 2, 3].into_array(),
            buffer![0i32, 3].into_array(),
            Validity::NonNullable,
        )?
        .into_array();
        let called = Arc::new(AtomicU32::new(0));
        let options = ArrowExportOptions::default().with(PluginOption(Arc::clone(&called)));
        let arrow =
            session
                .arrow()
                .execute_arrow_with_options(array, Some(&field), &options, &mut ctx)?;
        assert_eq!(arrow.len(), 1);
        assert_eq!(called.load(Ordering::Relaxed), 42);
        Ok(())
    }
}
