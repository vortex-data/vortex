// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arrow interoperability for Vortex extension types.
//!
//! Implement [`ArrowExtVTable`] and register it with [`ArrowSession::register_extension`] to enable
//! Arrow import and export. Registration wraps the callbacks in an adapter that handles Vortex
//! metadata, while the session validates schema round trips. Plugins only implement conversions.
//!
//! A Vortex extension does not need an equivalent standard Arrow type. It can use a custom Arrow
//! extension name, or decline native export so the session exports storage with the Vortex ID and
//! metadata. When a plugin supplies an Arrow representation, its metadata may not describe the full
//! Vortex dtype. Handling that gap in the adapter prevents plugins from silently losing metadata.
//!
//! Declare an [`ArrowMetadataPolicy`] in [`ArrowExtVTable::metadata_policy`] based on whether the
//! Arrow representation preserves the full Vortex dtype on its own. Callbacks leave transport to
//! the adapter and import arrays with the restored dtype they receive. Plugins remain responsible
//! for preserving values and lengths.

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_schema::Field;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtVTable;
use vortex_error::VortexResult;
use vortex_session::registry::Id;

use crate::ArrowExport;
use crate::ArrowExportOptions;
use crate::ArrowImport;
use crate::ArrowSession;

/// Whether an Arrow representation needs separate Vortex metadata or not to restore the full dtype.
///
/// On schema export, the session imports the generated field and checks that it reproduces the
/// original dtype, including metadata, storage, and nullability. On successful array import, the
/// session checks that the returned array has the dtype inferred from the field.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArrowMetadataPolicy {
    /// Rely on the Arrow representation alone to preserve the complete Vortex dtype.
    ///
    /// Choose this only when the plugin's Arrow field can reconstruct the dtype without additional
    /// Vortex metadata. The adapter neither writes nor reads a separate Vortex metadata payload.
    ArrowNative,

    /// Preserve Vortex metadata separately when the Arrow representation alone is insufficient.
    ///
    /// If the exported field does not reconstruct the original dtype, the adapter adds a Vortex
    /// metadata payload before the session checks the completed field. The Arrow extension's own
    /// metadata stays unchanged. On import, the payload restores metadata over the plugin's inferred
    /// storage.
    ///
    /// An absent payload uses the native interpretation. A required empty payload is still
    /// written, because a codec can distinguish it from absence.
    PreserveVortex,
}

/// Native Arrow conversion callbacks for one Vortex extension type.
///
/// Register through [`ArrowSession::register_extension`], and use the session's conversion APIs.
/// Calling these callbacks directly bypasses metadata transport and schema validation. Callbacks
/// **must** describe only the native Arrow representation, leaving Vortex metadata transport to the
/// adapter. Register the owning dtype in the Vortex session as well to enable its Vortex ID path.
///
/// Register the dtype and its Arrow conversions in the same session:
///
/// ```
/// use vortex_array::dtype::session::DTypeSessionExt;
/// use vortex_arrow::{ArrowExtVTable, ArrowSessionExt};
/// use vortex_session::VortexSession;
///
/// fn register<V: ArrowExtVTable>(session: &VortexSession, vtable: V) {
///     session.dtypes().register(vtable.clone());
///     session.arrow().register_extension(vtable);
/// }
/// ```
pub trait ArrowExtVTable: ExtVTable {
    /// The metadata policy used by the adapter for this extension type.
    ///
    /// The result **must** remain the same across calls and sessions.
    fn metadata_policy() -> ArrowMetadataPolicy;

    /// The native Arrow extension ID used for both import and export.
    fn arrow_ext_id(&self) -> Id;

    /// Builds the native field, or declines so that the session uses the Vortex ID path.
    ///
    /// Use [`Field::try_with_extension_type`] for standard Arrow extension metadata, or
    /// [`crate::ext_metadata::set_arrow_extension_name`] for a custom name. The adapter adds any
    /// Vortex metadata before the session validates the completed field against the source dtype.
    fn to_arrow_field(
        &self,
        name: &str,
        ext_dtype: &ExtDType<Self>,
        session: &ArrowSession,
    ) -> VortexResult<Option<Field>>;

    /// Converts an array of this extension type to the native target representation.
    ///
    /// The adapter checks the source extension type before calling this method. Return
    /// [`ArrowExport::Unsupported`] when the target is unsupported. Child exports **must** receive
    /// the same `options` through [`crate::ExecuteArrowExt::execute_arrow_with_options`].
    fn execute_arrow(
        &self,
        array: ArrayRef,
        target: &Field,
        options: &ArrowExportOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrowExport>;

    /// Interprets the native field, without applying Vortex side metadata.
    ///
    /// Infer the Vortex storage representation here, which can differ from Arrow's physical type.
    /// Return `None` when the field is unsupported.
    fn from_arrow_field(
        field: &Field,
        session: &ArrowSession,
    ) -> VortexResult<Option<ExtDType<Self>>>;

    /// Imports native values with the extension dtype restored by the adapter.
    ///
    /// `ext_dtype` includes any Vortex metadata restored by the adapter, so this callback does not
    /// need to decode it. Inferring the dtype again from the native Arrow representation could lose
    /// that metadata.
    ///
    /// A successful result **must** have exactly `ext_dtype` and the input length. Return
    /// [`ArrowImport::Unsupported`] when the representation is unsupported. Import storage through
    /// the session with a nullability rather than the extension field, to avoid dispatching this
    /// plugin recursively.
    fn from_arrow_array(
        array: ArrowArrayRef,
        field: &Field,
        ext_dtype: &ExtDType<Self>,
        session: &ArrowSession,
    ) -> VortexResult<ArrowImport>;
}

mod adapter;

#[cfg(test)]
mod tests;
