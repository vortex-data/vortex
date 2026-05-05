// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Plugin trait for exporting extension-typed arrays to Arrow.

use std::fmt::Debug;
use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_schema::DataType;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::dtype::extension::ExtDTypeRef;
use crate::dtype::extension::ExtId;
use crate::executor::ExecutionCtx;

/// Shared reference to an [`ArrowExportPlugin`].
pub type ArrowExportPluginRef = Arc<dyn ArrowExportPlugin>;

/// Plugin for exporting an extension-typed array to Arrow.
///
/// Extension types register an [`ArrowExportPlugin`] to own the mapping from their extension
/// dtype to Arrow [`DataType`] and the conversion of array data. The core arrow executor
/// delegates to these plugins instead of hard-coding extension-specific behavior.
pub trait ArrowExportPlugin: 'static + Send + Sync + Debug {
    /// The extension type id this plugin handles.
    fn id(&self) -> ExtId;

    /// Preferred Arrow [`DataType`] for this extension type, given its metadata.
    ///
    /// Called by the executor when no target Arrow type was supplied by the caller.
    fn to_arrow_data_type(&self, ext_dtype: &ExtDTypeRef) -> VortexResult<DataType>;

    /// Execute the extension-typed `array` to an Arrow array of type `target`.
    ///
    /// `array` is the full extension-typed array; the plugin is responsible for unwrapping it
    /// to storage. If `target` is not a type this plugin can produce, the plugin must
    /// `vortex_bail!` rather than attempting a best-effort conversion.
    fn execute_to_arrow(
        &self,
        array: ArrayRef,
        target: &DataType,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrowArrayRef>;
}
