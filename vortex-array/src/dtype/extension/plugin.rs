// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::extension::ExtDType;
use crate::dtype::extension::ExtDTypeRef;
use crate::dtype::extension::ExtId;
use crate::dtype::extension::ExtVTable;

/// Reference-counted pointer to an extension dtype plugin.
pub type ExtDTypePluginRef = Arc<dyn ExtDTypePlugin>;

/// Registry trait for ID-based deserialization of extension dtypes.
///
/// Plugins are registered in the session by their [`ExtId`]. When a serialized extension dtype
/// is encountered, the session resolves the ID to the plugin and calls [`deserialize`] to
/// reconstruct the value as an [`ExtDTypeRef`].
///
/// [`deserialize`]: ExtDTypePlugin::deserialize
pub trait ExtDTypePlugin: 'static + Send + Sync + Debug {
    /// Returns the ID for this extension type.
    fn id(&self) -> ExtId;

    /// Deserialize an extension type from serialized metadata and the stored storage dtype.
    ///
    /// The implementation for every [`ExtVTable`] rebuilds and validates the metadata. The result
    /// takes its top-level nullability from `storage_dtype`.
    ///
    /// # Errors
    ///
    /// Returns an error if the metadata cannot be rebuilt or validated, or if `storage_dtype`
    /// differs from the declared storage dtype in anything other than its top-level nullability,
    /// including a nested nullability.
    fn deserialize(&self, data: &[u8], storage_dtype: DType) -> VortexResult<ExtDTypeRef>;
}

impl<V: ExtVTable> ExtDTypePlugin for V {
    fn id(&self) -> ExtId {
        ExtVTable::id(self)
    }

    fn deserialize(&self, data: &[u8], storage_dtype: DType) -> VortexResult<ExtDTypeRef> {
        let metadata = ExtVTable::deserialize_metadata(self, data, &storage_dtype)?;
        let ext_dtype = ExtDType::try_with_vtable(self.clone(), metadata, Nullability::Nullable)?;
        let declared = ext_dtype.storage_dtype();

        // Arrays decode their storage child with the declared storage dtype, so a stored form that
        // differs, even only in nested nullability, would be misread.
        //
        // NB: compare the nullable forms before the stored top-level nullability applies. A
        // declared storage without a non-nullable form, such as `DType::Null`, would panic if a
        // non-nullable stored level applied first. After a match, the stored dtype is the
        // declaration at the stored level, so applying that level cannot panic.
        if *declared != storage_dtype.as_nullable() {
            // Show the declaration at the stored top-level nullability where it has that form.
            let expected = if has_non_nullable_form(declared) {
                declared.with_nullability(storage_dtype.nullability())
            } else {
                declared.clone()
            };
            vortex_bail!(
                "{} storage must be {expected}, got {storage_dtype}",
                ext_dtype.id()
            );
        }

        Ok(ext_dtype.with_nullability(storage_dtype.nullability()))
    }
}

/// Returns whether `dtype` has a non-nullable form, without which [`DType::with_nullability`]
/// panics. [`DType::Null`] has none, and neither has an extension dtype whose storage has none.
fn has_non_nullable_form(dtype: &DType) -> bool {
    match dtype {
        DType::Null => false,
        DType::Extension(ext) => has_non_nullable_form(ext.storage_dtype()),
        _ => true,
    }
}
