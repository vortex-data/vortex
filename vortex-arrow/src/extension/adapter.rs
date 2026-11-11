// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Connects typed extension conversions to the Arrow session's importer and exporter registries.
//!
//! [`ArrowExtVTable`] defines conversion in both directions. This adapter provides the separate
//! interfaces that [`ArrowSession`] registers and applies the vtable's [`ArrowMetadataPolicy`].
//! Keeping metadata handling here prevents plugins from forgetting it. The session checks that
//! completed fields and imported arrays preserve their dtypes.
//!
//! [`VORTEX_EXTENSION_METADATA_KEY`] names `vortex:extension:metadata`. This separate field entry
//! leaves `ARROW:extension:metadata` for the Arrow extension's own metadata. The codec in
//! [`crate::ext_metadata`] uses `vortex:extension:metadata:encoding` to mark the payload as base64
//! and checks that marker before decoding.

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_schema::Field;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::dtype::DType;
use vortex_array::dtype::extension::ExtDType;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_session::registry::Id;

use super::ArrowExtVTable;
use super::ArrowMetadataPolicy;
use crate::ArrowExport;
use crate::ArrowExportOptions;
use crate::ArrowExportVTable;
use crate::ArrowImport;
use crate::ArrowImportVTable;
use crate::ArrowSession;
use crate::ArrowSessionExt;
use crate::ext_metadata::VORTEX_EXTENSION_METADATA_KEY;
use crate::ext_metadata::read_vortex_metadata;
use crate::ext_metadata::write_vortex_metadata;

impl ArrowSession {
    /// Registers both directions of a typed extension using the vtable's metadata policy.
    ///
    /// The adapter handles Vortex metadata encoding, decoding, and target checks. The session
    /// verifies complete dtype identity for both typed and raw plugins.
    pub fn register_extension<V: ArrowExtVTable>(&self, vtable: V) {
        let adapter = Arc::new(ExtensionPlugin { vtable });
        let exporter = Arc::clone(&adapter);

        self.register_exporter(exporter);
        self.register_importer(adapter);
    }
}

/// Registers one typed extension as both an importer and an exporter with the same metadata policy.
#[derive(Debug)]
struct ExtensionPlugin<V> {
    vtable: V,
}

impl<V: ArrowExtVTable> ExtensionPlugin<V> {
    /// Rejects export targets whose Vortex metadata disagrees with `ext_dtype`.
    ///
    /// A target's metadata determines the meaning of exported values on import. Under
    /// [`ArrowMetadataPolicy::PreserveVortex`], a target with [`VORTEX_EXTENSION_METADATA_KEY`]
    /// **must** carry matching metadata. An absent payload adds no Vortex metadata constraint.
    /// Comparison uses decoded metadata so it does not depend on serialization details.
    fn validate_export_target(
        &self,
        ext_dtype: &ExtDType<V>,
        target: &Field,
        ctx: &ExecutionCtx,
    ) -> VortexResult<()> {
        if V::metadata_policy() != ArrowMetadataPolicy::PreserveVortex
            || !target
                .metadata()
                .contains_key(VORTEX_EXTENSION_METADATA_KEY)
        {
            return Ok(());
        }

        let native_dtype =
            V::from_arrow_field(target, &ctx.session().arrow())?.ok_or_else(|| {
                vortex_err!(
                    "Arrow field {:?} must have a supported native representation, got {}",
                    target.name(),
                    target.data_type()
                )
            })?;
        if let Some(actual) =
            read_vortex_metadata(target, &self.vtable, native_dtype.storage_dtype())?
        {
            vortex_ensure_eq!(
                actual.metadata::<V>(),
                ext_dtype.metadata(),
                "Arrow field {:?} must carry the expected Vortex metadata",
                target.name()
            );
        }

        Ok(())
    }
}

impl<V: ArrowExtVTable> ArrowExportVTable for ExtensionPlugin<V> {
    fn arrow_ext_id(&self) -> Id {
        self.vtable.arrow_ext_id()
    }

    fn vortex_id(&self) -> Id {
        self.vtable.id()
    }

    fn to_arrow_field(
        &self,
        name: &str,
        dtype: &DType,
        session: &ArrowSession,
    ) -> VortexResult<Option<Field>> {
        let Some(ext_dtype) = dtype.as_extension_opt() else {
            return Ok(None);
        };
        let Ok(ext_dtype) = ext_dtype.clone().try_downcast::<V>() else {
            return Ok(None);
        };

        let Some(mut field) = self.vtable.to_arrow_field(name, &ext_dtype, session)? else {
            return Ok(None);
        };

        // Import routing determines whether the native field needs a separate Vortex payload.
        // The session checks the completed field after any payload has been added.
        if V::metadata_policy() == ArrowMetadataPolicy::PreserveVortex
            && session.from_arrow_field(&field)? != *dtype
        {
            write_vortex_metadata(&mut field, &self.vtable, ext_dtype.metadata())?;
        }

        Ok(Some(field))
    }

    fn execute_arrow(
        &self,
        array: ArrayRef,
        target: &Field,
        options: &ArrowExportOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrowExport> {
        let Some(ext_dtype) = array.dtype().as_extension_opt() else {
            return Ok(ArrowExport::Unsupported(array));
        };
        let Ok(ext_dtype) = ext_dtype.clone().try_downcast::<V>() else {
            return Ok(ArrowExport::Unsupported(array));
        };

        self.validate_export_target(&ext_dtype, target, ctx)?;

        self.vtable.execute_arrow(array, target, options, ctx)
    }
}

impl<V: ArrowExtVTable> ArrowImportVTable for ExtensionPlugin<V> {
    fn arrow_ext_id(&self) -> Id {
        self.vtable.arrow_ext_id()
    }

    fn from_arrow_field(
        &self,
        field: &Field,
        session: &ArrowSession,
    ) -> VortexResult<Option<DType>> {
        let Some(native_dtype) = V::from_arrow_field(field, session)? else {
            return Ok(None);
        };

        // The plugin can map Arrow's physical storage to a different Vortex storage dtype. Restore
        // metadata against that inferred storage so the dtype plugin can validate the combination.
        let restored_dtype = if V::metadata_policy() == ArrowMetadataPolicy::PreserveVortex {
            read_vortex_metadata(field, &self.vtable, native_dtype.storage_dtype())?
        } else {
            None
        };

        Ok(Some(DType::Extension(
            restored_dtype.unwrap_or_else(|| native_dtype.erased()),
        )))
    }

    fn from_arrow_array(
        &self,
        array: ArrowArrayRef,
        field: &Field,
        dtype: &DType,
        session: &ArrowSession,
    ) -> VortexResult<ArrowImport> {
        let Some(ext_dtype) = dtype.as_extension_opt() else {
            return Ok(ArrowImport::Unsupported(array));
        };
        let Ok(ext_dtype) = ext_dtype.clone().try_downcast::<V>() else {
            return Ok(ArrowImport::Unsupported(array));
        };

        V::from_arrow_array(array, field, &ext_dtype, session)
    }
}
