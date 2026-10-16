// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![warn(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]
#![warn(clippy::missing_errors_doc)]
#![warn(clippy::missing_panics_doc)]
#![warn(clippy::missing_safety_doc)]

//! UUID extension type for Vortex.
//!
//! Provides a UUID extension type backed by `FixedSizeList(Primitive(U8), 16)` storage. Each UUID
//! is stored as 16 bytes in big-endian (network) byte order, matching [RFC 4122] and Arrow's
//! [canonical UUID extension].
//!
//! [`initialize`] registers the extension dtype and the Arrow import/export plugins that map it
//! to Arrow's `arrow.uuid` extension type over `FixedSizeBinary[16]` storage.
//!
//! [RFC 4122]: https://www.rfc-editor.org/rfc/rfc4122
//! [canonical UUID extension]: https://arrow.apache.org/docs/format/CanonicalExtensions.html#uuid

mod arrow;
mod metadata;
mod vtable;

use std::sync::Arc;

pub use metadata::UuidMetadata;
use vortex_array::dtype::session::DTypeSessionExt;
use vortex_arrow::ArrowSessionExt;
use vortex_session::VortexSession;

/// The VTable for the UUID extension type.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Uuid;

/// Register UUID extension support with a session.
///
/// This registers the `vortex.uuid` extension dtype along with the Arrow exporter and importer
/// for Arrow's canonical `arrow.uuid` extension type.
pub fn initialize(session: &VortexSession) {
    session.dtypes().register(Uuid);
    session.arrow().register_exporter(Arc::new(Uuid));
    session.arrow().register_importer(Arc::new(Uuid));
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::NumericalAggregateOpts;
    use vortex_array::aggregate_fn::fns::min_max::MinMaxResult;
    use vortex_array::aggregate_fn::fns::min_max::min_max;
    use vortex_array::array_session;
    use vortex_array::arrays::ExtensionArray;
    use vortex_array::arrays::FixedSizeListArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::dtype::extension::ExtDType;
    use vortex_array::validity::Validity;
    use vortex_error::VortexResult;

    use crate::Uuid;
    use crate::UuidMetadata;

    /// UUID extrema order by unsigned big-endian bytes, as [`uuid::Uuid`], Postgres, DuckDB and
    /// Parquet order UUIDs, and skip null UUIDs.
    #[test]
    fn min_max_orders_like_uuid() -> VortexResult<()> {
        let uuids = [
            Some(0x7fff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_u128),
            None,
            // Above every 0x7f-led UUID only when bytes compare unsigned.
            Some(0x8000_0000_0000_0000_0000_0000_0000_0000),
            Some(0x0000_0000_0000_0000_0000_0000_0000_0100),
            // Below 0x...0100 only when bytes compare big-endian.
            Some(0x0000_0000_0000_0000_0000_0000_0000_00ff),
        ]
        .map(|uuid| uuid.map(uuid::Uuid::from_u128));

        let bytes = uuids
            .iter()
            .flat_map(|uuid| uuid.unwrap_or_default().into_bytes());
        let storage = FixedSizeListArray::try_new(
            PrimitiveArray::from_iter(bytes).into_array(),
            16,
            Validity::from_iter(uuids.iter().map(Option::is_some)),
            uuids.len(),
        )?;
        let ext_dtype =
            ExtDType::try_with_vtable(Uuid, UuidMetadata::default(), storage.dtype().clone())?
                .erased();
        let array = ExtensionArray::new(ext_dtype, storage.into_array()).into_array();

        let mut ctx = array_session().create_execution_ctx();
        let row_of = |uuid: Option<&uuid::Uuid>| {
            uuids
                .iter()
                .position(|candidate| candidate.as_ref() == uuid)
                .expect("extremum is a row")
        };
        let expected = MinMaxResult {
            min: array.execute_scalar(row_of(uuids.iter().flatten().min()), &mut ctx)?,
            max: array.execute_scalar(row_of(uuids.iter().flatten().max()), &mut ctx)?,
        };

        let result = min_max(&array, &mut ctx, NumericalAggregateOpts::default())?
            .expect("non-null UUIDs have extrema");
        assert_eq!(result, expected);
        assert!(!result.min.dtype().is_nullable() && !result.max.dtype().is_nullable());
        Ok(())
    }
}
