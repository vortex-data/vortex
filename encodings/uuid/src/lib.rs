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
//! [`initialize`] registers the extension dtype, the Arrow import/export plugins that map it to
//! Arrow's `arrow.uuid` extension type over `FixedSizeBinary[16]` storage, and `is_sorted` and
//! `min_max` over its storage, since big-endian bytes order like the UUIDs they encode.
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
/// for Arrow's canonical `arrow.uuid` extension type. UUIDs are stored big-endian, so they compare
/// like their storage bytes, and `is_sorted` and `min_max` are pushed down into the storage.
pub fn initialize(session: &VortexSession) {
    session.dtypes().register(Uuid);
    vortex_aggregates::register_storage_ordered(session, Uuid);
    session.arrow().register_exporter(Arc::new(Uuid));
    session.arrow().register_importer(Arc::new(Uuid));
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_aggregates::is_sorted::IsSorted;
    use vortex_aggregates::is_sorted::IsSortedOptions;
    use vortex_aggregates::min_max::MinMax;
    use vortex_aggregates::min_max::MinMaxResult;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::Accumulator;
    use vortex_array::aggregate_fn::AggregateFnVTable;
    use vortex_array::aggregate_fn::DynAccumulator;
    use vortex_array::aggregate_fn::ExtensionStoragePushdown;
    use vortex_array::aggregate_fn::NumericalAggregateOpts;
    use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
    use vortex_array::arrays::ExtensionArray;
    use vortex_array::arrays::FixedSizeListArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::extension::ExtDType;
    use vortex_array::scalar::Scalar;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::*;

    fn uuid_session() -> VortexSession {
        let session = VortexSession::empty();
        initialize(&session);
        session
    }

    /// A nullable UUID column, with each UUID given by its leading byte.
    fn uuid_column(leading_bytes: &[Option<u8>]) -> VortexResult<ArrayRef> {
        let bytes = leading_bytes
            .iter()
            .flat_map(|byte| {
                let mut uuid = [0u8; 16];
                uuid[0] = byte.unwrap_or_default();
                uuid
            })
            .collect::<Buffer<u8>>();
        let storage = FixedSizeListArray::new(
            PrimitiveArray::new(bytes, Validity::NonNullable).into_array(),
            16,
            leading_bytes.iter().map(Option::is_some).collect(),
            leading_bytes.len(),
        )
        .into_array();
        let ext_dtype =
            ExtDType::try_with_vtable(Uuid, UuidMetadata::default(), storage.dtype().clone())?;
        Ok(ExtensionArray::new(ext_dtype.erased(), storage).into_array())
    }

    #[test]
    fn initialize_registers_order_aggregates() {
        let session = uuid_session();
        for id in [
            ExtensionStoragePushdown::new(IsSorted, Uuid).id(),
            ExtensionStoragePushdown::new(MinMax, Uuid).id(),
        ] {
            assert!(
                session.aggregate_fns().find_plugin(&id).is_some(),
                "{id} is not registered"
            );
        }
    }

    #[rstest]
    // Nulls sort first.
    #[case::sorted(&[None, Some(1), Some(2), Some(2)], false, true)]
    #[case::unsorted(&[Some(2), Some(1), None, Some(3)], false, false)]
    #[case::strict_with_duplicates(&[None, Some(1), Some(2), Some(2)], true, false)]
    #[case::strict(&[None, Some(1), Some(2), Some(0xff)], true, true)]
    fn is_sorted_orders_by_bytes(
        #[case] leading_bytes: &[Option<u8>],
        #[case] strict: bool,
        #[case] expected: bool,
    ) -> VortexResult<()> {
        let session = uuid_session();
        let column = uuid_column(leading_bytes)?;
        let mut acc = Accumulator::try_new(
            ExtensionStoragePushdown::new(IsSorted, Uuid),
            IsSortedOptions { strict },
            column.dtype().clone(),
        )?;
        acc.accumulate(&column, &mut session.create_execution_ctx())?;
        assert_eq!(bool::try_from(&acc.finish()?)?, expected);
        Ok(())
    }

    #[test]
    fn min_max_orders_by_bytes() -> VortexResult<()> {
        let session = uuid_session();
        let column = uuid_column(&[Some(7), None, Some(0xff), Some(3)])?;
        let mut acc = Accumulator::try_new(
            ExtensionStoragePushdown::new(MinMax, Uuid),
            NumericalAggregateOpts::default(),
            column.dtype().clone(),
        )?;
        acc.accumulate(&column, &mut session.create_execution_ctx())?;

        let uuid = |leading_byte: u8| {
            let mut bytes = [0u8; 16];
            bytes[0] = leading_byte;
            Scalar::fixed_size_list(
                DType::Primitive(PType::U8, Nullability::NonNullable),
                bytes
                    .into_iter()
                    .map(|byte| Scalar::primitive(byte, Nullability::NonNullable))
                    .collect(),
                Nullability::NonNullable,
            )
        };
        assert_eq!(
            MinMaxResult::from_scalar(&acc.finish()?)?,
            Some(MinMaxResult {
                min: uuid(3),
                max: uuid(0xff),
            })
        );
        Ok(())
    }
}
