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
//! Arrow's `arrow.uuid` extension type over `FixedSizeBinary[16]` storage. The UUID dtype claims
//! that its storage order matches the order of its values.
//!
//! [RFC 4122]: https://www.rfc-editor.org/rfc/rfc4122
//! [canonical UUID extension]: https://arrow.apache.org/docs/format/CanonicalExtensions.html#uuid

mod arrow;
mod metadata;
mod vtable;

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
    session.arrow().register_extension(Uuid);
}

#[cfg(test)]
mod arrow_properties;

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::Accumulator;
    use vortex_array::aggregate_fn::DynAccumulator;
    use vortex_array::aggregate_fn::NumericalAggregateOpts;
    use vortex_array::aggregate_fn::fns::is_sorted::IsSorted;
    use vortex_array::aggregate_fn::fns::is_sorted::IsSortedOptions;
    use vortex_array::aggregate_fn::fns::min::Min;
    use vortex_array::aggregate_fn::fns::min_max::MinMax;
    use vortex_array::aggregate_fn::fns::min_max::MinMaxResult;
    use vortex_array::arrays::ExtensionArray;
    use vortex_array::arrays::FixedSizeListArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::extension::ExtensionArrayExt;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::compute::conformance::extension::test_ext_vtable_conformance;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::extension::ExtDType;
    use vortex_array::scalar::Scalar;
    use vortex_array::test_harness::ExpectedOutcome;
    use vortex_array::test_harness::assert_execution_paths;
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

    #[test]
    fn cast_never_reinterprets_storage() -> VortexResult<()> {
        let input = uuid_column(&[Some(1), None, Some(2)])?;
        let target = input.dtype().as_extension().storage_dtype().clone();

        assert_execution_paths(
            &[input],
            |inputs| inputs[0].cast(target.clone()),
            ExpectedOutcome::Error,
            &mut uuid_session().create_execution_ctx(),
        );
        Ok(())
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
        let ext_dtype = ExtDType::try_with_vtable(
            Uuid,
            UuidMetadata::default(),
            storage.dtype().nullability(),
        )?;
        Ok(ExtensionArray::new(ext_dtype.erased(), storage).into_array())
    }

    #[test]
    fn uuid_claim_conformance() -> VortexResult<()> {
        let mut ctx = uuid_session().create_execution_ctx();
        let column = uuid_column(&[Some(0xff), None, Some(0x01), Some(0xff)])?
            .execute::<ExtensionArray>(&mut ctx)?;
        let dtype = column.ext_dtype().clone().downcast::<Uuid>();

        test_ext_vtable_conformance(
            &dtype,
            column.storage_array(),
            Some(&|lhs, rhs| lhs.as_bytes().cmp(rhs.as_bytes())),
            &mut ctx,
        );
        Ok(())
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
        let mut acc =
            Accumulator::try_new(IsSorted, IsSortedOptions { strict }, column.dtype().clone())?;
        acc.accumulate(&column, &mut session.create_execution_ctx())?;
        assert_eq!(bool::try_from(&acc.finish()?)?, expected);
        Ok(())
    }

    #[test]
    fn min_and_min_max_preserve_logical_uuid_values() -> VortexResult<()> {
        let mut ctx = uuid_session().create_execution_ctx();
        let column = uuid_column(&[Some(7), None, Some(0xff), Some(3)])?;
        let mut acc = Accumulator::try_new(
            MinMax,
            NumericalAggregateOpts::default(),
            column.dtype().clone(),
        )?;
        acc.accumulate(&column, &mut ctx)?;

        let uuid = |leading_byte: u8| {
            let mut bytes = [0u8; 16];
            bytes[0] = leading_byte;
            Scalar::extension_ref(
                column.dtype().as_nonnullable().as_extension().clone(),
                Scalar::fixed_size_list(
                    DType::Primitive(PType::U8, Nullability::NonNullable),
                    bytes
                        .into_iter()
                        .map(|byte| Scalar::primitive(byte, Nullability::NonNullable))
                        .collect(),
                    Nullability::NonNullable,
                ),
            )
        };
        let mut min = Accumulator::try_new(
            Min,
            NumericalAggregateOpts::default(),
            column.dtype().clone(),
        )?;
        min.accumulate(&column, &mut ctx)?;
        assert_eq!(min.finish()?, uuid(3).into_nullable());
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
