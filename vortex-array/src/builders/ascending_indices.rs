// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Checked construction of strictly increasing index arrays.
//!
//! The builder owns its values and checks each append before publishing sortedness on the finished
//! Primitive array. Encoding producers can preserve their construction facts without a sortedness
//! scan or accepting caller-supplied aggregate claims.

use vortex_buffer::BufferMut;
use vortex_error::VortexResult;

use crate::ExecutionCtx;
use crate::aggregate_fn::fns::is_sorted::IS_STRICT_SORTED;
use crate::arrays::PrimitiveArray;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::expr::stats::Precision;
use crate::validity::Validity;

/// A builder that checks strictly increasing u64 indices as they are appended.
///
/// The buffer stays private so every value is covered by the append check. Finishing publishes
/// exact strict sortedness on the non-nullable array, including empty and singleton arrays.
#[derive(Default)]
pub struct AscendingIndexBuilder {
    values: BufferMut<u64>,
}

impl AscendingIndexBuilder {
    /// Create an empty builder without preallocating a buffer.
    pub fn new() -> Self {
        Self {
            values: BufferMut::empty(),
        }
    }

    /// Append an index greater than every previously appended index.
    ///
    /// # Panics
    /// Panics before appending if `index` is not strictly greater than the previous index.
    pub fn push(&mut self, index: u64) {
        if let Some(previous) = self.values.last() {
            assert!(
                index > *previous,
                "Ascending indices require a value greater than {previous}, got {index}"
            );
        }

        self.values.push(index);
    }

    /// Consume the builder into a non-nullable u64 array with exact strict sortedness.
    pub fn finish(self) -> PrimitiveArray {
        let array = PrimitiveArray::new(self.values.freeze(), Validity::NonNullable);
        array
            .aggregations()
            .insert_result(IS_STRICT_SORTED.clone(), Precision::Exact(true.into()));

        array
    }

    /// Consume the builder into the smallest integer array that holds its indices.
    ///
    /// Narrowing preserves the checked values. Publish strict sortedness on the narrowed output
    /// because aggregate inheritance requires an unchanged dtype.
    pub fn finish_narrow(self, ctx: &mut ExecutionCtx) -> VortexResult<PrimitiveArray> {
        let array = PrimitiveArray::new(self.values.freeze(), Validity::NonNullable).narrow(ctx)?;
        array
            .aggregations()
            .insert_result(IS_STRICT_SORTED.clone(), Precision::Exact(true.into()));

        Ok(array)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_error::VortexResult;

    use super::AscendingIndexBuilder;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::fns::is_sorted::IS_STRICT_SORTED;
    use crate::array_session;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::primitive::PrimitiveArrayExt;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::stats::Precision;

    #[rstest]
    #[case::empty(vec![], PType::U8)]
    #[case::singleton(vec![0], PType::U8)]
    #[case::u8(vec![1, 255], PType::U8)]
    #[case::u16(vec![1, 256], PType::U16)]
    #[case::u32(vec![1, 65536], PType::U32)]
    #[case::u64(vec![1, u64::MAX], PType::U64)]
    fn finish_retains_checked_values_and_sortedness(
        #[case] indices: Vec<u64>,
        #[case] narrowed_ptype: PType,
        #[values(false, true)] narrow: bool,
    ) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let mut builder = AscendingIndexBuilder::new();
        for &index in &indices {
            builder.push(index);
        }

        let array = if narrow {
            builder.finish_narrow(&mut ctx)?
        } else {
            builder.finish()
        };
        let expected_ptype = if narrow { narrowed_ptype } else { PType::U64 };
        assert_eq!(array.ptype(), expected_ptype);
        assert_eq!(
            array
                .aggregations()
                .get_result_as::<bool>(&IS_STRICT_SORTED)?,
            Precision::Exact(true),
        );
        let widened = array
            .as_ref()
            .cast(DType::Primitive(PType::U64, Nullability::NonNullable))?;
        assert_arrays_eq!(
            widened,
            PrimitiveArray::from_iter(indices).into_array(),
            &mut ctx
        );

        Ok(())
    }

    #[rstest]
    #[case::duplicate(1, 1)]
    #[case::descending(2, 1)]
    #[should_panic(expected = "Ascending indices require a value greater than")]
    fn rejects_nonascending_indices(#[case] first: u64, #[case] second: u64) {
        let mut builder = AscendingIndexBuilder::new();
        builder.push(first);
        builder.push(second);
    }
}
