use std::fmt::Debug;

use serde::{Deserialize, Serialize};
use vortex_array::array::PrimitiveArray;
use vortex_array::compute::{
    scalar_at, search_sorted_usize, search_sorted_usize_many, SearchSortedSide,
};
use vortex_array::stats::StatsSet;
use vortex_array::variants::{BoolArrayTrait, PrimitiveArrayTrait};
use vortex_array::visitor::ArrayVisitor;
use vortex_array::vtable::{
    CanonicalVTable, ValidateVTable, ValidityVTable, VariantsVTable, VisitorVTable,
};
use vortex_array::{
    encoding_ids, impl_encoding, Array, Canonical, IntoArray, IntoArrayVariant, SerdeMetadata,
};
use vortex_buffer::Buffer;
use vortex_dtype::{DType, PType};
use vortex_error::{vortex_bail, VortexExpect as _, VortexResult};
use vortex_mask::Mask;

use crate::compress::{runend_decode_bools, runend_decode_primitive, runend_encode};

impl_encoding!(
    "vortex.runend",
    encoding_ids::RUN_END,
    RunEnd,
    SerdeMetadata<RunEndMetadata>
);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunEndMetadata {
    ends_ptype: PType,
    num_runs: usize,
    offset: usize,
}

impl RunEndArray {
    pub fn try_new(ends: Array, values: Array) -> VortexResult<Self> {
        let length = if ends.is_empty() {
            0
        } else {
            scalar_at(&ends, ends.len() - 1)?.as_ref().try_into()?
        };
        Self::with_offset_and_length(ends, values, 0, length)
    }

    pub(crate) fn with_offset_and_length(
        ends: Array,
        values: Array,
        offset: usize,
        length: usize,
    ) -> VortexResult<Self> {
        if !matches!(values.dtype(), &DType::Bool(_) | &DType::Primitive(_, _)) {
            vortex_bail!(
                "RunEnd array can only have Bool or Primitive values, {} given",
                values.dtype()
            );
        }

        if offset != 0 {
            let first_run_end: usize = scalar_at(&ends, 0)?.as_ref().try_into()?;
            if first_run_end <= offset {
                vortex_bail!("First run end {first_run_end} must be bigger than offset {offset}");
            }
        }

        if !ends.dtype().is_unsigned_int() || ends.dtype().is_nullable() {
            vortex_bail!(MismatchedTypes: "non-nullable unsigned int", ends.dtype());
        }
        if !ends.statistics().compute_is_strict_sorted().unwrap_or(true) {
            vortex_bail!("Ends array must be strictly sorted");
        }

        let dtype = values.dtype().clone();
        let metadata = RunEndMetadata {
            ends_ptype: PType::try_from(ends.dtype())?,
            num_runs: ends.len(),
            offset,
        };

        Self::try_from_parts(
            dtype,
            length,
            SerdeMetadata(metadata),
            None,
            Some(vec![ends, values].into()),
            StatsSet::default(),
        )
    }

    /// Convert the given logical index to an index into the `values` array
    pub fn find_physical_index(&self, index: usize) -> VortexResult<usize> {
        search_sorted_usize(&self.ends(), index + self.offset(), SearchSortedSide::Right)
            .map(|s| s.to_ends_index(self.ends().len()))
    }

    /// Convert a batch of logical indices into an index for the values. Expects indices to be adjusted by offset unlike
    /// [Self::find_physical_index]
    ///
    /// See: [find_physical_index][Self::find_physical_index].
    pub fn find_physical_indices(&self, indices: &[usize]) -> VortexResult<Buffer<u64>> {
        search_sorted_usize_many(&self.ends(), indices, SearchSortedSide::Right).map(|results| {
            results
                .iter()
                .map(|result| result.to_ends_index(self.ends().len()) as u64)
                .collect()
        })
    }

    /// Run the array through run-end encoding.
    pub fn encode(array: Array) -> VortexResult<Self> {
        if let Ok(parray) = PrimitiveArray::try_from(array) {
            let (ends, values) = runend_encode(&parray)?;
            Self::try_new(ends.into_array(), values)
        } else {
            vortex_bail!("REE can only encode primitive arrays")
        }
    }

    /// The offset that the `ends` is relative to.
    ///
    /// This is generally zero for a "new" array, and non-zero after a slicing operation.
    #[inline]
    pub fn offset(&self) -> usize {
        self.metadata().offset
    }

    /// The encoded "ends" of value runs.
    ///
    /// The `i`-th element indicates that there is a run of the same value, beginning
    /// at `ends[i]` (inclusive) and terminating at `ends[i+1]` (exclusive).
    #[inline]
    pub fn ends(&self) -> Array {
        self.as_ref()
            .child(
                0,
                &DType::from(self.metadata().ends_ptype),
                self.metadata().num_runs,
            )
            .vortex_expect("RunEndArray is missing its run ends")
    }

    /// The scalar values.
    ///
    /// The `i`-th element is the scalar value for the `i`-th repeated run. The run begins
    /// at `ends[i]` (inclusive) and terminates at `ends[i+1]` (exclusive).
    #[inline]
    pub fn values(&self) -> Array {
        self.as_ref()
            .child(1, self.dtype(), self.metadata().num_runs)
            .vortex_expect("RunEndArray is missing its values")
    }
}

impl ValidateVTable<RunEndArray> for RunEndEncoding {}

impl VariantsVTable<RunEndArray> for RunEndEncoding {
    fn as_bool_array<'a>(&self, array: &'a RunEndArray) -> Option<&'a dyn BoolArrayTrait> {
        Some(array)
    }

    fn as_primitive_array<'a>(
        &self,
        array: &'a RunEndArray,
    ) -> Option<&'a dyn PrimitiveArrayTrait> {
        Some(array)
    }
}

impl PrimitiveArrayTrait for RunEndArray {}

impl BoolArrayTrait for RunEndArray {}

impl ValidityVTable<RunEndArray> for RunEndEncoding {
    fn is_valid(&self, array: &RunEndArray, index: usize) -> VortexResult<bool> {
        let physical_idx = array
            .find_physical_index(index)
            .vortex_expect("Invalid index");
        array.values().is_valid(physical_idx)
    }

    fn all_valid(&self, array: &RunEndArray) -> VortexResult<bool> {
        array.values().all_valid()
    }

    fn validity_mask(&self, array: &RunEndArray) -> VortexResult<Mask> {
        Ok(match array.values().validity_mask()? {
            Mask::AllTrue(_) => Mask::AllTrue(array.len()),
            Mask::AllFalse(_) => Mask::AllFalse(array.len()),
            Mask::Values(values) => {
                let ree_validity = RunEndArray::with_offset_and_length(
                    array.ends(),
                    values.into_array(),
                    array.offset(),
                    array.len(),
                )
                .vortex_expect("invalid array")
                .into_array();
                Mask::from_buffer(ree_validity.into_bool()?.boolean_buffer())
            }
        })
    }
}

impl CanonicalVTable<RunEndArray> for RunEndEncoding {
    fn into_canonical(&self, array: RunEndArray) -> VortexResult<Canonical> {
        let pends = array.ends().into_primitive()?;
        match array.dtype() {
            DType::Bool(_) => {
                let bools = array.values().into_bool()?;
                runend_decode_bools(pends, bools, array.offset(), array.len()).map(Canonical::Bool)
            }
            DType::Primitive(..) => {
                let pvalues = array.values().into_primitive()?;
                runend_decode_primitive(pends, pvalues, array.offset(), array.len())
                    .map(Canonical::Primitive)
            }
            _ => vortex_bail!("Only Primitive and Bool values are supported"),
        }
    }
}

impl VisitorVTable<RunEndArray> for RunEndEncoding {
    fn accept(&self, array: &RunEndArray, visitor: &mut dyn ArrayVisitor) -> VortexResult<()> {
        visitor.visit_child("ends", &array.ends())?;
        visitor.visit_child("values", &array.values())
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::compute::scalar_at;
    use vortex_array::test_harness::check_metadata;
    use vortex_array::{IntoArray, SerdeMetadata};
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, Nullability, PType};

    use crate::{RunEndArray, RunEndMetadata};

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_runend_metadata() {
        check_metadata(
            "runend.metadata",
            SerdeMetadata(RunEndMetadata {
                offset: usize::MAX,
                ends_ptype: PType::U64,
                num_runs: usize::MAX,
            }),
        );
    }

    #[test]
    fn test_runend_constructor() {
        let arr = RunEndArray::try_new(
            buffer![2u32, 5, 10].into_array(),
            buffer![1i32, 2, 3].into_array(),
        )
        .unwrap();
        assert_eq!(arr.len(), 10);
        assert_eq!(
            arr.dtype(),
            &DType::Primitive(PType::I32, Nullability::NonNullable)
        );

        // 0, 1 => 1
        // 2, 3, 4 => 2
        // 5, 6, 7, 8, 9 => 3
        assert_eq!(scalar_at(arr.as_ref(), 0).unwrap(), 1.into());
        assert_eq!(scalar_at(arr.as_ref(), 2).unwrap(), 2.into());
        assert_eq!(scalar_at(arr.as_ref(), 5).unwrap(), 3.into());
        assert_eq!(scalar_at(arr.as_ref(), 9).unwrap(), 3.into());
    }
}
