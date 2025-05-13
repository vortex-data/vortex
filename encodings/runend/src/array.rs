use std::fmt::Debug;

use vortex_array::arrays::PrimitiveVTable;
use vortex_array::search_sorted::{SearchSorted, SearchSortedSide};
use vortex_array::stats::{ArrayStats, StatsSetRef};
use vortex_array::vtable::{ArrayVTable, CanonicalVTable, NotSupported, VTable, ValidityVTable};
use vortex_array::{
    Array, ArrayExt, ArrayRef, Canonical, EncodingId, EncodingRef, IntoArray, ToCanonical, vtable,
};
use vortex_dtype::DType;
use vortex_error::{VortexExpect as _, VortexResult, vortex_bail};
use vortex_mask::Mask;
use vortex_scalar::PValue;

use crate::compress::{runend_decode_bools, runend_decode_primitive, runend_encode};

vtable!(RunEnd);

impl VTable for RunEndVTable {
    type Array = RunEndArray;
    type Encoding = RunEndEncoding;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = Self;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = Self;
    type SerdeVTable = Self;

    fn id(_encoding: &Self::Encoding) -> EncodingId {
        EncodingId::new_ref("vortex.runend")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        EncodingRef::new_ref(RunEndEncoding.as_ref())
    }
}

#[derive(Clone, Debug)]
pub struct RunEndArray {
    ends: ArrayRef,
    values: ArrayRef,
    offset: usize,
    length: usize,
    stats_set: ArrayStats,
}

#[derive(Clone, Debug)]
pub struct RunEndEncoding;

impl RunEndArray {
    pub fn try_new(ends: ArrayRef, values: ArrayRef) -> VortexResult<Self> {
        let length = if ends.is_empty() {
            0
        } else {
            ends.scalar_at(ends.len() - 1)?.as_ref().try_into()?
        };
        Self::with_offset_and_length(ends, values, 0, length)
    }

    pub(crate) fn with_offset_and_length(
        ends: ArrayRef,
        values: ArrayRef,
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
            let first_run_end: usize = ends.scalar_at(0)?.as_ref().try_into()?;
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

        Ok(Self {
            ends,
            values,
            offset,
            length,
            stats_set: Default::default(),
        })
    }

    /// Convert the given logical index to an index into the `values` array
    pub fn find_physical_index(&self, index: usize) -> VortexResult<usize> {
        Ok(self
            .ends()
            .as_primitive_typed()
            .search_sorted(
                &PValue::from(index + self.offset()),
                SearchSortedSide::Right,
            )
            .to_ends_index(self.ends().len()))
    }

    /// Run the array through run-end encoding.
    pub fn encode(array: ArrayRef) -> VortexResult<Self> {
        if let Some(parray) = array.as_opt::<PrimitiveVTable>() {
            let (ends, values) = runend_encode(parray)?;
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
        self.offset
    }

    /// The encoded "ends" of value runs.
    ///
    /// The `i`-th element indicates that there is a run of the same value, beginning
    /// at `ends[i]` (inclusive) and terminating at `ends[i+1]` (exclusive).
    #[inline]
    pub fn ends(&self) -> &ArrayRef {
        &self.ends
    }

    /// The scalar values.
    ///
    /// The `i`-th element is the scalar value for the `i`-th repeated run. The run begins
    /// at `ends[i]` (inclusive) and terminates at `ends[i+1]` (exclusive).
    #[inline]
    pub fn values(&self) -> &ArrayRef {
        &self.values
    }
}

impl ArrayVTable<RunEndVTable> for RunEndVTable {
    fn len(array: &RunEndArray) -> usize {
        array.length
    }

    fn dtype(array: &RunEndArray) -> &DType {
        array.values.dtype()
    }

    fn stats(array: &RunEndArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }
}

impl ValidityVTable<RunEndVTable> for RunEndVTable {
    fn is_valid(array: &RunEndArray, index: usize) -> VortexResult<bool> {
        let physical_idx = array
            .find_physical_index(index)
            .vortex_expect("Invalid index");
        array.values().is_valid(physical_idx)
    }

    fn all_valid(array: &RunEndArray) -> VortexResult<bool> {
        array.values().all_valid()
    }

    fn all_invalid(array: &RunEndArray) -> VortexResult<bool> {
        array.values().all_invalid()
    }

    fn validity_mask(array: &RunEndArray) -> VortexResult<Mask> {
        Ok(match array.values().validity_mask()? {
            Mask::AllTrue(_) => Mask::AllTrue(array.len()),
            Mask::AllFalse(_) => Mask::AllFalse(array.len()),
            Mask::Values(values) => {
                let ree_validity = RunEndArray::with_offset_and_length(
                    array.ends().clone(),
                    values.into_array(),
                    array.offset(),
                    array.len(),
                )
                .vortex_expect("invalid array")
                .into_array();
                Mask::from_buffer(ree_validity.to_bool()?.boolean_buffer().clone())
            }
        })
    }
}

impl CanonicalVTable<RunEndVTable> for RunEndVTable {
    fn canonicalize(array: &RunEndArray) -> VortexResult<Canonical> {
        let pends = array.ends().to_primitive()?;
        match array.dtype() {
            DType::Bool(_) => {
                let bools = array.values().to_bool()?;
                runend_decode_bools(pends, bools, array.offset(), array.len()).map(Canonical::Bool)
            }
            DType::Primitive(..) => {
                let pvalues = array.values().to_primitive()?;
                runend_decode_primitive(pends, pvalues, array.offset(), array.len())
                    .map(Canonical::Primitive)
            }
            _ => vortex_bail!("Only Primitive and Bool values are supported"),
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, Nullability, PType};

    use crate::RunEndArray;

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
        assert_eq!(arr.scalar_at(0).unwrap(), 1.into());
        assert_eq!(arr.scalar_at(2).unwrap(), 2.into());
        assert_eq!(arr.scalar_at(5).unwrap(), 3.into());
        assert_eq!(arr.scalar_at(9).unwrap(), 3.into());
    }
}
