// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::hash::Hash;

use vortex_array::Array;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::DeserializeMetadata;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::Precision;
use vortex_array::ProstMetadata;
use vortex_array::SerializeMetadata;
use vortex_array::ToCanonical;
use vortex_array::arrays::PrimitiveVTable;
use vortex_array::buffer::BufferHandle;
use vortex_array::search_sorted::SearchSorted;
use vortex_array::search_sorted::SearchSortedSide;
use vortex_array::serde::ArrayChildren;
use vortex_array::stats::ArrayStats;
use vortex_array::stats::StatsSetRef;
use vortex_array::validity::Validity;
use vortex_array::vtable;
use vortex_array::vtable::ArrayId;
use vortex_array::vtable::ArrayVTable;
use vortex_array::vtable::ArrayVTableExt;
use vortex_array::vtable::BaseArrayVTable;
use vortex_array::vtable::NotSupported;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityVTable;
use vortex_dtype::DType;
use vortex_dtype::Nullability;
use vortex_dtype::PType;
use vortex_error::VortexExpect as _;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_mask::Mask;
use vortex_scalar::PValue;

use crate::compress::runend_decode_bools;
use crate::compress::runend_decode_primitive;
use crate::compress::runend_encode;
use crate::kernel::PARENT_KERNELS;
use crate::rules::RULES;

vtable!(RunEnd);

#[derive(Clone, prost::Message)]
pub struct RunEndMetadata {
    #[prost(enumeration = "PType", tag = "1")]
    pub ends_ptype: i32,
    #[prost(uint64, tag = "2")]
    pub num_runs: u64,
    #[prost(uint64, tag = "3")]
    pub offset: u64,
}

impl VTable for RunEndVTable {
    type Array = RunEndArray;

    type Metadata = ProstMetadata<RunEndMetadata>;

    type ArrayVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = Self;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;

    fn id(&self) -> ArrayId {
        ArrayId::new_ref("vortex.runend")
    }

    fn encoding(_array: &Self::Array) -> ArrayVTable {
        RunEndVTable.as_vtable()
    }

    fn metadata(array: &RunEndArray) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(RunEndMetadata {
            ends_ptype: PType::try_from(array.ends().dtype()).vortex_expect("Must be a valid PType")
                as i32,
            num_runs: array.ends().len() as u64,
            offset: array.offset() as u64,
        }))
    }

    fn serialize(metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(metadata.serialize()))
    }

    fn deserialize(buffer: &[u8]) -> VortexResult<Self::Metadata> {
        let inner = <ProstMetadata<RunEndMetadata> as DeserializeMetadata>::deserialize(buffer)?;
        Ok(ProstMetadata(inner))
    }

    fn build(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<RunEndArray> {
        let ends_dtype = DType::Primitive(metadata.ends_ptype(), Nullability::NonNullable);
        let runs = usize::try_from(metadata.num_runs).vortex_expect("Must be a valid usize");
        let ends = children.get(0, &ends_dtype, runs)?;

        let values = children.get(1, dtype, runs)?;

        RunEndArray::try_new_offset_length(
            ends,
            values,
            usize::try_from(metadata.offset).vortex_expect("Offset must be a valid usize"),
            len,
        )
    }

    fn with_children(array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        vortex_ensure!(
            children.len() == 2,
            "RunEndArray expects 2 children, got {}",
            children.len()
        );

        let mut children_iter = children.into_iter();
        array.ends = children_iter.next().vortex_expect("ends child");
        array.values = children_iter.next().vortex_expect("values child");

        Ok(())
    }

    fn reduce_parent(
        array: &Self::Array,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        RULES.evaluate(array, parent, child_idx)
    }

    fn execute_parent(
        array: &Self::Array,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Canonical>> {
        PARENT_KERNELS.execute(array, parent, child_idx, ctx)
    }

    fn execute(array: &Self::Array, _ctx: &mut ExecutionCtx) -> VortexResult<Canonical> {
        run_end_canonicalize(array)
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

#[derive(Debug)]
pub struct RunEndVTable;

impl RunEndArray {
    fn validate(
        ends: &dyn Array,
        values: &dyn Array,
        offset: usize,
        length: usize,
    ) -> VortexResult<()> {
        // DType validation
        vortex_ensure!(
            ends.dtype().is_unsigned_int(),
            "run ends must be unsigned integers, was {}",
            ends.dtype(),
        );
        vortex_ensure!(
            values.dtype().is_primitive() || values.dtype().is_boolean(),
            "RunEnd array can only have Bool or Primitive values, {} given",
            values.dtype()
        );

        vortex_ensure!(
            ends.len() == values.len(),
            "run ends len != run values len, {} != {}",
            ends.len(),
            values.len()
        );

        // Handle empty run-ends
        if ends.is_empty() {
            vortex_ensure!(
                offset == 0,
                "non-zero offset provided for empty RunEndArray"
            );
            return Ok(());
        }

        // Avoid building a non-empty array with zero logical length.
        if length == 0 {
            vortex_ensure!(
                ends.is_empty(),
                "run ends must be empty when length is zero"
            );
            return Ok(());
        }

        // Validate the offset and length are valid for the given ends and values
        if offset != 0 && length != 0 {
            let first_run_end: usize = ends.scalar_at(0).as_ref().try_into()?;
            if first_run_end <= offset {
                vortex_bail!("First run end {first_run_end} must be bigger than offset {offset}");
            }
        }

        let last_run_end: usize = ends.scalar_at(ends.len() - 1).as_ref().try_into()?;
        let min_required_end = offset + length;
        if last_run_end < min_required_end {
            vortex_bail!("Last run end {last_run_end} must be >= offset+length {min_required_end}");
        }

        Ok(())
    }
}

impl RunEndArray {
    /// Build a new `RunEndArray` from an array of run `ends` and an array of `values`.
    ///
    /// Panics if any of the validation conditions described in [`RunEndArray::try_new`] is
    /// not satisfied.
    ///
    /// # Examples
    ///
    /// ```
    /// # use vortex_array::arrays::BoolArray;
    /// # use vortex_array::IntoArray;
    /// # use vortex_buffer::buffer;
    /// # use vortex_runend::RunEndArray;
    /// let ends = buffer![2u8, 3u8].into_array();
    /// let values = BoolArray::from_iter([false, true]).into_array();
    /// let run_end = RunEndArray::new(ends, values);
    ///
    /// // Array encodes
    /// assert_eq!(run_end.scalar_at(0), false.into());
    /// assert_eq!(run_end.scalar_at(1), false.into());
    /// assert_eq!(run_end.scalar_at(2), true.into());
    /// ```
    pub fn new(ends: ArrayRef, values: ArrayRef) -> Self {
        Self::try_new(ends, values).vortex_expect("RunEndArray new")
    }

    /// Build a new `RunEndArray` from components.
    ///
    /// # Validation
    ///
    /// The `ends` must be non-nullable unsigned integers. The values may be `Bool` or `Primitive`
    /// types.
    ///
    /// # Examples
    ///
    /// ```
    /// # use vortex_array::arrays::{BoolArray, VarBinViewArray};
    /// # use vortex_array::IntoArray;
    /// # use vortex_buffer::buffer;
    /// # use vortex_runend::RunEndArray;
    ///
    /// // Error to provide incorrectly-typed values!
    /// let result = RunEndArray::try_new(
    ///     buffer![1u8, 2u8].into_array(),
    ///     VarBinViewArray::from_iter_str(["bad", "dtype"]).into_array(),
    /// );
    /// assert!(result.is_err());
    ///
    /// // This array is happy
    /// let result = RunEndArray::try_new(
    ///     buffer![1u8, 2u8].into_array(),
    ///     BoolArray::from_iter([false, true]).into_array(),
    /// );
    ///
    /// assert!(result.is_ok());
    /// ```
    pub fn try_new(ends: ArrayRef, values: ArrayRef) -> VortexResult<Self> {
        let length: usize = if ends.is_empty() {
            0
        } else {
            ends.scalar_at(ends.len() - 1).as_ref().try_into()?
        };

        Self::try_new_offset_length(ends, values, 0, length)
    }

    /// Construct a new sliced `RunEndArray` with the provided offset and length.
    ///
    /// This performs all the same validation as [`RunEndArray::try_new`].
    pub fn try_new_offset_length(
        ends: ArrayRef,
        values: ArrayRef,
        offset: usize,
        length: usize,
    ) -> VortexResult<Self> {
        Self::validate(&ends, &values, offset, length)?;

        Ok(Self {
            ends,
            values,
            offset,
            length,
            stats_set: Default::default(),
        })
    }

    /// Build a new `RunEndArray` without validation.
    ///
    /// # Safety
    ///
    /// The caller must ensure that all the validation performed in [`RunEndArray::try_new`] is
    /// satisfied before calling this function.
    ///
    /// See [`RunEndArray::try_new`] for the preconditions needed to build a new array.
    pub unsafe fn new_unchecked(
        ends: ArrayRef,
        values: ArrayRef,
        offset: usize,
        length: usize,
    ) -> Self {
        Self {
            ends,
            values,
            offset,
            length,
            stats_set: Default::default(),
        }
    }

    /// Convert the given logical index to an index into the `values` array
    pub fn find_physical_index(&self, index: usize) -> usize {
        self.ends()
            .as_primitive_typed()
            .search_sorted(
                &PValue::from(index + self.offset()),
                SearchSortedSide::Right,
            )
            .to_ends_index(self.ends().len())
    }

    /// Run the array through run-end encoding.
    pub fn encode(array: ArrayRef) -> VortexResult<Self> {
        if let Some(parray) = array.as_opt::<PrimitiveVTable>() {
            let (ends, values) = runend_encode(parray);
            // SAFETY: runend_encode handles this
            unsafe {
                Ok(Self::new_unchecked(
                    ends.into_array(),
                    values,
                    0,
                    array.len(),
                ))
            }
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

impl BaseArrayVTable<RunEndVTable> for RunEndVTable {
    fn len(array: &RunEndArray) -> usize {
        array.length
    }

    fn dtype(array: &RunEndArray) -> &DType {
        array.values.dtype()
    }

    fn stats(array: &RunEndArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(array: &RunEndArray, state: &mut H, precision: Precision) {
        array.ends.array_hash(state, precision);
        array.values.array_hash(state, precision);
        array.offset.hash(state);
        array.length.hash(state);
    }

    fn array_eq(array: &RunEndArray, other: &RunEndArray, precision: Precision) -> bool {
        array.ends.array_eq(&other.ends, precision)
            && array.values.array_eq(&other.values, precision)
            && array.offset == other.offset
            && array.length == other.length
    }
}

impl ValidityVTable<RunEndVTable> for RunEndVTable {
    fn validity(array: &RunEndArray) -> VortexResult<Validity> {
        Ok(match array.values().validity()? {
            Validity::NonNullable | Validity::AllValid => Validity::AllValid,
            Validity::AllInvalid => Validity::AllInvalid,
            Validity::Array(values_validity) => Validity::Array(unsafe {
                RunEndArray::new_unchecked(
                    array.ends().clone(),
                    values_validity,
                    array.offset(),
                    array.len(),
                )
                .into_array()
            }),
        })
    }

    fn validity_mask(array: &RunEndArray) -> Mask {
        match array.values().validity_mask() {
            Mask::AllTrue(_) => Mask::AllTrue(array.len()),
            Mask::AllFalse(_) => Mask::AllFalse(array.len()),
            Mask::Values(values) => {
                // SAFETY: we preserve ends from an existing validated RunEndArray.
                //  Validity is checked on construction to have the correct len.
                let ree_validity = unsafe {
                    RunEndArray::new_unchecked(
                        array.ends().clone(),
                        values.into_array(),
                        array.offset(),
                        array.len(),
                    )
                    .into_array()
                };
                Mask::from_buffer(ree_validity.to_bool().bit_buffer().clone())
            }
        }
    }
}

pub(super) fn run_end_canonicalize(array: &RunEndArray) -> VortexResult<Canonical> {
    let pends = array.ends().to_primitive();
    Ok(match array.dtype() {
        DType::Bool(_) => {
            let bools = array.values().to_bool();
            Canonical::Bool(runend_decode_bools(
                pends,
                bools,
                array.offset(),
                array.len(),
            ))
        }
        DType::Primitive(..) => {
            let pvalues = array.values().to_primitive();
            Canonical::Primitive(runend_decode_primitive(
                pends,
                pvalues,
                array.offset(),
                array.len(),
            ))
        }
        _ => vortex_panic!("Only Primitive and Bool values are supported"),
    })
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::assert_arrays_eq;
    use vortex_buffer::buffer;
    use vortex_dtype::DType;
    use vortex_dtype::Nullability;
    use vortex_dtype::PType;

    use crate::RunEndArray;

    #[test]
    fn test_runend_constructor() {
        let arr = RunEndArray::new(
            buffer![2u32, 5, 10].into_array(),
            buffer![1i32, 2, 3].into_array(),
        );
        assert_eq!(arr.len(), 10);
        assert_eq!(
            arr.dtype(),
            &DType::Primitive(PType::I32, Nullability::NonNullable)
        );

        // 0, 1 => 1
        // 2, 3, 4 => 2
        // 5, 6, 7, 8, 9 => 3
        let expected = buffer![1, 1, 2, 2, 2, 3, 3, 3, 3, 3].into_array();
        assert_arrays_eq!(arr.to_array(), expected);
    }
}
