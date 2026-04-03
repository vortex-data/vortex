// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::hash::Hash;

use vortex_array::Array;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::DeserializeMetadata;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::Precision;
use vortex_array::ProstMetadata;
use vortex_array::SerializeMetadata;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::scalar::PValue;
use vortex_array::search_sorted::SearchSorted;
use vortex_array::search_sorted::SearchSortedSide;
use vortex_array::serde::ArrayChildren;
use vortex_array::validity::Validity;
use vortex_array::vtable;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityVTable;
use vortex_error::VortexExpect as _;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::compress::runend_decode_primitive;
use crate::compress::runend_decode_varbinview;
use crate::compress::runend_encode;
use crate::decompress_bool::runend_decode_bools;
use crate::kernel::PARENT_KERNELS;
use crate::rules::RULES;

vtable!(RunEnd, RunEnd, RunEndData);

#[derive(Clone, prost::Message)]
pub struct RunEndMetadata {
    #[prost(enumeration = "PType", tag = "1")]
    pub ends_ptype: i32,
    #[prost(uint64, tag = "2")]
    pub num_runs: u64,
    #[prost(uint64, tag = "3")]
    pub offset: u64,
}

impl VTable for RunEnd {
    type ArrayData = RunEndData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn validate(&self, data: &Self::ArrayData, dtype: &DType, len: usize) -> VortexResult<()> {
        RunEndData::validate(data.ends(), data.values(), data.offset, len)?;
        vortex_ensure!(
            data.values().dtype() == dtype,
            "expected dtype {}, got {}",
            dtype,
            data.values().dtype()
        );
        Ok(())
    }

    fn array_hash<H: std::hash::Hasher>(array: &RunEndData, state: &mut H, precision: Precision) {
        array.ends().array_hash(state, precision);
        array.values().array_hash(state, precision);
        array.offset.hash(state);
    }

    fn array_eq(array: &RunEndData, other: &RunEndData, precision: Precision) -> bool {
        array.ends().array_eq(other.ends(), precision)
            && array.values().array_eq(other.values(), precision)
            && array.offset == other.offset
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("RunEndArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        vortex_panic!("RunEndArray buffer_name index {idx} out of bounds")
    }

    fn serialize(array: ArrayView<'_, Self>) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            ProstMetadata(RunEndMetadata {
                ends_ptype: PType::try_from(array.ends().dtype())
                    .vortex_expect("Must be a valid PType") as i32,
                num_runs: array.ends().len() as u64,
                offset: array.offset() as u64,
            })
            .serialize(),
        ))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<RunEndData> {
        let metadata = ProstMetadata::<RunEndMetadata>::deserialize(metadata)?;
        let ends_dtype = DType::Primitive(metadata.ends_ptype(), Nullability::NonNullable);
        let runs = usize::try_from(metadata.num_runs).vortex_expect("Must be a valid usize");
        let ends = children.get(0, &ends_dtype, runs)?;

        let values = children.get(1, dtype, runs)?;

        RunEndData::try_new_offset_length(
            ends,
            values,
            usize::try_from(metadata.offset).vortex_expect("Offset must be a valid usize"),
            len,
        )
    }

    fn slots(array: ArrayView<'_, Self>) -> &[Option<ArrayRef>] {
        &array.data().slots
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn with_slots(array: &mut Self::ArrayData, slots: Vec<Option<ArrayRef>>) -> VortexResult<()> {
        vortex_ensure!(
            slots.len() == NUM_SLOTS,
            "RunEndArray expects exactly {} slots, got {}",
            NUM_SLOTS,
            slots.len()
        );
        array.slots = slots;
        Ok(())
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        RULES.evaluate(array, parent, child_idx)
    }

    fn execute_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_KERNELS.execute(array, parent, child_idx, ctx)
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        run_end_canonicalize(&array, ctx).map(ExecutionResult::done)
    }
}

/// The run-end positions marking where each run terminates.
pub(super) const ENDS_SLOT: usize = 0;
/// The values for each run.
pub(super) const VALUES_SLOT: usize = 1;
pub(super) const NUM_SLOTS: usize = 2;
pub(super) const SLOT_NAMES: [&str; NUM_SLOTS] = ["ends", "values"];

#[derive(Clone, Debug)]
pub struct RunEndData {
    pub(super) slots: Vec<Option<ArrayRef>>,
    offset: usize,
}

pub struct RunEndDataParts {
    pub ends: ArrayRef,
    pub values: ArrayRef,
    pub offset: usize,
}

#[derive(Clone, Debug)]
pub struct RunEnd;

impl RunEnd {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.runend");

    /// Build a new [`RunEndArray`] without validation.
    ///
    /// # Safety
    /// See [`RunEndData::new_unchecked`] for preconditions.
    pub unsafe fn new_unchecked(
        ends: ArrayRef,
        values: ArrayRef,
        offset: usize,
        length: usize,
    ) -> RunEndArray {
        let data = unsafe { RunEndData::new_unchecked(ends, values, offset, length) };
        let dtype = data.dtype().clone();
        Array::try_from_parts(ArrayParts::new(RunEnd, dtype, length, data))
            .vortex_expect("RunEndData is always valid")
    }

    /// Build a new [`RunEndArray`] from ends and values.
    pub fn try_new(ends: ArrayRef, values: ArrayRef) -> VortexResult<RunEndArray> {
        let len = RunEndData::logical_len_from_ends(&ends)?;
        let data = RunEndData::try_new_offset_length(ends, values, 0, len)?;
        let dtype = data.dtype().clone();
        Array::try_from_parts(ArrayParts::new(RunEnd, dtype, len, data))
    }

    /// Build a new [`RunEndArray`] from ends, values, offset, and length.
    pub fn try_new_offset_length(
        ends: ArrayRef,
        values: ArrayRef,
        offset: usize,
        length: usize,
    ) -> VortexResult<RunEndArray> {
        let data = RunEndData::try_new_offset_length(ends, values, offset, length)?;
        let dtype = data.dtype().clone();
        Array::try_from_parts(ArrayParts::new(RunEnd, dtype, length, data))
    }

    /// Build a new [`RunEndArray`] from ends and values (panics on invalid input).
    pub fn new(ends: ArrayRef, values: ArrayRef) -> RunEndArray {
        Self::try_new(ends, values).vortex_expect("RunEndData is always valid")
    }

    /// Run the array through run-end encoding.
    pub fn encode(array: ArrayRef) -> VortexResult<RunEndArray> {
        let len = array.len();
        let data = RunEndData::encode(array)?;
        let dtype = data.dtype().clone();
        Array::try_from_parts(ArrayParts::new(RunEnd, dtype, len, data))
    }
}

impl RunEndData {
    fn logical_len_from_ends(ends: &ArrayRef) -> VortexResult<usize> {
        if ends.is_empty() {
            Ok(0)
        } else {
            usize::try_from(&ends.scalar_at(ends.len() - 1)?)
        }
    }

    fn validate(
        ends: &ArrayRef,
        values: &ArrayRef,
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

        // Zero-length logical slices may retain run metadata from the source array.
        if length == 0 {
            return Ok(());
        }

        debug_assert!({
            // Run ends must be strictly sorted for binary search to work correctly.
            let pre_validation = ends.statistics().to_owned();

            let is_sorted = ends
                .statistics()
                .compute_is_strict_sorted()
                .unwrap_or(false);

            // Preserve the original statistics since compute_is_strict_sorted may have mutated them.
            // We don't want to run with different stats in debug mode and outside.
            ends.statistics().inherit(pre_validation.iter());
            is_sorted
        });

        // Skip host-only validation when ends are not host-resident.
        if !ends.is_host() {
            return Ok(());
        }

        // Validate the offset and length are valid for the given ends and values
        if offset != 0 && length != 0 {
            let first_run_end = usize::try_from(&ends.scalar_at(0)?)?;
            if first_run_end < offset {
                vortex_bail!(
                    "First run end {first_run_end} must be >= offset {offset}"
                );
            }
        }

        let last_run_end = usize::try_from(&ends.scalar_at(ends.len() - 1)?)?;
        let min_required_end = offset + length;
        if last_run_end < min_required_end {
            vortex_bail!("Last run end {last_run_end} must be >= offset+length {min_required_end}");
        }

        Ok(())
    }
}

impl RunEndData {
    /// Build a new `RunEndArray` from an array of run `ends` and an array of `values`.
    ///
    /// Panics if any of the validation conditions described in [`RunEnd::try_new`] is
    /// not satisfied.
    ///
    /// # Examples
    ///
    /// ```
    /// # use vortex_array::arrays::BoolArray;
    /// # use vortex_array::IntoArray;
    /// # use vortex_buffer::buffer;
    /// # use vortex_error::VortexResult;
    /// # use vortex_runend::RunEnd;
    /// # fn main() -> VortexResult<()> {
    /// let ends = buffer![2u8, 3u8].into_array();
    /// let values = BoolArray::from_iter([false, true]).into_array();
    /// let run_end = RunEnd::new(ends, values);
    ///
    /// // Array encodes
    /// assert_eq!(run_end.scalar_at(0)?, false.into());
    /// assert_eq!(run_end.scalar_at(1)?, false.into());
    /// assert_eq!(run_end.scalar_at(2)?, true.into());
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(ends: ArrayRef, values: ArrayRef) -> Self {
        Self::try_new(ends, values).vortex_expect("RunEndArray new")
    }

    /// Build a new `RunEndArray` from components.
    ///
    /// # Validation
    ///
    /// The `ends` must be non-nullable unsigned integers.
    pub(crate) fn try_new(ends: ArrayRef, values: ArrayRef) -> VortexResult<Self> {
        let length = Self::logical_len_from_ends(&ends)?;
        Self::try_new_offset_length(ends, values, 0, length)
    }

    /// Construct a new sliced `RunEndArray` with the provided offset and length.
    ///
    /// This performs all the same validation as [`RunEnd::try_new_offset_length`].
    pub(crate) fn try_new_offset_length(
        ends: ArrayRef,
        values: ArrayRef,
        offset: usize,
        length: usize,
    ) -> VortexResult<Self> {
        Self::validate(&ends, &values, offset, length)?;

        Ok(Self {
            slots: vec![Some(ends), Some(values)],
            offset,
        })
    }

    /// Build a new `RunEndArray` without validation.
    ///
    /// # Safety
    ///
    /// The caller must ensure that all the validation performed in
    /// [`RunEnd::try_new_offset_length`] is
    /// satisfied before calling this function.
    ///
    /// See [`RunEnd::try_new_offset_length`] for the preconditions needed to build a new array.
    pub unsafe fn new_unchecked(
        ends: ArrayRef,
        values: ArrayRef,
        offset: usize,
        _length: usize,
    ) -> Self {
        Self {
            slots: vec![Some(ends), Some(values)],
            offset,
        }
    }

    /// Convert the given logical index to an index into the `values` array
    pub fn find_physical_index(&self, index: usize) -> VortexResult<usize> {
        Ok(self
            .ends()
            .as_primitive_typed()
            .search_sorted(
                &PValue::from(index + self.offset()),
                SearchSortedSide::Right,
            )?
            .to_ends_index(self.ends().len()))
    }

    /// Run the array through run-end encoding.
    pub fn encode(array: ArrayRef) -> VortexResult<Self> {
        if let Some(parray) = array.as_opt::<Primitive>() {
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

    /// Returns the logical data type of the array.
    #[inline]
    pub fn dtype(&self) -> &DType {
        self.values().dtype()
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
        self.slots[ENDS_SLOT]
            .as_ref()
            .vortex_expect("RunEndArray ends slot")
    }

    /// The scalar values.
    ///
    /// The `i`-th element is the scalar value for the `i`-th repeated run. The run begins
    /// at `ends[i]` (inclusive) and terminates at `ends[i+1]` (exclusive).
    #[inline]
    pub fn values(&self) -> &ArrayRef {
        self.slots[VALUES_SLOT]
            .as_ref()
            .vortex_expect("RunEndArray values slot")
    }

    pub fn into_parts(mut self) -> RunEndDataParts {
        RunEndDataParts {
            ends: self.slots[ENDS_SLOT]
                .take()
                .vortex_expect("RunEndArray ends slot"),
            values: self.slots[VALUES_SLOT]
                .take()
                .vortex_expect("RunEndArray values slot"),
            offset: self.offset,
        }
    }
}

impl ValidityVTable<RunEnd> for RunEnd {
    fn validity(array: ArrayView<'_, RunEnd>) -> VortexResult<Validity> {
        Ok(match array.values().validity()? {
            Validity::NonNullable | Validity::AllValid => Validity::AllValid,
            Validity::AllInvalid => Validity::AllInvalid,
            Validity::Array(values_validity) => Validity::Array(unsafe {
                RunEnd::new_unchecked(
                    array.ends().clone(),
                    values_validity,
                    array.offset(),
                    array.len(),
                )
                .into_array()
            }),
        })
    }
}

pub(super) fn run_end_canonicalize(
    array: &RunEndArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let pends = array.ends().clone().execute_as("ends", ctx)?;

    Ok(match array.dtype() {
        DType::Bool(_) => {
            let bools = array.values().clone().execute_as("values", ctx)?;
            runend_decode_bools(pends, bools, array.offset(), array.len())?
        }
        DType::Primitive(..) => {
            let pvalues = array.values().clone().execute_as("values", ctx)?;
            runend_decode_primitive(pends, pvalues, array.offset(), array.len())?.into_array()
        }
        DType::Utf8(_) | DType::Binary(_) => {
            let values = array
                .values()
                .clone()
                .execute_as::<VarBinViewArray>("values", ctx)?;
            runend_decode_varbinview(pends, values, array.offset(), array.len())?.into_array()
        }
        _ => vortex_bail!("Unsupported RunEnd value type: {}", array.dtype()),
    })
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::arrays::DictArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_buffer::buffer;

    use crate::RunEnd;

    #[test]
    fn test_runend_constructor() {
        let arr = RunEnd::new(
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
        assert_arrays_eq!(arr.into_array(), expected);
    }

    #[test]
    fn test_runend_utf8() {
        let values = VarBinViewArray::from_iter_str(["a", "b", "c"]).into_array();
        let arr = RunEnd::new(buffer![2u32, 5, 10].into_array(), values);
        assert_eq!(arr.len(), 10);
        assert_eq!(arr.dtype(), &DType::Utf8(Nullability::NonNullable));

        let expected =
            VarBinViewArray::from_iter_str(["a", "a", "b", "b", "b", "c", "c", "c", "c", "c"])
                .into_array();
        assert_arrays_eq!(arr.into_array(), expected);
    }

    #[test]
    fn test_runend_dict() {
        let dict_values = VarBinViewArray::from_iter_str(["x", "y", "z"]).into_array();
        let dict_codes = buffer![0u32, 1, 2].into_array();
        let dict = DictArray::try_new(dict_codes, dict_values).unwrap();

        let arr = RunEnd::try_new(buffer![2u32, 5, 10].into_array(), dict.into_array()).unwrap();
        assert_eq!(arr.len(), 10);

        let expected =
            VarBinViewArray::from_iter_str(["x", "x", "y", "y", "y", "z", "z", "z", "z", "z"])
                .into_array();
        assert_arrays_eq!(arr.into_array(), expected);
    }
}
