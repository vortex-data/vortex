// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;

use prost::Message;
use vortex_error::VortexExpect as _;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::AnyCanonical;
use crate::Array;
use crate::ArrayEq;
use crate::ArrayHash;
use crate::ArrayId;
use crate::ArrayParts;
use crate::ArrayRef;
use crate::ArrayView;
use crate::EqMode;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::IntoArray;
use crate::TypedArrayRef;
use crate::VortexSessionExecute;
use crate::array_slots;
use crate::arrays::DecimalArray;
use crate::arrays::ListViewArray;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::VarBinViewArray;
use crate::arrays::listview::ListViewArraySlotsExt;
use crate::arrays::runend::compress::runend_decode_decimal;
use crate::arrays::runend::compress::runend_decode_primitive;
use crate::arrays::runend::compress::runend_decode_varbinview;
use crate::arrays::runend::compress::runend_encode;
use crate::arrays::runend::decompress_bool::runend_decode_bools;
use crate::arrays::runend::ops::find_physical_index;
use crate::arrays::runend::ops::find_slice_end_index;
use crate::arrays::runend::rules::RULES;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::require_child;
use crate::serde::ArrayChildren;
use crate::validity::Validity;
use crate::vtable::VTable;
use crate::vtable::ValidityVTable;

/// A [`RunEnd`]-encoded Vortex array.
pub type RunEndArray = Array<RunEnd>;

#[derive(Clone, prost::Message)]
pub struct RunEndMetadata {
    #[prost(int32, tag = "1")]
    pub(crate) ends_ptype: i32,
    #[prost(uint64, tag = "2")]
    pub(crate) num_runs: u64,
    #[prost(uint64, tag = "3")]
    pub(crate) offset: u64,
}

impl RunEndMetadata {
    fn get_ends_ptype(&self) -> VortexResult<PType> {
        PType::try_from(self.ends_ptype).map_err(|_| {
            vortex_err!(
                "RunEndMetadata ends_ptype must be a known PType, got {}",
                self.ends_ptype
            )
        })
    }
}

impl ArrayHash for RunEndData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _accuracy: EqMode) {
        self.offset.hash(state);
    }
}

impl ArrayEq for RunEndData {
    fn array_eq(&self, other: &Self, _accuracy: EqMode) -> bool {
        self.offset == other.offset
    }
}

impl VTable for RunEnd {
    type TypedArrayData = RunEndData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.runend");
        *ID
    }

    fn validate(
        &self,
        data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let run_end_slots = RunEndSlotsView::from_slots(slots);
        let ends = run_end_slots.ends;
        let values = run_end_slots.values;
        RunEndData::validate_shape(ends, values, data.offset, len)?;
        vortex_ensure_eq!(values.dtype(), dtype);
        Ok(())
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

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        crate::vtable::with_empty_buffers(self, array, buffers)
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            RunEndMetadata {
                ends_ptype: PType::try_from(array.ends().dtype())
                    .vortex_expect("Must be a valid PType") as i32,
                num_runs: array.ends().len() as u64,
                offset: array.offset() as u64,
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        let metadata = RunEndMetadata::decode(metadata)?;
        let ends_ptype = metadata.get_ends_ptype()?;
        let ends_dtype = DType::Primitive(ends_ptype, Nullability::NonNullable);
        let runs = usize::try_from(metadata.num_runs).map_err(|_| {
            vortex_err!(
                "RunEndMetadata num_runs {} exceeds usize range",
                metadata.num_runs
            )
        })?;
        let ends = children.get(0, &ends_dtype, runs)?;

        let values = children.get(1, dtype, runs)?;
        let offset = usize::try_from(metadata.offset).map_err(|_| {
            vortex_err!(
                "RunEndMetadata offset {} exceeds usize range",
                metadata.offset
            )
        })?;
        RunEndData::validate_parts(
            &ends,
            &values,
            offset,
            len,
            &mut session.create_execution_ctx(),
        )?;
        let slots = RunEndSlots { ends, values }.into_slots();
        let data = RunEndData::new(offset);
        Ok(ArrayParts::new(
            self.clone(),
            dtype.clone(),
            len,
            data,
            slots,
        ))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        RunEndSlots::NAMES[idx].to_string()
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        RULES.evaluate(array, parent, child_idx)
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let array = require_child!(array, array.ends(), RunEndSlots::ENDS => Primitive);
        let array = require_child!(array, array.values(), RunEndSlots::VALUES => AnyCanonical);
        run_end_canonicalize(&array, ctx).map(ExecutionResult::done)
    }
}

#[array_slots(RunEnd)]
pub struct RunEndSlots {
    /// The run-end positions marking where each run terminates.
    #[slot(0)]
    pub ends: ArrayRef,
    /// The values for each run.
    #[slot(1)]
    pub values: ArrayRef,
}

#[derive(Clone, Debug)]
pub struct RunEndData {
    offset: usize,
}

impl Display for RunEndData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "offset: {}", self.offset)
    }
}

pub struct RunEndDataParts {
    pub ends: ArrayRef,
    pub values: ArrayRef,
    pub offset: usize,
}

pub trait RunEndArrayExt: RunEndArraySlotsExt {
    fn offset(&self) -> usize {
        self.offset
    }

    fn dtype(&self) -> &DType {
        self.values().dtype()
    }

    fn find_physical_index(&self, index: usize, ctx: &mut ExecutionCtx) -> VortexResult<usize> {
        find_physical_index(self.ends(), index + self.offset(), ctx)
    }

    fn find_slice_end_index(&self, index: usize, ctx: &mut ExecutionCtx) -> VortexResult<usize> {
        find_slice_end_index(self.ends(), index + self.offset(), ctx)
    }
}

impl<T: TypedArrayRef<RunEnd>> RunEndArrayExt for T {}

#[derive(Clone, Debug)]
pub struct RunEnd;

impl RunEnd {
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
        let dtype = values.dtype().clone();
        let slots = RunEndSlots { ends, values }.into_slots();
        let data = unsafe { RunEndData::new_unchecked(offset) };
        unsafe { Array::from_parts_unchecked(ArrayParts::new(RunEnd, dtype, length, data, slots)) }
    }

    /// Build a new [`RunEndArray`] from ends and values.
    pub fn try_new(
        ends: ArrayRef,
        values: ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<RunEndArray> {
        let len = RunEndData::logical_len_from_ends(&ends, ctx)?;
        RunEndData::validate_parts(&ends, &values, 0, len, ctx)?;
        let dtype = values.dtype().clone();
        let slots = RunEndSlots { ends, values }.into_slots();
        let data = RunEndData::new(0);
        Array::try_from_parts(ArrayParts::new(RunEnd, dtype, len, data, slots))
    }

    /// Build a new [`RunEndArray`] from ends, values, offset, and length.
    pub fn try_new_offset_length(
        ends: ArrayRef,
        values: ArrayRef,
        offset: usize,
        length: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<RunEndArray> {
        RunEndData::validate_parts(&ends, &values, offset, length, ctx)?;
        let dtype = values.dtype().clone();
        let slots = RunEndSlots { ends, values }.into_slots();
        let data = RunEndData::new(offset);
        Array::try_from_parts(ArrayParts::new(RunEnd, dtype, length, data, slots))
    }

    /// Build a new [`RunEndArray`] from ends and values (panics on invalid input).
    pub fn new(ends: ArrayRef, values: ArrayRef, ctx: &mut ExecutionCtx) -> RunEndArray {
        Self::try_new(ends, values, ctx).vortex_expect("RunEndData is always valid")
    }

    /// Run the array through run-end encoding.
    pub fn encode(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<RunEndArray> {
        if let Some(parray) = array.as_opt::<Primitive>() {
            let (ends, values) = runend_encode(parray, ctx);
            let ends = ends.into_array();
            let len = array.len();
            let dtype = values.dtype().clone();
            let slots = RunEndSlots { ends, values }.into_slots();
            let data = unsafe { RunEndData::new_unchecked(0) };
            Array::try_from_parts(ArrayParts::new(RunEnd, dtype, len, data, slots))
        } else {
            vortex_bail!("REE can only encode primitive arrays")
        }
    }
}

impl RunEndData {
    fn logical_len_from_ends(ends: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<usize> {
        if ends.is_empty() {
            Ok(0)
        } else {
            usize::try_from(&ends.execute_scalar(ends.len() - 1, ctx)?)
        }
    }

    /// Validate that `ends` and `values` form a well-formed run-end array covering
    /// `offset..offset + length`.
    pub fn validate_parts(
        ends: &ArrayRef,
        values: &ArrayRef,
        offset: usize,
        length: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        if !Self::validate_shape(ends, values, offset, length)? {
            return Ok(());
        }

        #[cfg(debug_assertions)]
        {
            // Run ends must be strictly sorted for binary search to work correctly.
            let pre_validation = ends.statistics().to_owned();

            let is_sorted = ends
                .statistics()
                .compute_is_strict_sorted(ctx)
                .unwrap_or(false);

            // Preserve the original statistics since compute_is_strict_sorted may have mutated them.
            // We don't want to run with different stats in debug mode and outside.
            ends.statistics().inherit(pre_validation.iter());
            debug_assert!(is_sorted);
        }

        // Skip host-only validation when ends are not host-resident.
        if !ends.is_host() {
            return Ok(());
        }

        let first_run_end = if offset == 0 {
            0
        } else {
            usize::try_from(&ends.execute_scalar(0, ctx)?)?
        };
        let last_run_end = usize::try_from(&ends.execute_scalar(ends.len() - 1, ctx)?)?;
        Self::validate_run_end_bounds(first_run_end, last_run_end, offset, length)
    }

    /// Validates the dtypes and lengths of the parts. Returns `false` when there are no runs to
    /// check against `offset..offset + length`.
    fn validate_shape(
        ends: &ArrayRef,
        values: &ArrayRef,
        offset: usize,
        length: usize,
    ) -> VortexResult<bool> {
        // DType validation
        vortex_ensure!(
            ends.dtype().is_unsigned_int(),
            "run ends must be unsigned integers, was {}",
            ends.dtype(),
        );
        vortex_ensure_eq!(ends.len(), values.len());

        // Handle empty run-ends
        if ends.is_empty() {
            vortex_ensure_eq!(offset, 0, "non-zero offset provided for empty RunEndArray");
            return Ok(false);
        }

        // Zero-length logical slices may retain run metadata from the source array.
        Ok(length != 0)
    }

    /// Validates that the first and last run ends cover `offset..offset + length`.
    fn validate_run_end_bounds(
        first_run_end: usize,
        last_run_end: usize,
        offset: usize,
        length: usize,
    ) -> VortexResult<()> {
        if first_run_end < offset {
            vortex_bail!("First run end {first_run_end} must be >= offset {offset}");
        }

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
    /// # use vortex_array::arrays::runend::RunEnd;
    /// # use vortex_array::IntoArray;
    /// # use vortex_session::VortexSession;
    /// # use vortex_array::VortexSessionExecute;
    /// # use vortex_buffer::buffer;
    /// # use vortex_error::VortexResult;
    /// # fn main() -> VortexResult<()> {
    /// let session = VortexSession::empty();
    /// let mut ctx = session.create_execution_ctx();
    /// let ends = buffer![2u8, 3u8].into_array();
    /// let values = BoolArray::from_iter([false, true]).into_array();
    /// let run_end = RunEnd::new(ends, values, &mut ctx);
    ///
    /// // Array encodes
    /// assert_eq!(run_end.execute_scalar(0, &mut ctx)?, false.into());
    /// assert_eq!(run_end.execute_scalar(1, &mut ctx)?, false.into());
    /// assert_eq!(run_end.execute_scalar(2, &mut ctx)?, true.into());
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(offset: usize) -> Self {
        Self { offset }
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
    pub unsafe fn new_unchecked(offset: usize) -> Self {
        Self { offset }
    }

    /// Run the array through run-end encoding.
    pub fn encode(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        if let Some(parray) = array.as_opt::<Primitive>() {
            let (_ends, _values) = runend_encode(parray, ctx);
            // SAFETY: runend_encode handles this
            unsafe { Ok(Self::new_unchecked(0)) }
        } else {
            vortex_bail!("REE can only encode primitive arrays")
        }
    }

    pub fn into_parts(self, ends: ArrayRef, values: ArrayRef) -> RunEndDataParts {
        RunEndDataParts {
            ends,
            values,
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
            runend_decode_bools(pends, bools, array.offset(), array.len(), ctx)?
        }
        DType::Primitive(..) => {
            let pvalues = array.values().clone().execute_as("values", ctx)?;
            runend_decode_primitive(pends, pvalues, array.offset(), array.len(), ctx)?
        }
        DType::Decimal(..) => {
            let values = array
                .values()
                .clone()
                .execute_as::<DecimalArray>("values", ctx)?;
            runend_decode_decimal(pends, values, array.offset(), array.len(), ctx)?
        }
        DType::Utf8(_) | DType::Binary(_) => {
            let values = array
                .values()
                .clone()
                .execute_as::<VarBinViewArray>("values", ctx)?;
            runend_decode_varbinview(pends, values, array.offset(), array.len(), ctx)?
        }
        DType::List(..) => {
            let values = array
                .values()
                .clone()
                .execute_as::<ListViewArray>("values", ctx)?;
            runend_decode_listview(pends, values, array.offset(), array.len())?.into_array()
        }
        _ => vortex_bail!("Unsupported RunEnd value type: {}", array.dtype()),
    })
}

fn runend_decode_listview(
    ends: PrimitiveArray,
    values: ListViewArray,
    offset: usize,
    length: usize,
) -> VortexResult<ListViewArray> {
    let validity = match values.validity()? {
        Validity::NonNullable => Validity::NonNullable,
        Validity::AllValid => Validity::AllValid,
        Validity::AllInvalid => Validity::AllInvalid,
        Validity::Array(validity) => Validity::Array(unsafe {
            RunEnd::new_unchecked(ends.clone().into_array(), validity, offset, length).into_array()
        }),
    };

    // SAFETY: the `RunEndArray`s re-express valid per-run ListView metadata over the logical output
    // length. The original `elements` child is reused, so every view still points at a valid range.
    Ok(unsafe {
        ListViewArray::new_unchecked(
            values.elements().clone(),
            RunEnd::new_unchecked(
                ends.clone().into_array(),
                values.offsets().clone(),
                offset,
                length,
            )
            .into_array(),
            RunEnd::new_unchecked(ends.into_array(), values.sizes().clone(), offset, length)
                .into_array(),
            validity,
        )
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::LazyLock;

    use prost::Message;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::RunEndMetadata;
    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::DecimalArray;
    use crate::arrays::DictArray;
    use crate::arrays::ListArray;
    use crate::arrays::ListViewArray;
    use crate::arrays::VarBinViewArray;
    use crate::arrays::listview::ListViewArraySlotsExt;
    use crate::arrays::runend::RunEnd;
    use crate::assert_arrays_eq;
    use crate::builders::VarBinBuilder;
    use crate::dtype::DType;
    use crate::dtype::DecimalDType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::i256;
    use crate::validity::Validity;
    use crate::vtable::VTable;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[test]
    fn test_runend_constructor() {
        let mut ctx = SESSION.create_execution_ctx();
        let arr = RunEnd::new(
            buffer![2u32, 5, 10].into_array(),
            buffer![1i32, 2, 3].into_array(),
            &mut ctx,
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
        assert_arrays_eq!(arr.into_array(), expected, &mut ctx);
    }

    #[test]
    fn test_runend_utf8() {
        let mut ctx = SESSION.create_execution_ctx();
        let values =
            VarBinViewArray::from_iter_nullable_str([Some("a"), None, Some("c")]).into_array();
        let arr = RunEnd::new(buffer![2u32, 5, 10].into_array(), values, &mut ctx);
        assert_eq!(arr.len(), 10);
        assert_eq!(arr.dtype(), &DType::Utf8(Nullability::Nullable));

        let expected = VarBinViewArray::from_iter_nullable_str([
            Some("a"),
            Some("a"),
            None,
            None,
            None,
            Some("c"),
            Some("c"),
            Some("c"),
            Some("c"),
            Some("c"),
        ])
        .into_array();
        let mut builder = VarBinBuilder::<i32>::with_capacity_in(
            arr.dtype().clone(),
            arr.len(),
            vortex_buffer::BufferAllocatorRef::static_ref(),
        );
        arr.append_to_builder(&mut builder, &mut ctx).unwrap();
        assert_arrays_eq!(builder.finish_into_varbin(), expected, &mut ctx);
        assert_arrays_eq!(arr.into_array(), expected, &mut ctx);
    }

    #[test]
    fn test_runend_decimal() {
        let mut ctx = SESSION.create_execution_ctx();
        let decimal_dtype = DecimalDType::new(10, 2);
        let values = DecimalArray::from_iter([12345i64, 67890, -12300], decimal_dtype).into_array();
        let arr = RunEnd::new(buffer![2u32, 5, 10].into_array(), values, &mut ctx);
        assert_eq!(arr.len(), 10);
        assert_eq!(
            arr.dtype(),
            &DType::Decimal(decimal_dtype, Nullability::NonNullable)
        );

        let expected = DecimalArray::from_iter(
            [
                12345i64, 12345, 67890, 67890, 67890, -12300, -12300, -12300, -12300, -12300,
            ],
            decimal_dtype,
        )
        .into_array();
        assert_arrays_eq!(arr.into_array(), expected, &mut ctx);
    }

    #[test]
    fn test_runend_list_i64() {
        let mut ctx = SESSION.create_execution_ctx();
        let values = ListArray::from_iter_slow::<u32, _>(
            vec![vec![1i64, 2], vec![3], vec![4, 5, 6]],
            Arc::new(DType::Primitive(PType::I64, Nullability::NonNullable)),
        )
        .unwrap()
        .into_array();
        let arr = RunEnd::new(buffer![2u32, 5, 10].into_array(), values, &mut ctx);

        let expected = ListArray::from_iter_slow::<u32, _>(
            vec![
                vec![1i64, 2],
                vec![1, 2],
                vec![3],
                vec![3],
                vec![3],
                vec![4, 5, 6],
                vec![4, 5, 6],
                vec![4, 5, 6],
                vec![4, 5, 6],
                vec![4, 5, 6],
            ],
            Arc::new(DType::Primitive(PType::I64, Nullability::NonNullable)),
        )
        .unwrap()
        .into_array();
        assert_arrays_eq!(arr.into_array(), expected, &mut ctx);
    }

    #[test]
    fn test_runend_nullable_decimal() {
        let mut ctx = SESSION.create_execution_ctx();
        let decimal_dtype = DecimalDType::new(10, 2);
        let values =
            DecimalArray::from_option_iter([Some(12345i64), None, Some(-12300)], decimal_dtype)
                .into_array();
        let arr = RunEnd::new(buffer![2u32, 5, 10].into_array(), values, &mut ctx);
        assert_eq!(arr.len(), 10);
        assert_eq!(
            arr.dtype(),
            &DType::Decimal(decimal_dtype, Nullability::Nullable)
        );

        let expected = DecimalArray::from_option_iter(
            [
                Some(12345i64),
                Some(12345),
                None,
                None,
                None,
                Some(-12300),
                Some(-12300),
                Some(-12300),
                Some(-12300),
                Some(-12300),
            ],
            decimal_dtype,
        )
        .into_array();
        assert_arrays_eq!(arr.into_array(), expected, &mut ctx);
    }

    #[test]
    fn test_runend_list_bool() {
        let mut ctx = SESSION.create_execution_ctx();
        let values = ListArray::from_iter_slow::<u32, _>(
            vec![vec![true, false], vec![false], vec![true, true, false]],
            Arc::new(DType::Bool(Nullability::NonNullable)),
        )
        .unwrap()
        .into_array();
        let arr = RunEnd::new(buffer![2u32, 5, 10].into_array(), values, &mut ctx);

        let expected = ListArray::from_iter_slow::<u32, _>(
            vec![
                vec![true, false],
                vec![true, false],
                vec![false],
                vec![false],
                vec![false],
                vec![true, true, false],
                vec![true, true, false],
                vec![true, true, false],
                vec![true, true, false],
                vec![true, true, false],
            ],
            Arc::new(DType::Bool(Nullability::NonNullable)),
        )
        .unwrap()
        .into_array();
        assert_arrays_eq!(arr.into_array(), expected, &mut ctx);
    }

    #[test]
    fn test_runend_list_utf8() {
        let mut ctx = SESSION.create_execution_ctx();
        let values = ListArray::try_new(
            VarBinViewArray::from_iter_str(["a", "b", "c", "d", "e", "f"]).into_array(),
            buffer![0u32, 2, 3, 6].into_array(),
            Validity::NonNullable,
            &mut ctx,
        )
        .unwrap()
        .into_array();
        let arr = RunEnd::new(buffer![2u32, 5, 10].into_array(), values, &mut ctx);

        let expected = ListArray::try_new(
            VarBinViewArray::from_iter_str([
                "a", "b", "a", "b", "c", "c", "c", "d", "e", "f", "d", "e", "f", "d", "e", "f",
                "d", "e", "f", "d", "e", "f",
            ])
            .into_array(),
            buffer![0u32, 2, 4, 5, 6, 7, 10, 13, 16, 19, 22].into_array(),
            Validity::NonNullable,
            &mut ctx,
        )
        .unwrap()
        .into_array();
        assert_arrays_eq!(arr.into_array(), expected, &mut ctx);
    }

    #[test]
    fn test_runend_list_canonicalizes_to_runend_listview_slots() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = ListArray::try_new(
            buffer![1i64, 2, 3, 4, 5, 6].into_array(),
            buffer![0u32, 2, 3, 6].into_array(),
            Validity::from_iter([true, false, true]),
            &mut ctx,
        )?
        .into_array();
        let arr = RunEnd::try_new(buffer![2u32, 5, 6].into_array(), values, &mut ctx)?;

        let listview = arr
            .clone()
            .into_array()
            .execute::<ListViewArray>(&mut ctx)?;
        assert!(listview.offsets().is::<RunEnd>());
        assert!(listview.sizes().is::<RunEnd>());
        match listview.validity()? {
            Validity::Array(validity) => assert!(validity.is::<RunEnd>()),
            validity => panic!("expected array-backed validity, got {validity:?}"),
        }

        let expected = ListArray::try_new(
            buffer![1i64, 2, 1, 2, 3, 3, 3, 4, 5, 6].into_array(),
            buffer![0u32, 2, 4, 5, 6, 7, 10].into_array(),
            Validity::from_iter([true, true, false, false, false, true]),
            &mut ctx,
        )?
        .into_array();
        assert_arrays_eq!(arr.into_array(), expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_runend_dict() {
        let mut ctx = SESSION.create_execution_ctx();
        let dict_values = VarBinViewArray::from_iter_str(["x", "y", "z"]).into_array();
        let dict_codes = buffer![0u32, 1, 2].into_array();
        let dict = DictArray::try_new(dict_codes, dict_values).unwrap();

        let arr = RunEnd::try_new(
            buffer![2u32, 5, 10].into_array(),
            dict.into_array(),
            &mut ctx,
        )
        .unwrap();
        assert_eq!(arr.len(), 10);

        let expected =
            VarBinViewArray::from_iter_str(["x", "x", "y", "y", "y", "z", "z", "z", "z", "z"])
                .into_array();
        assert_arrays_eq!(arr.into_array(), expected, &mut ctx);
    }

    #[test]
    fn test_runend_decimal_i128() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let decimal_dtype = DecimalDType::new(20, 2);
        let values = DecimalArray::from_iter([12_345i128, -67_890, 100], decimal_dtype);
        let arr = RunEnd::try_new(
            buffer![2u32, 5, 6].into_array(),
            values.into_array(),
            &mut ctx,
        )?;

        let expected = DecimalArray::from_iter(
            [12_345i128, 12_345, -67_890, -67_890, -67_890, 100],
            decimal_dtype,
        );
        assert_arrays_eq!(arr, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_runend_decimal_nullable() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let decimal_dtype = DecimalDType::new(20, 2);
        let values =
            DecimalArray::from_option_iter([Some(12_345i128), None, Some(-67_890)], decimal_dtype);
        let arr = RunEnd::try_new(
            buffer![2u32, 5, 7].into_array(),
            values.into_array(),
            &mut ctx,
        )?;

        let expected = DecimalArray::from_option_iter(
            [
                Some(12_345i128),
                Some(12_345),
                None,
                None,
                None,
                Some(-67_890),
                Some(-67_890),
            ],
            decimal_dtype,
        );
        assert_arrays_eq!(arr, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_runend_decimal_slice() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let decimal_dtype = DecimalDType::new(20, 2);
        let values = DecimalArray::from_iter([100i128, 200, 300], decimal_dtype);
        let arr = RunEnd::try_new(
            buffer![3u32, 5, 10].into_array(),
            values.into_array(),
            &mut ctx,
        )?;

        let sliced = arr.slice(2..8)?;
        let expected = DecimalArray::from_iter([100i128, 200, 200, 300, 300, 300], decimal_dtype);
        assert_arrays_eq!(sliced, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_runend_decimal_i256() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let decimal_dtype = DecimalDType::new(40, 4);
        let first = i256::from_i128(123_456);
        let second = i256::from_i128(-789_012);
        let values = DecimalArray::from_iter([first, second], decimal_dtype);
        let arr = RunEnd::try_new(buffer![2u32, 5].into_array(), values.into_array(), &mut ctx)?;

        let expected =
            DecimalArray::from_iter([first, first, second, second, second], decimal_dtype);
        assert_arrays_eq!(arr, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_deserialize_rejects_unknown_ends_ptype() -> VortexResult<()> {
        let metadata = RunEndMetadata {
            ends_ptype: 11,
            ..Default::default()
        }
        .encode_to_vec();
        let children: Vec<ArrayRef> = Vec::new();
        let result = RunEnd.deserialize(
            &DType::Primitive(PType::I32, Nullability::NonNullable),
            0,
            &metadata,
            &[],
            &children,
            &SESSION,
        );
        assert!(result.is_err_and(|err| {
            err.to_string()
                .contains("RunEndMetadata ends_ptype must be a known PType, got 11")
        }));

        let known = RunEndMetadata {
            ends_ptype: PType::U32 as i32,
            ..Default::default()
        };
        assert_eq!(known.get_ends_ptype()?, PType::U32);

        Ok(())
    }

    // On 32-bit targets, a crafted num_runs / offset above usize::MAX used to panic via
    // vortex_expect. Keep a regression that only builds where the conversion can fail.
    #[cfg(target_pointer_width = "32")]
    #[test]
    fn test_deserialize_rejects_num_runs_overflow() -> VortexResult<()> {
        let metadata = RunEndMetadata {
            ends_ptype: PType::U32 as i32,
            num_runs: u64::from(u32::MAX) + 1,
            offset: 0,
        }
        .encode_to_vec();
        let children: Vec<ArrayRef> = Vec::new();
        let result = RunEnd.deserialize(
            &DType::Primitive(PType::I32, Nullability::NonNullable),
            0,
            &metadata,
            &[],
            &children,
            &SESSION,
        );
        assert!(result.is_err_and(|err| {
            err.to_string()
                .contains("RunEndMetadata num_runs")
                && err.to_string().contains("exceeds usize range")
        }));
        Ok(())
    }

    #[cfg(target_pointer_width = "32")]
    #[test]
    fn test_deserialize_rejects_offset_overflow() -> VortexResult<()> {
        let metadata = RunEndMetadata {
            ends_ptype: PType::U32 as i32,
            num_runs: 0,
            offset: u64::from(u32::MAX) + 1,
        }
        .encode_to_vec();
        let children: Vec<ArrayRef> = Vec::new();
        let result = RunEnd.deserialize(
            &DType::Primitive(PType::I32, Nullability::NonNullable),
            0,
            &metadata,
            &[],
            &children,
            &SESSION,
        );
        assert!(result.is_err_and(|err| {
            err.to_string()
                .contains("RunEndMetadata offset")
                && err.to_string().contains("exceeds usize range")
        }));
        Ok(())
    }
}
