// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::Array;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::EmptyMetadata;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::Precision;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_array::serde::ArrayChildren;
use vortex_array::stats::ArrayStats;
use vortex_array::vtable;
use vortex_array::vtable::OperationsVTable;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityChild;
use vortex_array::vtable::ValidityVTableFromChild;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use zigzag::ZigZag as ExternalZigZag;

use crate::compute::ZigZagEncoded;
use crate::kernel::PARENT_KERNELS;
use crate::rules::RULES;
use crate::zigzag_decode;

vtable!(ZigZag, ZigZag, ZigZagData);

impl VTable for ZigZag {
    type ArrayData = ZigZagData;

    type Metadata = EmptyMetadata;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    fn vtable(_array: &Self::ArrayData) -> &Self {
        &ZigZag
    }

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn len(array: &ZigZagData) -> usize {
        array.encoded().len()
    }

    fn dtype(array: &ZigZagData) -> &DType {
        &array.dtype
    }

    fn stats(array: &ZigZagData) -> &ArrayStats {
        &array.stats_set
    }

    fn array_hash<H: std::hash::Hasher>(array: &ZigZagData, state: &mut H, precision: Precision) {
        array.encoded().array_hash(state, precision);
    }

    fn array_eq(array: &ZigZagData, other: &ZigZagData, precision: Precision) -> bool {
        array.encoded().array_eq(other.encoded(), precision)
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("ZigZagArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        vortex_panic!("ZigZagArray buffer_name index {idx} out of bounds")
    }

    fn metadata(_array: ArrayView<'_, Self>) -> VortexResult<Self::Metadata> {
        Ok(EmptyMetadata)
    }

    fn serialize(_metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(
        _bytes: &[u8],
        _dtype: &DType,
        _len: usize,
        _buffers: &[BufferHandle],
        _session: &VortexSession,
    ) -> VortexResult<Self::Metadata> {
        Ok(EmptyMetadata)
    }

    fn build(
        dtype: &DType,
        len: usize,
        _metadata: &Self::Metadata,
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<ZigZagData> {
        if children.len() != 1 {
            vortex_bail!("Expected 1 child, got {}", children.len());
        }

        let ptype = PType::try_from(dtype)?;
        let encoded_type = DType::Primitive(ptype.to_unsigned(), dtype.nullability());

        let encoded = children.get(0, &encoded_type, len)?;
        ZigZagData::try_new(encoded)
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
            "ZigZagArray expects exactly {} slots, got {}",
            NUM_SLOTS,
            slots.len()
        );
        array.slots = slots;
        Ok(())
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(
            zigzag_decode(array.encoded().clone().execute(ctx)?).into_array(),
        ))
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
}

/// The zigzag-encoded values (signed integers mapped to unsigned).
pub(super) const ENCODED_SLOT: usize = 0;
pub(super) const NUM_SLOTS: usize = 1;
pub(super) const SLOT_NAMES: [&str; NUM_SLOTS] = ["encoded"];

#[derive(Clone, Debug)]
pub struct ZigZagData {
    dtype: DType,
    pub(super) slots: Vec<Option<ArrayRef>>,
    stats_set: ArrayStats,
}

#[derive(Clone, Debug)]
pub struct ZigZag;

impl ZigZag {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.zigzag");

    /// Construct a new [`ZigZagArray`] from an encoded unsigned integer array.
    pub fn try_new(encoded: ArrayRef) -> VortexResult<ZigZagArray> {
        Array::try_from_data(ZigZagData::try_new(encoded)?)
    }
}

impl ZigZagData {
    pub fn new(encoded: ArrayRef) -> Self {
        Self::try_new(encoded).vortex_expect("ZigZagArray new")
    }

    pub fn try_new(encoded: ArrayRef) -> VortexResult<Self> {
        let encoded_dtype = encoded.dtype().clone();
        if !encoded_dtype.is_unsigned_int() {
            vortex_bail!(MismatchedTypes: "unsigned int", encoded_dtype);
        }

        let dtype = DType::from(PType::try_from(&encoded_dtype)?.to_signed())
            .with_nullability(encoded_dtype.nullability());

        Ok(Self {
            dtype,
            slots: vec![Some(encoded)],
            stats_set: Default::default(),
        })
    }

    /// Returns the length of the array.
    #[inline]
    pub fn len(&self) -> usize {
        self.encoded().len()
    }

    /// Returns whether the array is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.encoded().is_empty()
    }

    /// Returns the logical data type of the array.
    #[inline]
    pub fn dtype(&self) -> &DType {
        &self.dtype
    }

    pub fn ptype(&self) -> PType {
        self.dtype().as_ptype()
    }

    pub fn encoded(&self) -> &ArrayRef {
        self.slots[ENCODED_SLOT]
            .as_ref()
            .vortex_expect("ZigZagArray encoded slot")
    }
}

impl OperationsVTable<ZigZag> for ZigZag {
    fn scalar_at(
        array: ArrayView<'_, ZigZag>,
        index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let scalar = array.encoded().scalar_at(index)?;
        if scalar.is_null() {
            return scalar.primitive_reinterpret_cast(array.ptype());
        }

        let pscalar = scalar.as_primitive();
        Ok(match_each_unsigned_integer_ptype!(pscalar.ptype(), |P| {
            Scalar::primitive(
                <<P as ZigZagEncoded>::Int>::decode(
                    pscalar
                        .typed_value::<P>()
                        .vortex_expect("zigzag corruption"),
                ),
                array.dtype().nullability(),
            )
        }))
    }
}

impl ValidityChild<ZigZag> for ZigZag {
    fn validity_child(array: &ZigZagData) -> &ArrayRef {
        array.encoded()
    }
}

#[cfg(test)]
mod test {
    use vortex_array::IntoArray;
    use vortex_array::ToCanonical;
    use vortex_array::scalar::Scalar;
    use vortex_buffer::buffer;

    use super::*;
    use crate::zigzag_encode;

    #[test]
    fn test_compute_statistics() -> VortexResult<()> {
        let array = buffer![1i32, -5i32, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            .into_array()
            .to_primitive();
        let zigzag = zigzag_encode(array.clone())?;

        assert_eq!(
            zigzag.statistics().compute_max::<i32>(),
            array.statistics().compute_max::<i32>()
        );
        assert_eq!(
            zigzag.statistics().compute_null_count(),
            array.statistics().compute_null_count()
        );
        assert_eq!(
            zigzag.statistics().compute_is_constant(),
            array.statistics().compute_is_constant()
        );

        let sliced = zigzag.slice(0..2).unwrap();
        let sliced = sliced.as_::<ZigZag>();
        assert_eq!(
            sliced.array().scalar_at(sliced.len() - 1).unwrap(),
            Scalar::from(-5i32)
        );

        assert_eq!(
            sliced.statistics().compute_min::<i32>(),
            array.statistics().compute_min::<i32>()
        );
        assert_eq!(
            sliced.statistics().compute_null_count(),
            array.statistics().compute_null_count()
        );
        assert_eq!(
            sliced.statistics().compute_is_constant(),
            array.statistics().compute_is_constant()
        );
        Ok(())
    }
}
