// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hasher;

use vortex_array::Array;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::Precision;
use vortex_array::TypedArrayRef;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_array::serde::ArrayChildren;
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
use vortex_session::registry::CachedId;
use zigzag::ZigZag as ExternalZigZag;

use crate::compute::ZigZagEncoded;
use crate::kernel::PARENT_KERNELS;
use crate::rules::RULES;
use crate::zigzag_decode;

/// A [`ZigZag`]-encoded Vortex array.
pub type ZigZagArray = Array<ZigZag>;

impl VTable for ZigZag {
    type ArrayData = ZigZagData;

    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.zigzag");
        *ID
    }

    fn validate(
        &self,
        _data: &Self::ArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let encoded = slots[ENCODED_SLOT]
            .as_ref()
            .vortex_expect("ZigZagArray encoded slot");
        let expected_dtype = ZigZagData::dtype_from_encoded_dtype(encoded.dtype())?;
        vortex_ensure!(
            dtype == &expected_dtype,
            "expected dtype {expected_dtype}, got {dtype}"
        );
        vortex_ensure!(
            encoded.len() == len,
            "expected len {len}, got {}",
            encoded.len()
        );
        Ok(())
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

    fn serialize(
        _array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        if !metadata.is_empty() {
            vortex_bail!(
                "ZigZagArray expects empty metadata, got {} bytes",
                metadata.len()
            );
        }
        if children.len() != 1 {
            vortex_bail!("Expected 1 child, got {}", children.len());
        }

        let ptype = PType::try_from(dtype)?;
        let encoded_type = DType::Primitive(ptype.to_unsigned(), dtype.nullability());

        let encoded = children.get(0, &encoded_type, len)?;
        let slots = vec![Some(encoded.clone())];
        let data = ZigZagData::try_new(encoded.dtype())?;
        Ok(ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
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

impl ArrayHash for ZigZagData {
    fn array_hash<H: Hasher>(&self, _state: &mut H, _precision: Precision) {}
}

impl ArrayEq for ZigZagData {
    fn array_eq(&self, _other: &Self, _precision: Precision) -> bool {
        true
    }
}

/// The zigzag-encoded values (signed integers mapped to unsigned).
pub(super) const ENCODED_SLOT: usize = 0;
pub(super) const NUM_SLOTS: usize = 1;
pub(super) const SLOT_NAMES: [&str; NUM_SLOTS] = ["encoded"];

#[derive(Clone, Debug)]
pub struct ZigZagData {}

impl Display for ZigZagData {
    fn fmt(&self, _f: &mut Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

pub trait ZigZagArrayExt: TypedArrayRef<ZigZag> {
    fn encoded(&self) -> &ArrayRef {
        self.as_ref().slots()[ENCODED_SLOT]
            .as_ref()
            .vortex_expect("ZigZagArray encoded slot")
    }

    fn ptype(&self) -> PType {
        PType::try_from(self.encoded().dtype())
            .vortex_expect("ZigZagArray encoded dtype")
            .to_signed()
    }
}

impl<T: TypedArrayRef<ZigZag>> ZigZagArrayExt for T {}

#[derive(Clone, Debug)]
pub struct ZigZag;

impl ZigZag {
    /// Construct a new [`ZigZagArray`] from an encoded unsigned integer array.
    pub fn try_new(encoded: ArrayRef) -> VortexResult<ZigZagArray> {
        let dtype = ZigZagData::dtype_from_encoded_dtype(encoded.dtype())?;
        let len = encoded.len();
        let slots = vec![Some(encoded.clone())];
        let data = ZigZagData::try_new(encoded.dtype())?;
        Ok(unsafe {
            Array::from_parts_unchecked(ArrayParts::new(ZigZag, dtype, len, data).with_slots(slots))
        })
    }
}

impl ZigZagData {
    fn dtype_from_encoded_dtype(encoded_dtype: &DType) -> VortexResult<DType> {
        Ok(DType::from(PType::try_from(encoded_dtype)?.to_signed())
            .with_nullability(encoded_dtype.nullability()))
    }

    pub fn new() -> Self {
        Self {}
    }

    pub fn try_new(encoded_dtype: &DType) -> VortexResult<Self> {
        if !encoded_dtype.is_unsigned_int() {
            vortex_bail!(MismatchedTypes: "unsigned int", encoded_dtype);
        }

        Self::dtype_from_encoded_dtype(encoded_dtype)?;

        Ok(Self {})
    }
}

impl Default for ZigZagData {
    fn default() -> Self {
        Self::new()
    }
}

impl OperationsVTable<ZigZag> for ZigZag {
    fn scalar_at(
        array: ArrayView<'_, ZigZag>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let scalar = array.encoded().execute_scalar(index, ctx)?;
        if scalar.is_null() {
            return scalar.primitive_reinterpret_cast(ZigZagArrayExt::ptype(&array));
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
    fn validity_child(array: ArrayView<'_, ZigZag>) -> ArrayRef {
        array.encoded().clone()
    }
}

#[cfg(test)]
mod test {
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::scalar::Scalar;
    use vortex_buffer::buffer;

    use super::*;
    use crate::zigzag_encode;

    #[test]
    fn test_compute_statistics() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let array = buffer![1i32, -5i32, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            .into_array()
            .execute::<PrimitiveArray>(&mut ctx)?;
        let zigzag = zigzag_encode(array.as_view())?;

        assert_eq!(
            zigzag.statistics().compute_max::<i32>(&mut ctx),
            array.statistics().compute_max::<i32>(&mut ctx)
        );
        assert_eq!(
            zigzag.statistics().compute_null_count(&mut ctx),
            array.statistics().compute_null_count(&mut ctx)
        );
        assert_eq!(
            zigzag.statistics().compute_is_constant(&mut ctx),
            array.statistics().compute_is_constant(&mut ctx)
        );

        let sliced = zigzag.slice(0..2).unwrap();
        let sliced = sliced.as_::<ZigZag>();
        assert_eq!(
            sliced
                .array()
                .execute_scalar(sliced.len() - 1, &mut ctx,)
                .unwrap(),
            Scalar::from(-5i32)
        );

        assert_eq!(
            sliced.statistics().compute_min::<i32>(&mut ctx),
            array.statistics().compute_min::<i32>(&mut ctx)
        );
        assert_eq!(
            sliced.statistics().compute_null_count(&mut ctx),
            array.statistics().compute_null_count(&mut ctx)
        );
        assert_eq!(
            sliced.statistics().compute_is_constant(&mut ctx),
            array.statistics().compute_is_constant(&mut ctx)
        );
        Ok(())
    }
}
