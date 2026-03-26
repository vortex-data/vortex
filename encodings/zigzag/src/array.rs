// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;
use std::sync::Arc;

use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayRef;
use vortex_array::DynArray;
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
use vortex_array::stats::StatsSetRef;
use vortex_array::vtable;
use vortex_array::vtable::Array;
use vortex_array::vtable::ArrayId;
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

vtable!(ZigZag);

impl VTable for ZigZag {
    type Array = ZigZagArray;

    type Metadata = EmptyMetadata;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    fn vtable(_array: &Self::Array) -> &Self {
        &ZigZag
    }

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn len(array: &ZigZagArray) -> usize {
        array.encoded.len()
    }

    fn dtype(array: &ZigZagArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &ZigZagArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(array: &ZigZagArray, state: &mut H, precision: Precision) {
        array.dtype.hash(state);
        array.encoded.array_hash(state, precision);
    }

    fn array_eq(array: &ZigZagArray, other: &ZigZagArray, precision: Precision) -> bool {
        array.dtype == other.dtype && array.encoded.array_eq(&other.encoded, precision)
    }

    fn nbuffers(_array: &ZigZagArray) -> usize {
        0
    }

    fn buffer(_array: &ZigZagArray, idx: usize) -> BufferHandle {
        vortex_panic!("ZigZagArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: &ZigZagArray, idx: usize) -> Option<String> {
        vortex_panic!("ZigZagArray buffer_name index {idx} out of bounds")
    }

    fn nchildren(_array: &ZigZagArray) -> usize {
        1
    }

    fn child(array: &ZigZagArray, idx: usize) -> ArrayRef {
        match idx {
            0 => array.encoded().clone(),
            _ => vortex_panic!("ZigZagArray child index {idx} out of bounds"),
        }
    }

    fn child_name(_array: &ZigZagArray, idx: usize) -> String {
        match idx {
            0 => "encoded".to_string(),
            _ => vortex_panic!("ZigZagArray child_name index {idx} out of bounds"),
        }
    }

    fn metadata(_array: &ZigZagArray) -> VortexResult<Self::Metadata> {
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
    ) -> VortexResult<ZigZagArray> {
        if children.len() != 1 {
            vortex_bail!("Expected 1 child, got {}", children.len());
        }

        let ptype = PType::try_from(dtype)?;
        let encoded_type = DType::Primitive(ptype.to_unsigned(), dtype.nullability());

        let encoded = children.get(0, &encoded_type, len)?;
        ZigZagArray::try_new(encoded)
    }

    fn with_children(array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        vortex_ensure!(
            children.len() == 1,
            "ZigZagArray expects exactly 1 child (encoded), got {}",
            children.len()
        );
        array.encoded = children.into_iter().next().vortex_expect("checked");
        Ok(())
    }

    fn execute(array: Arc<Array<Self>>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(
            zigzag_decode(array.encoded().clone().execute(ctx)?).into_array(),
        ))
    }

    fn reduce_parent(
        array: &Array<Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        RULES.evaluate(array, parent, child_idx)
    }

    fn execute_parent(
        array: &Array<Self>,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_KERNELS.execute(array, parent, child_idx, ctx)
    }
}

#[derive(Clone, Debug)]
pub struct ZigZagArray {
    dtype: DType,
    encoded: ArrayRef,
    stats_set: ArrayStats,
}

#[derive(Clone, Debug)]
pub struct ZigZag;

impl ZigZag {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.zigzag");
}

impl ZigZagArray {
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
            encoded,
            stats_set: Default::default(),
        })
    }

    pub fn ptype(&self) -> PType {
        self.dtype().as_ptype()
    }

    pub fn encoded(&self) -> &ArrayRef {
        &self.encoded
    }
}

impl OperationsVTable<ZigZag> for ZigZag {
    fn scalar_at(
        array: &ZigZagArray,
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
    fn validity_child(array: &ZigZagArray) -> &ArrayRef {
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
            sliced.scalar_at(sliced.len() - 1).unwrap(),
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
