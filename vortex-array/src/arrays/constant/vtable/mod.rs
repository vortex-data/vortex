// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::hash::Hash;

use vortex_buffer::ByteBufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::Precision;
use crate::arrays::ConstantArray;
use crate::arrays::constant::compute::rules::PARENT_RULES;
use crate::arrays::constant::vtable::canonical::constant_canonicalize;
use crate::buffer::BufferHandle;
use crate::builders::ArrayBuilder;
use crate::builders::PrimitiveBuilder;
use crate::dtype::DType;
use crate::match_each_native_ptype;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;
use crate::serde::ArrayChildren;
use crate::stats::StatsSetRef;
use crate::vtable;
use crate::vtable::ArrayId;
use crate::vtable::VTable;
pub(crate) mod canonical;
mod operations;
mod validity;

vtable!(Constant);

#[derive(Debug)]
pub struct ConstantVTable;

impl ConstantVTable {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.constant");
}

impl VTable for ConstantVTable {
    type Array = ConstantArray;

    type Metadata = Scalar;
    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(_array: &Self::Array) -> ArrayId {
        Self::ID
    }

    fn len(array: &ConstantArray) -> usize {
        array.len
    }

    fn dtype(array: &ConstantArray) -> &DType {
        array.scalar.dtype()
    }

    fn stats(array: &ConstantArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(
        array: &ConstantArray,
        state: &mut H,
        _precision: Precision,
    ) {
        array.scalar.hash(state);
        array.len.hash(state);
    }

    fn array_eq(array: &ConstantArray, other: &ConstantArray, _precision: Precision) -> bool {
        array.scalar == other.scalar && array.len == other.len
    }

    fn nbuffers(_array: &ConstantArray) -> usize {
        1
    }

    fn buffer(array: &ConstantArray, idx: usize) -> BufferHandle {
        match idx {
            0 => BufferHandle::new_host(
                ScalarValue::to_proto_bytes::<ByteBufferMut>(array.scalar.value()).freeze(),
            ),
            _ => vortex_panic!("ConstantArray buffer index {idx} out of bounds"),
        }
    }

    fn buffer_name(_array: &ConstantArray, idx: usize) -> Option<String> {
        match idx {
            0 => Some("scalar".to_string()),
            _ => None,
        }
    }

    fn nchildren(_array: &ConstantArray) -> usize {
        0
    }

    fn child(_array: &ConstantArray, idx: usize) -> ArrayRef {
        vortex_panic!("ConstantArray child index {idx} out of bounds")
    }

    fn child_name(_array: &ConstantArray, idx: usize) -> String {
        vortex_panic!("ConstantArray child_name index {idx} out of bounds")
    }

    fn metadata(array: &ConstantArray) -> VortexResult<Self::Metadata> {
        Ok(array.scalar().clone())
    }

    fn serialize(_metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        // HACK: Because the scalar is stored in the buffers, we do not need to serialize the
        // metadata at all.
        Ok(Some(vec![]))
    }

    fn deserialize(
        _bytes: &[u8],
        dtype: &DType,
        _len: usize,
        buffers: &[BufferHandle],
        _session: &VortexSession,
    ) -> VortexResult<Self::Metadata> {
        vortex_ensure!(
            buffers.len() == 1,
            "Expected 1 buffer, got {}",
            buffers.len()
        );

        let buffer = buffers[0].clone().try_to_host_sync()?;
        let bytes: &[u8] = buffer.as_ref();

        let scalar_value = ScalarValue::from_proto_bytes(bytes, dtype)?;
        let scalar = Scalar::try_new(dtype.clone(), scalar_value)?;

        Ok(scalar)
    }

    fn build(
        _dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        _buffers: &[BufferHandle],
        _children: &dyn ArrayChildren,
    ) -> VortexResult<ConstantArray> {
        Ok(ConstantArray::new(metadata.clone(), len))
    }

    fn with_children(_array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        vortex_ensure!(
            children.is_empty(),
            "ConstantArray has no children, got {}",
            children.len()
        );
        Ok(())
    }

    fn reduce_parent(
        array: &Self::Array,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }

    fn execute(array: &Self::Array, _ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        Ok(constant_canonicalize(array)?.into_array())
    }

    fn append_to_builder(
        array: &ConstantArray,
        builder: &mut dyn ArrayBuilder,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        match array.dtype() {
            DType::Primitive(ptype, _) => {
                match_each_native_ptype!(ptype, |P| {
                    let pbuilder = builder
                        .as_any_mut()
                        .downcast_mut::<PrimitiveBuilder<P>>()
                        .vortex_expect("Expected PrimitiveBuilder for primitive ConstantArray");
                    if array.scalar().is_null() {
                        // SAFETY: append_nulls_unchecked requires a nullable builder.
                        // A null scalar implies a nullable dtype, so the builder created
                        // for this array's dtype is necessarily nullable.
                        unsafe { pbuilder.append_nulls_unchecked(array.len()) }
                    } else {
                        let value = P::try_from(array.scalar())
                            .vortex_expect("Couldn't unwrap constant scalar to primitive");
                        pbuilder.append_value_n(value, array.len());
                    }
                });
                Ok(())
            }
            _ => {
                let canonical = constant_canonicalize(array)?.into_array();
                builder.extend_from_array(&canonical);
                Ok(())
            }
        }
    }
}
