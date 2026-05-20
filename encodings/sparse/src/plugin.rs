// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A custom [`ArrayPlugin`] that lets you load in and deserialize a `Sparse` array as a
//! `PatchedArray` that wraps a constant fill array.
//!
//! A `Sparse` array is logically a set of patches applied on top of a constant fill value, which
//! is exactly what a `Patched` array over a [`ConstantArray`] represents. This plugin externalizes
//! that representation on deserialize when the array is primitive with non-null patches, which is
//! the subset that `Patched` can represent. All other sparse arrays are returned unchanged.

use vortex_array::Array;
use vortex_array::ArrayId;
use vortex_array::ArrayPlugin;
use vortex_array::ArrayRef;
use vortex_array::ArrayVTable;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Patched;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::serde::ArrayChildren;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::Sparse;
use crate::SparseExt;

/// Custom deserialization plugin that converts a primitive `Sparse` array into a `PatchedArray`
/// holding a [`ConstantArray`] fill.
#[derive(Debug, Clone)]
pub struct SparsePatchedPlugin;

impl ArrayPlugin for SparsePatchedPlugin {
    fn id(&self) -> ArrayId {
        // We reuse the existing `Sparse` ID so that we can take over its deserialization pathway.
        ArrayVTable::id(&Sparse)
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        // Delegate to the Sparse VTable for serialization.
        Sparse.serialize(array, session)
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        let sparse = Array::<Sparse>::try_from_parts(ArrayVTable::deserialize(
            &Sparse, dtype, len, metadata, buffers, children, session,
        )?)
        .map_err(|_| vortex_err!("Sparse plugin should only deserialize vortex.sparse"))?;

        // `Patched` can only represent primitive inners with non-null patch values, so anything
        // else (bool, varbin, struct, fixed-size-list, nullable patches) stays a Sparse array.
        if !dtype.is_primitive() {
            return Ok(sparse.into_array());
        }

        let patches = sparse.patches();
        let mut ctx = session.create_execution_ctx();
        if !patches.values().all_valid(&mut ctx)? {
            return Ok(sparse.into_array());
        }

        let fill = ConstantArray::new(sparse.fill_scalar().clone(), len).into_array();
        let patched = Patched::from_array_and_patches(fill, &patches, &mut ctx)?;

        Ok(patched.into_array())
    }

    fn is_supported_encoding(&self, id: &ArrayId) -> bool {
        id == ArrayVTable::id(&Sparse) || id == ArrayVTable::id(&Patched)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::ArrayPlugin;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PatchedArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::patched::PatchedArraySlotsExt;
    use vortex_array::buffer::BufferHandle;
    use vortex_array::patches::Patches;
    use vortex_array::scalar::Scalar;
    use vortex_array::session::ArraySession;
    use vortex_array::session::ArraySessionExt;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_session::VortexSession;

    use super::SparsePatchedPlugin;
    use crate::Sparse;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty().with::<ArraySession>();
        session.arrays().register(SparsePatchedPlugin);
        session
    });

    fn primitive_sparse() -> VortexResult<crate::SparseArray> {
        let patches = Patches::new(
            10,
            0,
            PrimitiveArray::from_iter([1u32, 3, 7]).into_array(),
            PrimitiveArray::from_iter([10u32, 30, 70]).into_array(),
            None,
        )?;
        Sparse::try_new_from_patches(
            patches,
            Scalar::primitive(0u32, vortex_array::dtype::Nullability::NonNullable),
        )
    }

    fn round_trip(array: &vortex_array::ArrayRef) -> VortexResult<vortex_array::ArrayRef> {
        let metadata = SESSION.array_serialize(array)?.unwrap();
        let children = array.children();
        let buffers = array
            .buffers()
            .into_iter()
            .map(BufferHandle::new_host)
            .collect::<Vec<_>>();

        SparsePatchedPlugin.deserialize(
            array.dtype(),
            array.len(),
            &metadata,
            &buffers,
            &children,
            &SESSION,
        )
    }

    #[test]
    fn primitive_sparse_becomes_patched() -> VortexResult<()> {
        let sparse = primitive_sparse()?.into_array();
        let deserialized = round_trip(&sparse)?;

        let patched: PatchedArray = deserialized
            .try_downcast()
            .map_err(|a| vortex_err!("Expected Patched, got {}", a.encoding_id()))?;

        // The inner is the constant fill.
        assert!(patched.inner().as_constant().is_some());

        // The decoded values must match the original sparse array.
        let mut ctx = SESSION.create_execution_ctx();
        let expected = sparse
            .execute::<PrimitiveArray>(&mut ctx)?
            .into_buffer::<u32>();
        let actual = patched
            .into_array()
            .execute::<PrimitiveArray>(&mut ctx)?
            .into_buffer::<u32>();
        assert_eq!(expected, actual);

        Ok(())
    }

    #[test]
    fn non_primitive_sparse_stays_sparse() -> VortexResult<()> {
        use vortex_array::arrays::BoolArray;

        let patches = Patches::new(
            5,
            0,
            PrimitiveArray::from_iter([1u32, 3]).into_array(),
            BoolArray::from_iter([true, false]).into_array(),
            None,
        )?;
        let sparse = Sparse::try_new_from_patches(
            patches,
            Scalar::bool(false, vortex_array::dtype::Nullability::NonNullable),
        )?
        .into_array();

        let deserialized = round_trip(&sparse)?;
        assert!(
            deserialized.is::<Sparse>(),
            "non-primitive sparse should stay sparse, got {}",
            deserialized.encoding_id()
        );

        Ok(())
    }
}
