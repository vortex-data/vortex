// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_panic};
use vortex_mask::Mask;
use vortex_vector::{Vector, VectorOps};

use crate::execution::{BatchKernelRef, BindCtx, DummyExecutionCtx, ExecutionCtx};
use crate::vtable::{OperatorVTable, VTable};
use crate::{Array, ArrayAdapter, ArrayRef};

/// Array functions as provided by the `OperatorVTable`.
///
/// Note: the public functions such as "execute" should move onto the main `Array` trait when
/// operators is stabilized. The other functions should remain on a `pub(crate)` trait.
pub trait ArrayOperator: 'static + Send + Sync {
    /// Execute the array's batch kernel with the given selection mask.
    ///
    /// # Panics
    ///
    /// If the mask length does not match the array length.
    /// If the array's implementation returns an invalid vector (wrong length, wrong type, etc).
    fn execute_batch(&self, selection: &Mask, ctx: &mut dyn ExecutionCtx) -> VortexResult<Vector>;

    /// Optimize the array by running the optimization rules.
    fn reduce_children(&self) -> VortexResult<Option<ArrayRef>>;

    /// Optimize the array by pushing down a parent array.
    fn reduce_parent(&self, parent: &ArrayRef, child_idx: usize) -> VortexResult<Option<ArrayRef>>;

    /// Bind the array to a batch kernel. This is an internal function
    fn bind(
        &self,
        selection: Option<&ArrayRef>,
        ctx: &mut dyn BindCtx,
    ) -> VortexResult<BatchKernelRef>;
}

impl ArrayOperator for Arc<dyn Array> {
    fn execute_batch(&self, selection: &Mask, ctx: &mut dyn ExecutionCtx) -> VortexResult<Vector> {
        self.as_ref().execute_batch(selection, ctx)
    }

    fn reduce_children(&self) -> VortexResult<Option<ArrayRef>> {
        self.as_ref().reduce_children()
    }

    fn reduce_parent(&self, parent: &ArrayRef, child_idx: usize) -> VortexResult<Option<ArrayRef>> {
        self.as_ref().reduce_parent(parent, child_idx)
    }

    fn bind(
        &self,
        selection: Option<&ArrayRef>,
        ctx: &mut dyn BindCtx,
    ) -> VortexResult<BatchKernelRef> {
        self.as_ref().bind(selection, ctx)
    }
}

impl<V: VTable> ArrayOperator for ArrayAdapter<V> {
    fn execute_batch(&self, selection: &Mask, ctx: &mut dyn ExecutionCtx) -> VortexResult<Vector> {
        let vector =
            <V::OperatorVTable as OperatorVTable<V>>::execute_batch(&self.0, selection, ctx)?;

        // Such a cheap check that we run it always. More expensive DType checks live in
        // debug_assertions.
        assert_eq!(
            vector.len(),
            selection.true_count(),
            "Batch execution returned vector of incorrect length"
        );

        #[cfg(debug_assertions)]
        {
            // Checks for correct type and nullability.
            if !vector_has_dtype(&vector, self.dtype()) {
                vortex_panic!(
                    "Returned vector {:?} does not match expected dtype {}",
                    vector,
                    self.dtype()
                );
            }
        }

        Ok(vector)
    }

    fn reduce_children(&self) -> VortexResult<Option<ArrayRef>> {
        <V::OperatorVTable as OperatorVTable<V>>::reduce_children(&self.0)
    }

    fn reduce_parent(&self, parent: &ArrayRef, child_idx: usize) -> VortexResult<Option<ArrayRef>> {
        <V::OperatorVTable as OperatorVTable<V>>::reduce_parent(&self.0, parent, child_idx)
    }

    fn bind(
        &self,
        selection: Option<&ArrayRef>,
        ctx: &mut dyn BindCtx,
    ) -> VortexResult<BatchKernelRef> {
        <V::OperatorVTable as OperatorVTable<V>>::bind(&self.0, selection, ctx)
    }
}

// TODO(ngates): create a smarter context in the future
impl BindCtx for () {
    fn bind(
        &mut self,
        array: &ArrayRef,
        selection: Option<&ArrayRef>,
    ) -> VortexResult<BatchKernelRef> {
        array.bind(selection, self)
    }
}

impl dyn Array + '_ {
    pub fn execute(&self) -> VortexResult<Vector> {
        self.execute_batch(&Mask::new_true(self.len()), &mut DummyExecutionCtx)
    }

    pub fn execute_with_selection(&self, mask: &Mask) -> VortexResult<Vector> {
        assert_eq!(self.len(), mask.len());
        self.execute_batch(mask, &mut DummyExecutionCtx)
    }
}

/// Returns true if the vector matches the provided data type.
///
/// This means that all values in the vector are contained within the domain of values described by
/// the logical data type.
///
/// Specifically, this means:
/// * `Vector::Null -> DType::Null`
/// * `Vector::Bool -> DType::Bool`
/// * `Vector::Primitive -> DType::Primitive` with matching PType
/// * `Vector::Decimal -> DType::Decimal` with matching precision/scale
/// * `Vector::String -> DType::Utf8`
/// * `Vector::Binary -> DType::Binary`
/// * `Vector::List -> DType::List` with matching element dtype
/// * `Vector::FixedSizeList -> DType::FixedSizeList` with matching elements dtype and element size
/// * `Vector::Struct -> DType::Struct` with matching field dtypes
/// * `* -> DType::Extension` where the vector must match the extension's storage dtype
///
/// Additionally, if the data type is non-nullable, the vector must contain no nulls.
fn vector_has_dtype(vector: &Vector, dtype: &DType) -> bool {
    if !dtype.is_nullable() && vector.validity().false_count() > 0 {
        // Non-nullable dtype cannot have nulls in the vector.
        return false;
    }

    // Note that we don't match a tuple here to make sure we have an exhaustive match that will
    // fail to compile if we ever add new DTypes.
    match dtype {
        DType::Null => {
            matches!(vector, Vector::Null(_))
        }
        DType::Bool(_) => {
            matches!(vector, Vector::Bool(_))
        }
        DType::Primitive(ptype, _) => match vector {
            Vector::Primitive(v) => ptype == &v.ptype(),
            _ => false,
        },
        DType::Decimal(dec_type, _) => match vector {
            Vector::Decimal(v) => {
                dec_type.precision() == v.precision() && dec_type.scale() == v.scale()
            }
            _ => false,
        },
        DType::Utf8(_) => {
            matches!(vector, Vector::String(_))
        }
        DType::Binary(_) => {
            matches!(vector, Vector::Binary(_))
        }
        DType::List(elements, _) => match vector {
            Vector::List(v) => vector_has_dtype(v.elements(), elements.as_ref()),
            _ => false,
        },
        DType::FixedSizeList(elements, size, _) => match vector {
            Vector::FixedSizeList(v) => {
                v.element_size() == *size && vector_has_dtype(v.elements(), elements.as_ref())
            }
            _ => false,
        },
        DType::Struct(fields, _) => match vector {
            Vector::Struct(v) => {
                if fields.nfields() != v.fields().len() {
                    return false;
                }
                for (field_dtype, field_vector) in fields.fields().zip(v.fields().iter()) {
                    if !vector_has_dtype(field_vector, &field_dtype) {
                        return false;
                    }
                }
                true
            }
            _ => false,
        },
        DType::Extension(ext_dtype) => {
            // For extension types, we check the storage type.
            vector_has_dtype(vector, ext_dtype.storage_dtype())
        }
    }
}
