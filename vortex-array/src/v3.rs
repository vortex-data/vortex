#![allow(dead_code)]
use std::any::Any;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::BitAnd;
use std::sync::Arc;

use vortex_buffer::Buffer;
use vortex_dtype::{DType, NativePType};
use vortex_error::{vortex_bail, vortex_err, VortexResult};
use vortex_mask::Mask;

use crate::stats::StatsSet;
use crate::validity::Validity;

//// ARRAY

// NOTE(ngates): explicitly do not require Clone for Array.
pub trait Array: 'static + Send + Sync + Debug {
    fn as_any(&self) -> &dyn Any;

    fn as_any_arc(self: Arc<Self>) -> Arc<dyn Any + Send + Sync>;

    /// Returns the array's VTable.
    fn vtable(&self) -> Arc<dyn ArrayVTable>;

    /// Returns the length of the array.
    fn len(&self) -> usize;

    /// Returns the DType of the array.
    fn dtype(&self) -> &DType;
}

pub type ArrayRef = Arc<dyn Array>;

/// An extension trait that implements much of the logic of the Array API.
pub trait ArrayExt: Array {
    fn validity_mask(&self) -> VortexResult<Mask>
    where
        Self: Sized,
    {
        self.vtable().validity_mask(self)
    }

    /// For owned functions, we must implement both on sized Self...
    fn mask_validity(self: Arc<Self>, mask: Mask) -> VortexResult<ArrayRef>
    where
        Self: Sized,
    {
        self.vtable().mask_validity(self, mask)
    }
}

/// ...and on the dyn Array trait.
impl dyn Array + '_ {
    fn mask_validity(self: Arc<Self>, mask: Mask) -> VortexResult<ArrayRef> {
        self.vtable().mask_validity(self, mask)
    }
}

impl<A: Array + ?Sized> ArrayExt for A {}

impl Array for Arc<dyn Array> {
    fn as_any(&self) -> &dyn Any {
        self.as_ref().as_any()
    }

    fn as_any_arc(self: Arc<Self>) -> Arc<dyn Any + Send + Sync> {
        self
    }

    fn vtable(&self) -> Arc<dyn ArrayVTable> {
        self.as_ref().vtable()
    }

    fn len(&self) -> usize {
        self.as_ref().len()
    }

    fn dtype(&self) -> &DType {
        self.as_ref().dtype()
    }
}

/// Implementation of the Array API.
pub trait ArrayVTable: ValidityVTable<dyn Array> {}

pub trait ValidityVTable<A: ?Sized> {
    /// Return the canonical validity mask for the array.
    fn validity_mask(&self, array: &A) -> VortexResult<Mask>;

    /// Update the validity of the array by intersecting it with the given [`Mask`].
    fn mask_validity(&self, array: Arc<A>, mask: Mask) -> VortexResult<ArrayRef>;
}

//// PRIMITIVE

#[derive(Clone, Debug)]
pub struct PrimitiveArray<T: NativePType> {
    dtype: DType,
    buffer: Buffer<T>,
    validity: Validity,
    stats: StatsSet,
}

impl<T: NativePType> PrimitiveArray<T> {
    pub fn new(buffer: Buffer<T>, validity: Validity, stats: StatsSet) -> Self {
        if let Validity::Array(validity) = &validity {
            assert_eq!(buffer.len(), validity.len());
        }
        Self {
            dtype: DType::Primitive(T::PTYPE, validity.nullability()),
            buffer,
            validity,
            stats,
        }
    }
}

impl<T: NativePType> Array for PrimitiveArray<T> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_arc(self: Arc<Self>) -> Arc<dyn Any + Send + Sync> {
        self
    }

    fn vtable(&self) -> Arc<dyn ArrayVTable> {
        Arc::new(PrimitiveArrayVTable::<T>(PhantomData))
    }

    fn len(&self) -> usize {
        self.buffer.len()
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }
}

pub struct PrimitiveArrayVTable<T>(PhantomData<T>);

impl<T: NativePType> VTableDowncast for PrimitiveArrayVTable<T> {
    type Array = PrimitiveArray<T>;
}

impl<T: NativePType> ArrayVTable for PrimitiveArrayVTable<T> {}

/// We implement the validity vtable against the specific array type.
impl<T: NativePType> ValidityVTable<PrimitiveArray<T>> for PrimitiveArrayVTable<T> {
    fn validity_mask(&self, array: &PrimitiveArray<T>) -> VortexResult<Mask> {
        array.validity.to_logical(array.buffer.len())
    }

    fn mask_validity(&self, array: Arc<PrimitiveArray<T>>, mask: Mask) -> VortexResult<ArrayRef> {
        let mut array = Arc::unwrap_or_clone(array);
        match array.validity {
            Validity::NonNullable => {
                vortex_bail!("Cannot mask validity of a non-nullable array")
            }
            Validity::AllValid => {
                array.validity = Validity::from_mask(mask, array.dtype.nullability());
            }
            Validity::AllInvalid => {
                // Nothing to do, everything is invalid already.
            }
            Validity::Array(a) => {
                let validity = Mask::try_from(a)?.bitand(&mask);
                array.validity = Validity::from_mask(validity, array.dtype.nullability());
            }
        }
        Ok(Arc::new(array))
    }
}

trait VTableDowncast {
    type Array: Array;
}

impl<V> ValidityVTable<dyn Array> for V
where
    V: VTableDowncast,
    V: ValidityVTable<V::Array>,
{
    fn validity_mask(&self, array: &dyn Array) -> VortexResult<Mask> {
        self.validity_mask(
            array
                .as_any()
                .downcast_ref::<V::Array>()
                .ok_or_else(|| vortex_err!("downcast failed",))?,
        )
    }

    fn mask_validity(&self, array: ArrayRef, mask: Mask) -> VortexResult<ArrayRef> {
        self.mask_validity(
            array
                .as_any_arc()
                .downcast::<V::Array>()
                .map_err(|_| vortex_err!("downcast failed"))?,
            mask,
        )
    }
}

#[cfg(test)]
mod test {
    use vortex_buffer::*;

    use super::*;

    #[test]
    fn test_arrays() {
        // Create a typed primitive array.
        let a =
            PrimitiveArray::<i32>::new(buffer![1, 2, 3], Validity::AllValid, StatsSet::default());

        // We can use `a` as an `Array`, including calling functions on `ArrayExt`.
        assert_eq!(a.len(), 3);
        assert_eq!(a.validity_mask().unwrap(), Mask::new_true(3));

        // If we Arc the array, we can call functions that take the array by ownership.
        // This is because we don't require that Array implements clone, therefore we cannot
        // automatically unwrap_or_clone an array from an `Arc<dyn Array>`.
        let a: Arc<PrimitiveArray<i32>> = Arc::new(a);
        let a_masked = a
            .clone()
            .mask_validity(Mask::from_iter([true, false, true]))
            .unwrap();
        assert_eq!(a_masked.validity_mask().unwrap().true_count(), 2);

        // We can type-erase the array into an `Arc<dyn Array>` (an `ArrayRef`), and do the same.
        let b: ArrayRef = a as _;
        assert_eq!(b.len(), 3);
        assert_eq!(b.validity_mask().unwrap(), Mask::new_true(3));
        let b_masked = b
            .mask_validity(Mask::from_iter([true, false, true]))
            .unwrap();
        assert_eq!(b_masked.validity_mask().unwrap().true_count(), 2);
    }
}
