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

/// The [`Array`] trait is the main interface for working with arrays in Vortex.
///
/// It is implemented by all array types, and provides a common API.
///
/// In order to control the API surface of the [`Array`] trait, we use an explicit VTable pattern
/// that allows us to wrap up API calls with common functionality such as input and output
/// assertions. See the [`ArrayExt`] trait for an example.
pub trait Array: 'static + Send + Sync + Debug {
    fn as_any(&self) -> &dyn Any;

    fn as_any_arc(self: Arc<Self>) -> Arc<dyn Any + Send + Sync>;

    fn into_array(self) -> ArrayRef
    where
        Self: Sized;

    /// Returns the array's VTable.
    fn vtable(&self) -> Arc<dyn ArrayVTable>;

    /// Returns the length of the array.
    fn len(&self) -> usize;

    /// Returns the DType of the array.
    fn dtype(&self) -> &DType;
}

/// As per convention in the Arrow / DataFusion codebase, we use a `_Ref` type alias for
/// the `Arc<dyn Array>` type.
pub type ArrayRef = Arc<dyn Array>;

/// Much of the API surface of the [`Array`] trait is implemented on extension traits.
/// The [`ArrayExt`] trait is an example of such a trait, although in practice I imagine these
/// are spread around the code base with names like [`ArrayValidity`].
///
/// These traits call into the [`ArrayVTable`] to dispatch function calls, and can apply common
/// functionality such as input and output assertions.
pub trait ArrayExt: Array {
    fn validity_mask(&self) -> VortexResult<Mask>
    where
        Self: Sized,
    {
        let mask = self.vtable().validity_mask(self)?;
        // Output assertions panic, since it's an implementation bug for the array.
        assert_eq!(
            mask.len(),
            self.len(),
            "Array {:?} returned validity mask with invalid length",
            self
        );
        Ok(mask)
    }

    fn mask_validity(self: Arc<Self>, mask: Mask) -> VortexResult<ArrayRef>
    where
        Self: Sized,
    {
        (self as ArrayRef).mask_validity(mask)
    }
}

impl<A: Array + ?Sized> ArrayExt for A {}

/// For functions that take the array by ownership, we implement them again on the [`Array`] trait.
/// The functions in the [`ArrayExt`] trait can be called on concrete arrays, e.g.
/// `Arc<PrimitiveArray<i32>>`, whereas the functions on `dyn Array + '_` can be called on
/// an `Arc<dyn Array>`.
impl dyn Array + '_ {
    fn mask_validity(self: Arc<Self>, mask: Mask) -> VortexResult<ArrayRef> {
        if mask.len() != self.len() {
            // Input assertions return a VortexResult, since it's a caller error.
            vortex_bail!(
                "Mask length {} does not match array length {}",
                mask.len(),
                self.len()
            );
        }
        self.vtable().mask_validity(self, mask)
    }
}

impl Array for Arc<dyn Array> {
    fn as_any(&self) -> &dyn Any {
        self.as_ref().as_any()
    }

    fn as_any_arc(self: Arc<Self>) -> Arc<dyn Any + Send + Sync> {
        self
    }

    fn into_array(self: Self) -> ArrayRef
    where
        Self: Sized,
    {
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

    fn into_array(self: Self) -> ArrayRef
    where
        Self: Sized,
    {
        Arc::new(self)
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
