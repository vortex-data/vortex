#![allow(dead_code)]
#![allow(unused_variables)]

use std::any::{type_name, Any};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::BitAnd;
use std::sync::Arc;

use vortex_buffer::{Buffer, ByteBuffer};
use vortex_dtype::{match_each_native_ptype, DType, NativePType, PType};
use vortex_error::{vortex_bail, vortex_err, VortexResult};
use vortex_mask::Mask;

use crate::stats::StatsSet;
use crate::validity::Validity;

//// ENCODING

pub struct Encoding<E>(E);

pub type EncodingRef = Encoding<Arc<dyn EncodingImpl>>;

impl<E: EncodingImpl> Encoding<Arc<E>> {
    pub fn load_array(
        &self,
        dtype: DType,
        metadata: Option<&[u8]>,
        buffers: &[ByteBuffer],
        children: &[ArrayRef],
    ) -> VortexResult<ArrayRef> {
        self.0.load_array(dtype, metadata, buffers, children)
    }
}

pub trait EncodingImpl {
    fn load_array(
        &self,
        dtype: DType,
        metadata: Option<&[u8]>,
        buffers: &[ByteBuffer],
        children: &[ArrayRef],
    ) -> VortexResult<ArrayRef>;
}

//// ARRAY

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

    fn mask_validity(self: Self, mask: Mask) -> VortexResult<ArrayRef>
    where
        Self: Sized,
    {
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

pub trait ValidityVTable<Array: ?Sized> {
    /// Return the canonical validity mask for the array.
    fn validity_mask(&self, array: &Array) -> VortexResult<Mask>;

    /// Update the validity of the array by intersecting it with the given [`Mask`].
    fn mask_validity(&self, array: Arc<Array>, mask: Mask) -> VortexResult<ArrayRef>;
}

//// PRIMITIVE

pub struct PrimitiveEncoding;

impl EncodingImpl for PrimitiveEncoding {
    fn load_array(
        &self,
        dtype: DType,
        _metadata: Option<&[u8]>,
        buffers: &[ByteBuffer],
        _children: &[ArrayRef],
    ) -> VortexResult<ArrayRef> {
        let ptype = PType::try_from(&dtype)?;
        match_each_native_ptype!(ptype, |$P| {
            let buffer = Buffer::<$P>::from_byte_buffer(buffers[0].clone());
            Ok(Arc::new(PrimitiveArray::new(buffer, Validity::AllValid, StatsSet::default())))
        })
    }
}

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

/// But it must be implemented against a dyn Array.
impl<T: NativePType> ValidityVTable<dyn Array> for PrimitiveArrayVTable<T> {
    fn validity_mask(&self, array: &dyn Array) -> VortexResult<Mask> {
        self.validity_mask(
            array
                .as_any()
                .downcast_ref::<PrimitiveArray<T>>()
                .ok_or_else(|| {
                    vortex_err!(
                        "downcast to {} failed {:?}",
                        type_name::<PrimitiveArray<T>>(),
                        array
                    )
                })?,
        )
    }

    fn mask_validity(&self, array: Arc<dyn Array>, mask: Mask) -> VortexResult<ArrayRef> {
        self.mask_validity(
            array
                .as_any_arc()
                .downcast::<PrimitiveArray<T>>()
                .map_err(|_| {
                    vortex_err!("downcast to {} failed", type_name::<PrimitiveArray<T>>(),)
                })?,
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
        // We can also run compute functions that take the array by ownership.
        let a_masked = a
            .mask_validity(Mask::from_iter([true, false, true]))
            .unwrap();
        assert_eq!(a_masked.validity_mask().unwrap().true_count(), 2);

        // We can pass `a` into any function that takes `&dyn Array`.

        // We can convert `a` to `ArrayRef`, and do the same.
        let b: ArrayRef = Arc::new(a);
        assert_eq!(b.len(), 3);
        assert_eq!(b.validity_mask().unwrap(), Mask::new_true(3));

        // We can also run compute functions that take the array by ownership.
        let b = b
            .mask_validity(Mask::from_iter([true, false, true]))
            .unwrap();
        assert_eq!(b.validity_mask().unwrap().true_count(), 2);
    }
}
