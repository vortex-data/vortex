#![allow(dead_code)]
#![allow(unused_variables)]

use std::any::Any;
use std::sync::Arc;

use vortex_buffer::{Buffer, ByteBuffer};
use vortex_dtype::{match_each_native_ptype, DType, NativePType, PType};
use vortex_error::{vortex_err, VortexResult};

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
        children: &[Array],
    ) -> VortexResult<Array> {
        self.0.load_array(dtype, metadata, buffers, children)
    }
}

pub trait EncodingImpl {
    fn load_array(
        &self,
        dtype: DType,
        metadata: Option<&[u8]>,
        buffers: &[ByteBuffer],
        children: &[Array],
    ) -> VortexResult<Array>;
}

//// ARRAY

/// API for a generic Vortex [`Array`].
pub struct Array(Arc<dyn ArrayImpl>);

impl Array {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn dtype(&self) -> &DType {
        self.0.dtype()
    }
}

/// Extra impls can add additional behavior to the Array API.
impl Array {
    pub fn add(self: Self, other: &Array) -> VortexResult<Array> {
        let result = self
            .0
            .add(other)?
            .ok_or_else(|| vortex_err!("Add not supported for this array type"))?;
        // Make some assertions about the result...
        Ok(result)
    }
}

pub trait ArrayImpl: 'static + Send + Sync + ComputeImpl {
    fn as_any(&self) -> &dyn Any;

    fn as_any_arc(self: Arc<Self>) -> Arc<dyn Any + Send + Sync>;

    fn len(&self) -> usize;

    fn dtype(&self) -> &DType;
}

pub trait ArrayImplExt: ArrayImpl {
    fn into_array(self) -> Array
    where
        Self: Sized,
    {
        Array(Arc::new(self))
    }
}

impl<A: ArrayImpl> ArrayImplExt for A {}

pub trait ComputeImpl {
    fn add(self: Arc<Self>, other: &Array) -> VortexResult<Option<Array>> {
        Ok(None)
    }
}

//// PRIMITIVE

pub struct PrimitiveEncoding;

impl EncodingImpl for PrimitiveEncoding {
    fn load_array(
        &self,
        dtype: DType,
        _metadata: Option<&[u8]>,
        buffers: &[ByteBuffer],
        _children: &[Array],
    ) -> VortexResult<Array> {
        let ptype = PType::try_from(&dtype)?;
        match_each_native_ptype!(ptype, |$P| {
            let buffer = Buffer::<$P>::from_byte_buffer(buffers[0].clone());
            Ok(PrimitiveArray::new(buffer, Validity::AllValid, StatsSet::default()).into_array())
        })
    }
}

#[derive(Clone)]
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

impl<T: NativePType> ArrayImpl for PrimitiveArray<T> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_arc(self: Arc<Self>) -> Arc<dyn Any + Send + Sync> {
        self
    }

    fn len(&self) -> usize {
        self.buffer.len()
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }
}

impl<T: NativePType> ComputeImpl for PrimitiveArray<T> {
    fn add(self: Arc<Self>, other: &Array) -> VortexResult<Option<Array>> {
        Ok(None)
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

        // Use `a` as an `Array`.
    }
}
