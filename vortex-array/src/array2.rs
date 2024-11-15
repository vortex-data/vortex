#![allow(dead_code)]
use std::marker::PhantomData;
use std::sync::{Arc, RwLock};

use vortex_buffer::Buffer;
use vortex_dtype::DType;
use vortex_error::{vortex_bail, VortexError, VortexResult};

use crate::stats::StatsSet;
use crate::{ArrayMetadata, Canonical, Context};

// Type-erased array.
pub struct ArrayData {
    encoding: &'static EncodingVTable,
    dtype: Arc<DType>,
    len: usize,
    inner: InnerArrayData,
}
enum InnerArrayData {
    Owned(OwnedArrayData),
    Viewed(ViewedArrayData),
}
struct OwnedArrayData {
    metadata: Arc<dyn ArrayMetadata>,
    buffer: Option<Buffer>,
    children: Arc<[ArrayData]>,
    stats_map: Arc<RwLock<StatsSet>>,
}
struct ViewedArrayData {
    flatbuffer: Buffer,
    flatbuffer_loc: usize,
    buffers: Arc<[Buffer]>,
    // TODO(ngates): move this onto ArrayData?
    ctx: Arc<Context>,
}

// The set of generic array functionality.
// Again, it's nice to split this into multiple traits to allow implementation to be split
// across multiple files, and for us to auto-derive some of the implementations.
pub trait ArrayImpl {
    fn as_array_data(&self) -> &ArrayData;
    fn into_array_data(self) -> ArrayData;
    fn into_canonical(self) -> VortexResult<Canonical>;
    fn is_valid(&self, index: usize) -> VortexResult<bool>;
}

// Array implementation is supported for the type-erased ArrayData by dispatching via the encoding
// VTable.
impl ArrayImpl for ArrayData {
    fn as_array_data(&self) -> &ArrayData {
        self
    }

    fn into_array_data(self) -> ArrayData {
        self
    }

    fn into_canonical(self) -> VortexResult<Canonical> {
        (self.encoding.into_canonical)(self)
    }

    fn is_valid(&self, index: usize) -> VortexResult<bool> {
        (self.encoding.is_valid)(self, index)
    }
}

pub struct EncodingVTable {
    id: &'static str,
    into_canonical: &'static dyn Fn(ArrayData) -> VortexResult<Canonical>,
    is_valid: &'static dyn Fn(&ArrayData, usize) -> VortexResult<bool>,
}

// A Vortex array is a typed wrapper around an ArrayData.
// This allows us to use optimized dispatch for ArrayImpl functions when the Array type is known
// to the compiler, and fall back to dynamic virtual function calls via the encoding when the
// type-erased ArrayData is used.
pub struct Array<E> {
    data: ArrayData,
    _marker: PhantomData<E>,
}

impl<E: Encoding> TryFrom<&ArrayData> for &Array<E> {
    type Error = VortexError;

    fn try_from(data: &ArrayData) -> Result<Self, Self::Error> {
        if data.encoding.id != E::ID {
            vortex_bail!("Mismatched encoding")
        }
        // This cast is permitted since we guarantee that the layout of Array == ArrayData.
        Ok(unsafe { *(data as *const ArrayData as *const Self) })
    }
}

impl<E: Encoding> TryFrom<ArrayData> for Array<E> {
    type Error = VortexError;

    fn try_from(data: ArrayData) -> Result<Self, Self::Error> {
        if data.encoding.id != E::ID {
            vortex_bail!("Mismatched encoding")
        }
        Ok(Array {
            data,
            _marker: PhantomData,
        })
    }
}

pub trait Encoding {
    const ID: &'static str;
}

///// Examples

struct BoolEncoding;
type BoolArray = Array<BoolEncoding>;

// Auto-generated vtable based on the Encoding trait.
// I don't really want a different trait for borrowed and owned functions, but maybe it's necessary.
pub const BOOL_VTABLE: EncodingVTable = EncodingVTable {
    id: "vortex.bool",
    into_canonical: &|data| {
        <Array<BoolEncoding> as TryFrom<ArrayData>>::try_from(data).and_then(|a| a.into_canonical())
    },
    is_valid: &|data, idx| {
        <&Array<BoolEncoding> as TryFrom<&ArrayData>>::try_from(data).and_then(|a| a.is_valid(idx))
    },
};

impl Encoding for BoolEncoding {
    const ID: &'static str = "vortex.bool";
}

impl ArrayImpl for BoolArray {
    fn as_array_data(&self) -> &ArrayData {
        &self.data
    }

    fn into_array_data(self) -> ArrayData {
        self.data
    }

    fn into_canonical(self) -> VortexResult<Canonical> {
        // Can implement array functionality that takes ownership of data!
        todo!()
    }

    fn is_valid(&self, _index: usize) -> VortexResult<bool> {
        todo!()
    }
}
