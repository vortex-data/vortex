use std::fmt::Debug;

use arrow_buffer::BooleanBuffer;
use vortex_array::arrays::BoolArray;
use vortex_array::stats::{ArrayStats, StatsSetRef};
use vortex_array::validity::Validity;
use vortex_array::variants::BoolArrayTrait;
use vortex_array::vtable::VTableRef;
use vortex_array::{
    Array, ArrayCanonicalImpl, ArrayImpl, ArrayStatisticsImpl, ArrayValidityImpl,
    ArrayVariantsImpl, Canonical, EmptyMetadata, Encoding, try_from_array_ref,
};
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_panic};
use vortex_mask::Mask;

#[derive(Clone, Debug)]
pub struct ByteBoolArray {
    dtype: DType,
    buffer: ByteBuffer,
    validity: Validity,
    stats_set: ArrayStats,
}

try_from_array_ref!(ByteBoolArray);

pub struct ByteBoolEncoding;
impl Encoding for ByteBoolEncoding {
    type Array = ByteBoolArray;
    type Metadata = EmptyMetadata;
}

impl ByteBoolArray {
    pub fn new(buffer: ByteBuffer, validity: Validity) -> Self {
        let length = buffer.len();
        if let Some(vlen) = validity.maybe_len() {
            if length != vlen {
                vortex_panic!(
                    "Buffer length ({}) does not match validity length ({})",
                    length,
                    vlen
                );
            }
        }
        Self {
            dtype: DType::Bool(validity.nullability()),
            buffer,
            validity,
            stats_set: Default::default(),
        }
    }

    // TODO(ngates): deprecate construction from vec
    pub fn from_vec<V: Into<Validity>>(data: Vec<bool>, validity: V) -> Self {
        let validity = validity.into();
        // SAFETY: we are transmuting a Vec<bool> into a Vec<u8>
        let data: Vec<u8> = unsafe { std::mem::transmute(data) };
        Self::new(ByteBuffer::from(data), validity)
    }

    pub fn buffer(&self) -> &ByteBuffer {
        &self.buffer
    }

    pub fn validity(&self) -> &Validity {
        &self.validity
    }

    pub fn as_slice(&self) -> &[bool] {
        // Safety: The internal buffer contains byte-sized bools
        unsafe { std::mem::transmute(self.buffer().as_slice()) }
    }
}

impl ArrayImpl for ByteBoolArray {
    type Encoding = ByteBoolEncoding;

    fn _len(&self) -> usize {
        self.buffer.len()
    }

    fn _dtype(&self) -> &DType {
        &self.dtype
    }

    fn _vtable(&self) -> VTableRef {
        VTableRef::new_ref(&ByteBoolEncoding)
    }
}

impl ArrayCanonicalImpl for ByteBoolArray {
    fn _to_canonical(&self) -> VortexResult<Canonical> {
        let boolean_buffer = BooleanBuffer::from(self.as_slice());
        let validity = self.validity().clone();
        Ok(Canonical::Bool(BoolArray::new(boolean_buffer, validity)))
    }
}

impl ArrayStatisticsImpl for ByteBoolArray {
    fn _stats_ref(&self) -> StatsSetRef<'_> {
        self.stats_set.to_ref(self)
    }
}

impl ArrayValidityImpl for ByteBoolArray {
    fn _is_valid(&self, index: usize) -> VortexResult<bool> {
        self.validity.is_valid(index)
    }

    fn _all_valid(&self) -> VortexResult<bool> {
        self.validity.all_valid()
    }

    fn _all_invalid(&self) -> VortexResult<bool> {
        self.validity.all_invalid()
    }

    fn _validity_mask(&self) -> VortexResult<Mask> {
        self.validity.to_mask(self.len())
    }
}

impl ArrayVariantsImpl for ByteBoolArray {
    fn _as_bool_typed(&self) -> Option<&dyn BoolArrayTrait> {
        Some(self)
    }
}

impl BoolArrayTrait for ByteBoolArray {}

impl From<Vec<bool>> for ByteBoolArray {
    fn from(value: Vec<bool>) -> Self {
        Self::from_vec(value, Validity::AllValid)
    }
}

impl From<Vec<Option<bool>>> for ByteBoolArray {
    fn from(value: Vec<Option<bool>>) -> Self {
        let validity = Validity::from_iter(value.iter().map(|v| v.is_some()));

        // This doesn't reallocate, and the compiler even vectorizes it
        let data = value.into_iter().map(Option::unwrap_or_default).collect();

        Self::from_vec(data, validity)
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    // #[cfg_attr(miri, ignore)]
    // #[test]
    // fn test_bytebool_metadata() {
    //     check_metadata(
    //         "bytebool.metadata",
    //         SerdeMetadata(ByteBoolMetadata {
    //             validity: ValidityMetadata::AllValid,
    //         }),
    //     );
    // }

    #[test]
    fn test_validity_construction() {
        let v = vec![true, false];
        let v_len = v.len();

        let arr = ByteBoolArray::from(v);
        assert_eq!(v_len, arr.len());

        for idx in 0..arr.len() {
            assert!(arr.is_valid(idx).unwrap());
        }

        let v = vec![Some(true), None, Some(false)];
        let arr = ByteBoolArray::from(v);
        assert!(arr.is_valid(0).unwrap());
        assert!(!arr.is_valid(1).unwrap());
        assert!(arr.is_valid(2).unwrap());
        assert_eq!(arr.len(), 3);

        let v: Vec<Option<bool>> = vec![None, None];
        let v_len = v.len();

        let arr = ByteBoolArray::from(v);
        assert_eq!(v_len, arr.len());

        for idx in 0..arr.len() {
            assert!(!arr.is_valid(idx).unwrap());
        }
        assert_eq!(arr.len(), 2);
    }
}
