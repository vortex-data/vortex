use std::fmt::{Debug, Display};
use std::mem::ManuallyDrop;

use arrow_buffer::BooleanBuffer;
use serde::{Deserialize, Serialize};
use vortex_array::array::visitor::{AcceptArrayVisitor, ArrayVisitor};
use vortex_array::array::BoolArray;
use vortex_array::encoding::ids;
use vortex_array::stats::StatsSet;
use vortex_array::validity::{ArrayValidity, LogicalValidity, Validity, ValidityMetadata};
use vortex_array::variants::{ArrayVariants, BoolArrayTrait};
use vortex_array::{
    impl_encoding, ArrayData, ArrayTrait, Canonical, IntoArrayData, IntoCanonical, TypedArray,
};
use vortex_buffer::Buffer;
use vortex_dtype::DType;
use vortex_error::{VortexExpect as _, VortexResult};

impl_encoding!("vortex.bytebool", ids::BYTE_BOOL, ByteBool);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ByteBoolMetadata {
    validity: ValidityMetadata,
}

impl Display for ByteBoolMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

impl ByteBoolArray {
    pub fn validity(&self) -> Validity {
        self.metadata().validity.to_validity(|| {
            self.as_ref()
                .child(0, &Validity::DTYPE, self.len())
                .vortex_expect("ByteBoolArray: accessing validity child")
        })
    }

    pub fn try_new(buffer: Buffer, validity: Validity) -> VortexResult<Self> {
        let length = buffer.len();

        let typed = TypedArray::try_from_parts(
            DType::Bool(validity.nullability()),
            length,
            ByteBoolMetadata {
                validity: validity.to_metadata(length)?,
            },
            Some(buffer),
            validity.into_array().into_iter().collect::<Vec<_>>().into(),
            StatsSet::new(),
        )?;

        Ok(typed.into())
    }

    pub fn try_from_vec<V: Into<Validity>>(data: Vec<bool>, validity: V) -> VortexResult<Self> {
        let validity = validity.into();
        let mut vec = ManuallyDrop::new(data);
        vec.shrink_to_fit();

        let ptr = vec.as_mut_ptr() as *mut u8;
        let length = vec.len();
        let capacity = vec.capacity();

        let bytes = unsafe { Vec::from_raw_parts(ptr, length, capacity) };

        let buffer = Buffer::from(bytes);

        Self::try_new(buffer, validity)
    }

    pub fn buffer(&self) -> &Buffer {
        self.as_ref()
            .buffer()
            .vortex_expect("ByteBoolArray is missing the underlying buffer")
    }

    pub fn maybe_null_slice(&self) -> &[bool] {
        // Safety: The internal buffer contains byte-sized bools
        unsafe { std::mem::transmute(self.buffer().as_slice()) }
    }
}

impl ArrayTrait for ByteBoolArray {}

impl ArrayVariants for ByteBoolArray {
    fn as_bool_array(&self) -> Option<&dyn BoolArrayTrait> {
        Some(self)
    }
}

impl BoolArrayTrait for ByteBoolArray {
    fn invert(&self) -> VortexResult<ArrayData> {
        ByteBoolArray::try_from_vec(
            self.maybe_null_slice().iter().map(|v| !v).collect(),
            self.validity(),
        )
        .map(|a| a.into_array())
    }

    fn maybe_null_indices_iter<'a>(&'a self) -> Box<dyn Iterator<Item = usize> + 'a> {
        todo!()
    }

    fn maybe_null_slices_iter<'a>(&'a self) -> Box<dyn Iterator<Item = (usize, usize)> + 'a> {
        todo!()
    }
}

impl From<Vec<bool>> for ByteBoolArray {
    fn from(value: Vec<bool>) -> Self {
        Self::try_from_vec(value, Validity::AllValid)
            .vortex_expect("Failed to create ByteBoolArray from Vec<bool>")
    }
}

impl From<Vec<Option<bool>>> for ByteBoolArray {
    fn from(value: Vec<Option<bool>>) -> Self {
        let validity = Validity::from_iter(value.iter());

        // This doesn't reallocate, and the compiler even vectorizes it
        let data = value.into_iter().map(Option::unwrap_or_default).collect();

        Self::try_from_vec(data, validity)
            .vortex_expect("Failed to create ByteBoolArray from nullable bools")
    }
}

impl IntoCanonical for ByteBoolArray {
    fn into_canonical(self) -> VortexResult<Canonical> {
        let boolean_buffer = BooleanBuffer::from(self.maybe_null_slice());
        let validity = self.validity();

        BoolArray::try_new(boolean_buffer, validity).map(Canonical::Bool)
    }
}

impl ArrayValidity for ByteBoolArray {
    fn is_valid(&self, index: usize) -> bool {
        self.validity().is_valid(index)
    }

    fn logical_validity(&self) -> LogicalValidity {
        self.validity().to_logical(self.len())
    }
}

impl AcceptArrayVisitor for ByteBoolArray {
    fn accept(&self, visitor: &mut dyn ArrayVisitor) -> VortexResult<()> {
        visitor.visit_buffer(self.buffer())?;
        visitor.visit_validity(&self.validity())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validity_construction() {
        let v = vec![true, false];
        let v_len = v.len();

        let arr = ByteBoolArray::from(v);
        assert_eq!(v_len, arr.len());

        for idx in 0..arr.len() {
            assert!(arr.is_valid(idx));
        }

        let v = vec![Some(true), None, Some(false)];
        let arr = ByteBoolArray::from(v);
        assert!(arr.is_valid(0));
        assert!(!arr.is_valid(1));
        assert!(arr.is_valid(2));
        assert_eq!(arr.len(), 3);

        let v: Vec<Option<bool>> = vec![None, None];
        let v_len = v.len();

        let arr = ByteBoolArray::from(v);
        assert_eq!(v_len, arr.len());

        for idx in 0..arr.len() {
            assert!(!arr.is_valid(idx));
        }
        assert_eq!(arr.len(), 2);
    }
}
