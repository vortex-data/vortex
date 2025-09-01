// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::vtable::OperationsVTable;
use vortex_array::{ArrayRef, IntoArray};
use vortex_buffer::BufferString;
use vortex_dtype::DType;
use vortex_scalar::Scalar;

use crate::fsst_view::{FSSTViewArray, FSSTViewVTable};

impl OperationsVTable<FSSTViewVTable> for FSSTViewVTable {
    fn slice(array: &FSSTViewArray, range: Range<usize>) -> ArrayRef {
        // SAFETY: slicing views buffer doesn't modify any internal pointers.
        unsafe {
            FSSTViewArray::new_unchecked(
                array.views.slice(range.clone()),
                array.fsst_buffer.clone(),
                array.symbols.clone(),
                array.symbol_lengths.clone(),
                array.compressed_offsets.clone(),
                array.uncompressed_offsets.clone(),
                array.dtype.clone(),
                array.validity.slice(range),
            )
            .into_array()
        }
    }

    fn scalar_at(array: &FSSTViewArray, index: usize) -> Scalar {
        let bytes = array.bytes_at(index);
        match array.dtype() {
            DType::Utf8(n) => Scalar::utf8(unsafe { BufferString::new_unchecked(bytes) }, *n),
            DType::Binary(n) => Scalar::binary(bytes, *n),
            _ => unreachable!("FSSTViewArray can only be utf8/binary, checked at construction"),
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::Array;
    use vortex_array::arrays::VarBinViewArray;

    use crate::fsst_view::FSSTViewEncoding;

    #[test]
    fn test_scalar_at() {
        let canonical = VarBinViewArray::from_iter_str(["hello", "helloworld!!!"]).to_canonical();

        let fsst_view = FSSTViewEncoding.encode(&canonical, None).unwrap().unwrap();
        assert_eq!(fsst_view.scalar_at(0), "hello".into());
        assert_eq!(fsst_view.scalar_at(1), "helloworld!!!".into());
    }

    #[test]
    fn test_slice() {
        let canonical = VarBinViewArray::from_iter_str([
            "short1",
            "short2",
            "very_long_string1",
            "short3",
            "very_long_string2",
        ])
        .to_canonical();

        let fsst_view = FSSTViewEncoding.encode(&canonical, None).unwrap().unwrap();
        let sliced = fsst_view.slice(1..4);

        assert_eq!(sliced.scalar_at(0), "short2".into());
        assert_eq!(sliced.scalar_at(1), "very_long_string1".into());
        assert_eq!(sliced.scalar_at(2), "short3".into());
    }
}
