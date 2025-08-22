// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_array::types::ArrowDictionaryKeyType;
use arrow_array::{AnyDictionaryArray, DictionaryArray};
use vortex_array::ArrayRef;
use vortex_array::arrow::FromArrowArray;

use crate::DictArray;

impl<K: ArrowDictionaryKeyType> FromArrowArray<&DictionaryArray<K>> for DictArray {
    fn from_arrow(array: &DictionaryArray<K>, nullable: bool) -> Self {
        let keys = AnyDictionaryArray::keys(array);
        let keys = ArrayRef::from_arrow(keys, keys.is_nullable());
        let values = ArrayRef::from_arrow(array.values().as_ref(), nullable);
        // SAFETY: we assume that Arrow has checked the invariants on construction
        unsafe { DictArray::new_unchecked(keys, values) }
    }
}
