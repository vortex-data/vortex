// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::Canonical;
use vortex_array::builders::ArrayBuilder;
use vortex_array::vtable::CanonicalVTable;
use vortex_dtype::match_each_integer_ptype;
use vortex_error::VortexExpect;

use crate::bitpack_decompress::{unpack, unpack_into};
use crate::{BitPackedArray, BitPackedVTable};

impl CanonicalVTable<BitPackedVTable> for BitPackedVTable {
    fn canonicalize(array: &BitPackedArray) -> Canonical {
        Canonical::Primitive(unpack(array))
    }

    fn append_to_builder(array: &BitPackedArray, builder: &mut dyn ArrayBuilder) {
        match_each_integer_ptype!(array.ptype(), |T| {
            unpack_into::<T>(
                array,
                builder
                    .as_any_mut()
                    .downcast_mut()
                    .vortex_expect("bit packed array must canonicalize into a primitive array"),
            )
        })
    }
}
