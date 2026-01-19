// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::Canonical;
use vortex_array::builders::ArrayBuilder;
use vortex_array::vtable::CanonicalVTable;
use vortex_dtype::match_each_integer_ptype;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::BitPackedArray;
use crate::BitPackedVTable;
use crate::bitpack_decompress::unpack_array;
use crate::bitpack_decompress::unpack_into_primitive_builder;

impl CanonicalVTable<BitPackedVTable> for BitPackedVTable {
    fn canonicalize(array: &BitPackedArray) -> VortexResult<Canonical> {
        Ok(Canonical::Primitive(unpack_array(array)))
    }

    fn append_to_builder(
        array: &BitPackedArray,
        builder: &mut dyn ArrayBuilder,
    ) -> VortexResult<()> {
        match_each_integer_ptype!(array.ptype(), |T| {
            unpack_into_primitive_builder::<T>(
                array,
                builder
                    .as_any_mut()
                    .downcast_mut()
                    .vortex_expect("bit packed array must canonicalize into a primitive array"),
            )
        });
        Ok(())
    }
}
