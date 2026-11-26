// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Native primitive array output for Populate benchmark fixtures.

use populate_core::Columns;
use populate_core::Distribution;
use populate_core::Nullable;
use populate_rand::Rng;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;

pub trait VortexColumns {
    fn into_vortex(self) -> ArrayRef;
}

impl<T: NativePType> VortexColumns for Vec<T> {
    fn into_vortex(self) -> ArrayRef {
        PrimitiveArray::new(Buffer::from(self), Validity::NonNullable).into_array()
    }
}

impl<T: NativePType> VortexColumns for Nullable<Vec<T>> {
    fn into_vortex(self) -> ArrayRef {
        let (mut words, len) = self.valid.into_parts();
        if cfg!(target_endian = "big") {
            for word in &mut words {
                *word = word.to_le();
            }
        }
        let bits = BitBuffer::new(Buffer::from(words).into_byte_buffer(), len);
        PrimitiveArray::new(Buffer::from(self.values), Validity::from(bits)).into_array()
    }
}

pub trait DistributionVortex: Distribution {
    fn collect_vortex(&mut self, rng: &mut impl Rng, len: usize) -> ArrayRef
    where
        Columns<Self::Item>: VortexColumns,
    {
        self.collect(rng, len).into_vortex()
    }
}

impl<D: Distribution> DistributionVortex for D {}
