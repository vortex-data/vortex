// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::Canonical;
use vortex_array::vtable::EncodeVTable;
use vortex_error::VortexResult;

use crate::{BitPackedArray, BitPackedVTable, bitpack_compress};

impl EncodeVTable<BitPackedVTable> for BitPackedVTable {
    fn encode(
        _vtable: &BitPackedVTable,
        canonical: &Canonical,
        like: Option<&BitPackedArray>,
    ) -> VortexResult<Option<BitPackedArray>> {
        let parray = canonical.clone().into_primitive();

        let bit_width = like
            .map(|like_array| like_array.bit_width())
            // Only reuse the bitwidth if its smaller than the array's original bitwidth.
            .filter(|bw| (*bw as usize) < parray.ptype().bit_width());

        // In our current benchmark suite this seems to be the faster option,
        // but it has an unbounded worst-case where some array becomes all patches.
        let (bit_width, bit_width_histogram) = match bit_width {
            Some(bw) => (bw, None),
            None => {
                let histogram = bitpack_compress::bit_width_histogram(&parray)?;
                let bit_width = bitpack_compress::find_best_bit_width(parray.ptype(), &histogram)?;
                (bit_width, Some(histogram))
            }
        };

        if bit_width as usize == parray.ptype().bit_width()
            || parray.ptype().is_signed_int()
                && parray.statistics().compute_min::<i64>().unwrap_or_default() < 0
        {
            // Bit-packed compression not supported.
            return Ok(None);
        }

        Ok(Some(bitpack_compress::bitpack_encode(
            &parray,
            bit_width,
            bit_width_histogram.as_deref(),
        )?))
    }
}
