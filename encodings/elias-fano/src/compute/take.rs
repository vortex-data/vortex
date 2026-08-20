// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Taking arbitrary rows out of an Elias-Fano array through the cursor.

use num_traits::AsPrimitive;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::dict::TakeExecute;
use vortex_array::dtype::IntegerPType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::AllOr;

use crate::EliasFano;
use crate::EliasFanoCursor;

/// The whole array is decoded in one pass, rather than walked through the cursor, once a request
/// reaches roughly one row in `BULK_DECODE_THRESHOLD` of it.
///
/// Same constant as `vortex-fastlanes`'s take kernel, whose reasoning transfers because the
/// low-bits child *is* bit-packed. Elias-Fano's extra sampled select per index only pushes the true
/// crossover further toward bulk, so reusing 8 errs safe.
pub(crate) const BULK_DECODE_THRESHOLD: usize = 8;

impl TakeExecute for EliasFano {
    fn take(
        array: ArrayView<'_, Self>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if indices.len() * BULK_DECODE_THRESHOLD > array.len() {
            let decoded = array.array().clone().execute::<PrimitiveArray>(ctx)?;
            return decoded.into_array().take(indices.clone()).map(Some);
        }

        // A null index selects nothing, so its payload is not a position: it is whatever the slot
        // happened to hold, and reading or bounds-checking one turns a legal take into a failure.
        // Those rows come back null either way, so the walk below just skips them.
        let selected = indices.validity()?.execute_mask(indices.len(), ctx)?;
        let valid: Option<&BitBuffer> = match selected.bit_buffer() {
            AllOr::All => None,
            // Nothing to read at all, so answer before opening a cursor — which would materialise
            // a low-bits child that cannot be read in place, for reads that never happen.
            AllOr::None => {
                return Ok(Some(
                    ConstantArray::new(Scalar::null(array.dtype().as_nullable()), indices.len())
                        .into_array(),
                ));
            }
            AllOr::Some(valid) => Some(valid),
        };

        let ptype = array.dtype().as_ptype();
        let reference_bits = array.reference_bits();
        let taken_validity = array.validity()?.take(indices)?;
        let indices = indices.clone().execute::<PrimitiveArray>(ctx)?;

        let mut cursor = EliasFanoCursor::try_new(array, ctx)?;
        let taken = gather(
            &mut cursor,
            &indices,
            valid,
            reference_bits,
            taken_validity,
            ptype,
        )?;
        Ok(Some(taken.into_array()))
    }
}

fn gather(
    cursor: &mut EliasFanoCursor<'_>,
    indices: &PrimitiveArray,
    valid: Option<&BitBuffer>,
    reference_bits: u64,
    validity: Validity,
    ptype: PType,
) -> VortexResult<PrimitiveArray> {
    Ok(match_each_integer_ptype!(ptype, |P| {
        match_each_integer_ptype!(indices.ptype(), |I| {
            PrimitiveArray::new(
                gather_rows::<P, I>(cursor, indices.as_slice::<I>(), valid, reference_bits)?,
                validity,
            )
        })
    }))
}

/// The requested rows, in the column's own width.
///
/// `valid` is `None` when every index is a real position. Otherwise the rows it marks invalid are
/// left as zero and their index payloads are never read, which is what the caller's validity
/// already says about them.
fn gather_rows<P: NativePType, I: IntegerPType>(
    cursor: &mut EliasFanoCursor<'_>,
    indices: &[I],
    valid: Option<&BitBuffer>,
    reference_bits: u64,
) -> VortexResult<Buffer<P>>
where
    u64: AsPrimitive<P>,
{
    let mut values = BufferMut::<P>::with_capacity(indices.len());
    for (position, &raw) in indices.iter().enumerate() {
        if valid.is_some_and(|valid| !valid.value(position)) {
            values.push(P::default());
            continue;
        }
        // Refused rather than wrapped into range; `access_element` bounds-checks what survives.
        let index = raw
            .to_usize()
            .ok_or_else(|| vortex_err!("Elias-Fano take index {raw} is not a position"))?;
        let bits = reference_bits.wrapping_add(cursor.access_element(index)?);
        // Truncating the pattern to the column's width is exactly the two's complement result,
        // signed or unsigned, because the reference was added in the same modular arithmetic.
        values.push(bits.as_());
    }
    Ok(values.freeze())
}
