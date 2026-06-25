// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use vortex_array::AnyColumnar;
use vortex_array::CanonicalView;
use vortex_array::ColumnarView::Canonical;
use vortex_array::ColumnarView::Constant;
use vortex_array::IntoArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::TemporalArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::dtype::DType;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::extension::datetime::Timestamp;
use vortex_array::match_each_integer_ptype;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect as _;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;

use crate::array::DateTimePartsParts;

/// Decode [`DateTimePartsParts`] into a [`TemporalArray`].
pub fn decode_to_temporal(parts: DateTimePartsParts, dtype: &DType) -> VortexResult<TemporalArray> {
    let DType::Extension(ext) = dtype else {
        vortex_panic!(Compute: "expected dtype to be DType::Extension variant")
    };

    let Some(options) = ext.metadata_opt::<Timestamp>() else {
        vortex_panic!(Compute: "must decode TemporalMetadata from extension metadata");
    };

    let divisor = match options.unit {
        TimeUnit::Nanoseconds => 1_000_000_000,
        TimeUnit::Microseconds => 1_000_000,
        TimeUnit::Milliseconds => 1_000,
        TimeUnit::Seconds => 1,
        TimeUnit::Days => vortex_panic!(InvalidArgument: "cannot decode into TimeUnit::D"),
    };

    // Days is guaranteed Primitive by require_child.
    let days = parts.days.as_::<Primitive>();
    let validity = days.validity()?;

    let seconds = parts.seconds.as_::<AnyColumnar>();
    let constant_seconds = if let Constant(seconds) = seconds {
        let seconds = seconds
            .scalar()
            .value()
            .vortex_expect("no value")
            .as_primitive()
            .as_i64()
            .vortex_expect("non-nullable");
        seconds * divisor
    } else {
        0
    };

    let subseconds = parts.subseconds.as_::<AnyColumnar>();
    let constant_subseconds = if let Constant(subseconds) = subseconds {
        subseconds
            .scalar()
            .value()
            .vortex_expect("no value")
            .as_primitive()
            .as_i64()
            .vortex_expect("non-nullable")
    } else {
        0
    };

    let mut values = decode_days(
        &days,
        86_400i64 * divisor,
        constant_seconds + constant_subseconds,
    );

    if let Canonical(seconds) = seconds {
        let CanonicalView::Primitive(seconds) = seconds else {
            vortex_panic!("not a primitive");
        };
        match_each_integer_ptype!(seconds.ptype(), |S| {
            for (v, second) in values.iter_mut().zip(seconds.as_slice::<S>()) {
                let second: i64 = second.as_();
                *v += second * divisor;
            }
        });
    }
    if let Canonical(subseconds) = subseconds {
        let CanonicalView::Primitive(subseconds) = subseconds else {
            vortex_panic!("not a primitive");
        };
        match_each_integer_ptype!(subseconds.ptype(), |S| {
            for (v, subsecond) in values.iter_mut().zip(subseconds.as_slice::<S>()) {
                let subsecond: i64 = subsecond.as_();
                *v += subsecond;
            }
        });
    }

    Ok(TemporalArray::new_timestamp(
        PrimitiveArray::new(values.freeze(), validity).into_array(),
        options.unit,
        options.tz.clone(),
    ))
}

// For constant seconds and subseconds, compute day * day_to_unit + const_offset
fn decode_days<P: PrimitiveArrayExt>(days: &P, day_to_unit: i64, offset: i64) -> BufferMut<i64> {
    /// If "days" are u16 or u32, LLVM doesn't auto-vectorize the code due to
    /// widening to i64. If we process the code in fixed-size chunks and unroll
    /// the chunks with seq_macro, vectorization happens.
    const CHUNK: usize = 64;
    let n = days.len();
    let mut values = BufferMut::<i64>::with_capacity(n);
    match_each_integer_ptype!(days.ptype(), |D| {
        let src = days.as_slice::<D>();
        let (src_chunks, src_rem) = src.as_chunks::<CHUNK>();
        let (dst_chunks, _) = values.spare_capacity_mut()[..n].as_chunks_mut::<CHUNK>();
        for (s_chunk, d_chunk) in src_chunks.iter().zip(dst_chunks.iter_mut()) {
            seq_macro::seq!(I in 0..64 {
                let day: i64 = s_chunk[I].as_();
                d_chunk[I].write(day * day_to_unit + offset);
            });
        }
        let tail_start = src_chunks.len() * CHUNK;
        let dst_tail = &mut values.spare_capacity_mut()[tail_start..n];
        for (s, d) in src_rem.iter().zip(dst_tail.iter_mut()) {
            let day: i64 = s.as_();
            d.write(day * day_to_unit + offset);
        }
    });
    // SAFETY: every element in 0..n was written above.
    unsafe { values.set_len(n) };
    values
}

#[cfg(test)]
mod test {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::TemporalArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::extension::datetime::TimeUnit;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::DateTimeParts;
    use crate::array::DateTimePartsArraySlotsExt;
    use crate::array::DateTimePartsParts;
    use crate::canonical::decode_to_temporal;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

    #[rstest]
    #[case(Validity::NonNullable)]
    #[case(Validity::AllValid)]
    #[case(Validity::AllInvalid)]
    #[case(Validity::from_iter([true, true, false, false, true, true]))]
    fn test_decode_to_temporal(#[case] validity: Validity) -> VortexResult<()> {
        let milliseconds = PrimitiveArray::new(
            buffer![
                86_400i64, // element with only day component
                -86_400i64,
                86_400i64 + 1000, // element with day + second components
                -86_400i64 - 1000,
                86_400i64 + 1000 + 1, // element with day + second + sub-second components
                -86_400i64 - 1000 - 1
            ],
            validity.clone(),
        );
        let mut ctx = SESSION.create_execution_ctx();
        let date_times = DateTimeParts::try_from_temporal(
            TemporalArray::new_timestamp(
                milliseconds.clone().into_array(),
                TimeUnit::Milliseconds,
                Some("UTC".into()),
            ),
            &mut ctx,
        )?;

        assert!(date_times.as_array().validity()?.mask_eq(
            &validity,
            milliseconds.len(),
            &mut ctx
        )?);

        let dtype = date_times.dtype().clone();
        let parts = DateTimePartsParts {
            days: date_times.days().clone(),
            seconds: date_times.seconds().clone(),
            subseconds: date_times.subseconds().clone(),
        };

        let primitive_values = decode_to_temporal(parts, &dtype)?
            .temporal_values()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?;

        assert_arrays_eq!(primitive_values, milliseconds, &mut ctx);
        Ok(())
    }
}
