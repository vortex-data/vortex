// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shrinkable storage values and extra nesting for the Arrow round-trip suites.
//!
//! Draws are separate so Hegel can reduce lengths, null masks, and values independently. Extension
//! metadata and logical value constraints belong to the generators in the owning plugin crates.
//! Temporal cases sample their family, unit, and timezone so those choices can also shrink.

use hegel::TestCase;
use hegel::generators as gs;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::builders::builder_with_capacity_in;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::BufferAllocatorRef;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::DefaultExport;
use super::check_containers;

/// Draws storage values with independently shrinkable validity, lengths, and primitive values.
///
/// Variable-length lists use a shared width from zero through four. Fixed-size lists follow the
/// dtype, including zero-width lists with nonzero row counts. Primitive values range from zero
/// through 100, while other leaf dtypes use their builder's zero value. Callers must generate any
/// additional logical constraints, such as valid JSON or geometry bytes, themselves.
///
/// Returns an error if child lengths or offsets cannot be represented.
pub fn draw_storage(
    tc: &TestCase,
    dtype: &DType,
    len: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    Ok(match dtype {
        DType::Struct(fields, _) => {
            let validity = draw_validity(tc, dtype.nullability(), len);

            StructArray::try_new(
                fields.names().clone(),
                fields
                    .fields()
                    .map(|d| draw_storage(tc, &d, len, ctx))
                    .collect::<VortexResult<Vec<_>>>()?,
                len,
                validity,
            )?
            .into_array()
        }
        DType::List(child, _) => {
            let validity = draw_validity(tc, dtype.nullability(), len);
            let width = tc.draw(gs::integers::<usize>().max_value(4));
            let child_len = len.checked_mul(width).ok_or_else(|| {
                vortex_err!("list child length must fit in usize, got {len} rows of width {width}")
            })?;
            let child = draw_storage(tc, child, child_len, ctx)?;
            let offsets = PrimitiveArray::from_iter(
                (0..=len)
                    .map(|i| u64::try_from(i * width))
                    .collect::<Result<Vec<_>, _>>()?,
            )
            .into_array();

            ListArray::try_new(child, offsets, validity, ctx)?.into_array()
        }
        DType::FixedSizeList(child, size, _) => {
            let validity = draw_validity(tc, dtype.nullability(), len);
            let width = usize::try_from(*size)?;
            let child_len = len.checked_mul(width).ok_or_else(|| {
                vortex_err!(
                    "fixed-size-list child length must fit in usize, got {len} rows of width {width}"
                )
            })?;
            let child = draw_storage(tc, child, child_len, ctx)?;

            FixedSizeListArray::try_new(child, *size, validity, len)?.into_array()
        }
        _ => draw_flat_storage(tc, dtype, len)?,
    })
}

fn draw_validity(tc: &TestCase, nullability: Nullability, len: usize) -> Validity {
    match nullability {
        Nullability::Nullable => {
            Validity::from_iter(tc.draw(gs::vecs(gs::booleans()).min_size(len).max_size(len)))
        }
        Nullability::NonNullable => Validity::NonNullable,
    }
}

/// Draws nullable flat values, using zeros for non-numeric storage.
fn draw_flat_storage(tc: &TestCase, dtype: &DType, len: usize) -> VortexResult<ArrayRef> {
    let mut builder = builder_with_capacity_in(dtype, len, BufferAllocatorRef::static_ref());

    for _ in 0..len {
        if dtype.is_nullable() && tc.draw(gs::booleans()) {
            builder.append_nulls(1);
        } else if dtype.is_primitive() {
            let value = tc.draw(gs::integers::<i32>().min_value(0).max_value(100));
            builder.append_scalar(&Scalar::primitive(value, dtype.nullability()).cast(dtype)?)?;
        } else {
            builder.append_zeros(1);
        }
    }

    Ok(builder.finish())
}

/// Checks every container placement, then repeats the checks with up to three generated parents.
///
/// Extra struct, list, and fixed-size-list parents preserve the leaf's row count and independently
/// draw their nullability and validity. `default_export` applies to the leaf. The generated parents
/// support field inference, including when the leaf requires an explicit target.
#[track_caller]
pub fn check_generated(
    tc: &TestCase,
    leaf: ArrayRef,
    default_export: DefaultExport,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    check_containers(leaf.clone(), default_export, ctx)?;

    let depth = tc.draw(gs::integers::<usize>().max_value(3));
    let mut array = leaf;

    for _ in 0..depth {
        let len = array.len();
        let validity = draw_validity(tc, tc.draw(gs::booleans()).into(), len);
        array = match tc.draw(gs::integers::<u8>().max_value(2)) {
            0 => StructArray::try_new(FieldNames::from(["child"]), vec![array], len, validity)?
                .into_array(),
            1 => ListArray::try_new(
                array,
                PrimitiveArray::from_iter(0..=u32::try_from(len)?).into_array(),
                validity,
                ctx,
            )?
            .into_array(),
            _ => FixedSizeListArray::try_new(array, 1, validity, len)?.into_array(),
        };
    }

    if depth > 0 {
        check_containers(array, DefaultExport::Supported, ctx)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use hegel::TestCase;
    use hegel::generators as gs;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ExtensionArray;
    use vortex_array::extension::datetime::Date;
    use vortex_array::extension::datetime::Time;
    use vortex_array::extension::datetime::TimeUnit;
    use vortex_array::extension::datetime::Timestamp;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::DefaultExport;
    use super::check_generated;
    use super::draw_storage;

    fn check_temporal_round_trip(tc: &TestCase) -> VortexResult<()> {
        let session = VortexSession::empty();
        let mut ctx = session.create_execution_ctx();

        let len = tc.draw(gs::integers::<usize>().max_value(32));
        let nullability = tc.draw(gs::booleans()).into();
        let date_unit = tc.draw(gs::sampled_from(&[TimeUnit::Days, TimeUnit::Milliseconds]));
        let time_unit = tc.draw(gs::sampled_from(&[
            TimeUnit::Seconds,
            TimeUnit::Milliseconds,
            TimeUnit::Microseconds,
            TimeUnit::Nanoseconds,
        ]));
        let timezone = tc.draw(gs::sampled_from(&[
            None,
            Some("UTC"),
            Some("America/New_York"),
            Some("+05:30"),
        ]));
        let dtypes = [
            Date::try_new(date_unit, nullability)?.erased(),
            Time::try_new(time_unit, nullability)?.erased(),
            Timestamp::try_new_with_tz(time_unit, timezone.map(Into::into), nullability)?.erased(),
        ];
        let dtype = tc.draw(gs::sampled_from(&dtypes));
        let storage = draw_storage(tc, dtype.storage_dtype(), len, &mut ctx)?;
        let array = ExtensionArray::try_new(dtype, storage)?.into_array();

        check_generated(tc, array, DefaultExport::Supported, &mut ctx)
    }

    #[hegel::test(test_cases = 100, print_blob = true)]
    fn temporal_arrow_transport(tc: TestCase) {
        check_temporal_round_trip(&tc)
            .vortex_expect("temporal Arrow transport must preserve dtype and stored values");
    }
}
