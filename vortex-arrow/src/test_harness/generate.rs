// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shrinkable storage values and extra nesting for the Arrow round-trip suites.
//!
//! Draws are separate so Hegel can reduce lengths, null masks, and values independently. Extension
//! metadata and logical value constraints belong to the generators in the owning plugin crates.

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::FixedSizeListArray as ArrowFixedSizeListArray;
use arrow_array::PrimitiveArray as ArrowPrimitiveArray;
use arrow_array::types::Float16Type;
use arrow_array::types::Float32Type;
use arrow_array::types::Float64Type;
use arrow_array::types::Int8Type;
use arrow_array::types::Int16Type;
use arrow_array::types::Int32Type;
use arrow_array::types::Int64Type;
use arrow_array::types::UInt8Type;
use arrow_array::types::UInt16Type;
use arrow_array::types::UInt32Type;
use arrow_array::types::UInt64Type;
use arrow_buffer::NullBuffer;
use arrow_schema::Field;
use half::f16;
use hegel::TestCase;
use hegel::generators as gs;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::session::DTypeSessionExt;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::check_containers;

/// A storage array and its independent Arrow reference, built from the same source draws.
pub struct StorageCase {
    /// Vortex storage to wrap in the owning extension.
    pub array: ArrayRef,
    /// Arrow storage built directly with arrow-rs constructors.
    pub arrow: ArrowArrayRef,
}

/// Draws numeric and fixed-size-list storage without invoking Arrow transport.
///
/// Callers bound row counts, list widths, and nesting.
/// Returns an error for unsupported dtypes or lengths that do not fit the storage layout.
pub fn draw_storage_case(
    tc: &TestCase,
    dtype: &DType,
    len: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<StorageCase> {
    let validity = draw_validity(tc, dtype.nullability(), len);
    let mask = validity.execute_mask(len, ctx)?;
    let nulls = dtype
        .is_nullable()
        .then(|| NullBuffer::from((0..len).map(|i| mask.value(i)).collect::<Vec<_>>()));
    let (array, arrow): (ArrayRef, ArrowArrayRef) = match dtype {
        DType::FixedSizeList(child, size, _) => {
            let child = draw_storage_case(
                tc,
                child,
                len.checked_mul(usize::try_from(*size)?)
                    .ok_or_else(|| vortex_err!("storage length overflow"))?,
                ctx,
            )?;
            let field = Arc::new(Field::new(
                "item",
                child.arrow.data_type().clone(),
                child.array.dtype().is_nullable(),
            ));
            let arrow = ArrowFixedSizeListArray::try_new_with_length(
                field,
                i32::try_from(*size)?,
                child.arrow,
                nulls,
                len,
            )?;
            let array =
                FixedSizeListArray::try_new(child.array, *size, validity, len)?.into_array();
            (array, Arc::new(arrow))
        }
        DType::Primitive(ptype, _) => {
            macro_rules! primitive {
                ($t:ty, $arrow:ty, $draw:expr) => {{
                    let values = (0..len).map(|_| $draw).collect::<Vec<$t>>();
                    let array =
                        PrimitiveArray::new(Buffer::from(values.clone()), validity).into_array();
                    let arrow = ArrowPrimitiveArray::<$arrow>::new(values.into(), nulls);
                    (array, Arc::new(arrow) as ArrowArrayRef)
                }};
            }
            match ptype {
                PType::U8 => primitive!(u8, UInt8Type, tc.draw(gs::integers::<u8>())),
                PType::U16 => primitive!(u16, UInt16Type, tc.draw(gs::integers::<u16>())),
                PType::U32 => primitive!(u32, UInt32Type, tc.draw(gs::integers::<u32>())),
                PType::U64 => primitive!(u64, UInt64Type, tc.draw(gs::integers::<u64>())),
                PType::I8 => primitive!(i8, Int8Type, tc.draw(gs::integers::<i8>())),
                PType::I16 => primitive!(i16, Int16Type, tc.draw(gs::integers::<i16>())),
                PType::I32 => primitive!(i32, Int32Type, tc.draw(gs::integers::<i32>())),
                PType::I64 => primitive!(i64, Int64Type, tc.draw(gs::integers::<i64>())),
                PType::F16 => primitive!(
                    f16,
                    Float16Type,
                    f16::from_bits(tc.draw(gs::integers::<u16>()))
                ),
                PType::F32 => primitive!(f32, Float32Type, tc.draw(gs::floats::<f32>())),
                PType::F64 => primitive!(f64, Float64Type, tc.draw(gs::floats::<f64>())),
            }
        }
        _ => return Err(vortex_err!("no independent storage generator for {dtype}")),
    };

    Ok(StorageCase { array, arrow })
}

pub(super) fn draw_validity(tc: &TestCase, nullability: Nullability, len: usize) -> Validity {
    match nullability {
        Nullability::Nullable => {
            Validity::from_iter(tc.draw(gs::vecs(gs::booleans()).min_size(len).max_size(len)))
        }
        Nullability::NonNullable => Validity::NonNullable,
    }
}

/// Checks extension metadata and every container placement, then up to three generated parents.
///
/// Extension leaves retain metadata and nested storage across top-level nullability changes and
/// reconstruction from serialized metadata and stored storage.
///
/// Extra struct, list, and fixed-size-list parents preserve the leaf's row count and independently
/// draw their nullability and validity.
#[track_caller]
pub fn check_generated(tc: &TestCase, leaf: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<()> {
    if let DType::Extension(dtype) = leaf.dtype() {
        let metadata = dtype.serialize_metadata()?;
        let dtypes = ctx.session().dtypes();
        let plugin = dtypes
            .registry()
            .get(&dtype.id())
            .vortex_expect("the owning crate registered its dtype");

        for nullability in [Nullability::NonNullable, Nullability::Nullable] {
            let changed = dtype.with_nullability(nullability);
            assert!(changed.eq_ignore_nullability(dtype));
            assert_eq!(
                changed.storage_dtype(),
                &dtype.storage_dtype().with_nullability(nullability)
            );
            assert_eq!(changed.serialize_metadata()?, metadata);
            assert_eq!(
                plugin.deserialize(&metadata, changed.storage_dtype().clone())?,
                changed
            );
        }
    }

    check_containers(leaf.clone(), ctx)?;

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
        check_containers(array, ctx)?;
    }

    Ok(())
}
