// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex_array::arrays::listview::ListViewArrayExt;
use vortex_array::arrays::map::MapArrayExt;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::arrays::varbin::varbin_scalar;
use vortex_array::dtype::DType;
use vortex_array::match_each_decimal_value_type;
use vortex_array::match_each_native_ptype;
use vortex_array::scalar::DecimalValue;
use vortex_array::scalar::Scalar;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

/// Baseline implementation of scalar_at that works on canonical arrays.
/// This implementation manually extracts the scalar value from each canonical type
/// without using the scalar_at method, to serve as an independent baseline for testing.
pub fn scalar_at_canonical_array(
    canonical: Canonical,
    index: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Scalar> {
    let canonical_ref = canonical.clone().into_array();
    if canonical_ref.is_invalid(index, ctx)? {
        return Ok(Scalar::null(canonical_ref.dtype().clone()));
    }
    Ok(match canonical {
        Canonical::Null(_array) => Scalar::null(DType::Null),
        Canonical::Bool(array) => Scalar::bool(
            array.to_bit_buffer().value(index),
            array.dtype().nullability(),
        ),
        Canonical::Primitive(array) => {
            match_each_native_ptype!(array.ptype(), |T| {
                Scalar::primitive(array.as_slice::<T>()[index], array.dtype().nullability())
            })
        }
        Canonical::Decimal(array) => {
            match_each_decimal_value_type!(array.values_type(), |D| {
                Scalar::decimal(
                    DecimalValue::from(array.buffer::<D>()[index]),
                    array.decimal_dtype(),
                    array.dtype().nullability(),
                )
            })
        }
        Canonical::VarBinView(array) => varbin_scalar(array.bytes_at(index), array.dtype()),
        Canonical::List(array) => {
            let list = array.list_elements_at(index)?;
            let children: Vec<Scalar> = (0..list.len())
                .map(|i| {
                    let canonical = list
                        .clone()
                        .execute::<Canonical>(ctx)
                        .vortex_expect("to_canonical should succeed in fuzz test");
                    scalar_at_canonical_array(canonical, i, ctx)
                        .vortex_expect("scalar_at_canonical_array should succeed in fuzz test")
                })
                .collect();
            Scalar::list(
                Arc::new(list.dtype().clone()),
                children,
                array.dtype().nullability(),
            )
        }
        Canonical::FixedSizeList(array) => {
            let list = array.fixed_size_list_elements_at(index)?;
            let children: Vec<Scalar> = (0..list.len())
                .map(|i| {
                    let canonical = list
                        .clone()
                        .execute::<Canonical>(ctx)
                        .vortex_expect("to_canonical should succeed in fuzz test");
                    scalar_at_canonical_array(canonical, i, ctx)
                        .vortex_expect("scalar_at_canonical_array should succeed in fuzz test")
                })
                .collect();
            Scalar::fixed_size_list(list.dtype().clone(), children, array.dtype().nullability())
        }
        Canonical::Struct(array) => {
            let field_scalars: Vec<Scalar> = array
                .iter_unmasked_fields()
                .map(|field| {
                    let canonical = field
                        .clone()
                        .execute::<Canonical>(ctx)
                        .vortex_expect("to_canonical should succeed in fuzz test");
                    scalar_at_canonical_array(canonical, index, ctx)
                        .vortex_expect("scalar_at_canonical_array should succeed in fuzz test")
                })
                .collect();
            Scalar::struct_(array.dtype().clone(), field_scalars)
        }
        Canonical::Extension(array) => {
            let storage_canonical = array.storage_array().clone().execute::<Canonical>(ctx)?;
            let storage_scalar = scalar_at_canonical_array(storage_canonical, index, ctx)?;
            Scalar::extension_ref(array.ext_dtype().clone(), storage_scalar)
        }
        Canonical::Union(_) => {
            todo!("TODO(connor)[Union]: support Union arrays in the scalar_at fuzzer")
        }
        Canonical::Map(array) => {
            let entries = array.entries_at(index)?.execute::<StructArray>(ctx)?;
            let keys = entries
                .unmasked_field(0)
                .clone()
                .execute::<Canonical>(ctx)?;
            let values = entries
                .unmasked_field(1)
                .clone()
                .execute::<Canonical>(ctx)?;
            let pairs = (0..entries.len())
                .map(|entry_index| {
                    Ok((
                        scalar_at_canonical_array(keys.clone(), entry_index, ctx)?,
                        scalar_at_canonical_array(values.clone(), entry_index, ctx)?,
                    ))
                })
                .collect::<VortexResult<Vec<_>>>()?;
            Scalar::try_map(array.dtype().clone(), pairs)?
        }
        Canonical::Variant(_) => unreachable!("Variant arrays are not fuzzed"),
    })
}

#[cfg(test)]
mod tests {
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::MapArray;
    use vortex_array::builders::ArrayBuilder;
    use vortex_array::builders::MapBuilder;
    use vortex_array::dtype::MapDType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;

    use super::*;
    use crate::SESSION;

    #[test]
    fn map_scalar_at_uses_independent_baseline() -> VortexResult<()> {
        let map_dtype = MapDType::try_new(
            DType::Primitive(PType::I32, Nullability::NonNullable),
            DType::Utf8(Nullability::Nullable),
            false,
        )?;
        let dtype = DType::Map(map_dtype.clone(), Nullability::Nullable);
        let rows = [
            Scalar::try_map(
                dtype.clone(),
                [(
                    Scalar::primitive(7i32, Nullability::NonNullable),
                    Scalar::utf8("seven", Nullability::Nullable),
                )],
            )?,
            Scalar::null(dtype.clone()),
            Scalar::try_map(dtype, [])?,
        ];
        let mut builder =
            MapBuilder::<u64, u64>::with_capacity(map_dtype, Nullability::Nullable, rows.len());
        for row in &rows {
            builder.append_scalar(row)?;
        }
        let array: MapArray = builder.finish_into_map();
        let mut ctx = SESSION.create_execution_ctx();

        for (index, expected) in rows.into_iter().enumerate() {
            let canonical = array.clone().into_array().execute::<Canonical>(&mut ctx)?;
            assert_eq!(
                scalar_at_canonical_array(canonical, index, &mut ctx)?,
                expected
            );
        }

        Ok(())
    }
}
