// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::ToCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::accessor::ArrayAccessor;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::builders::builder_with_capacity;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::NativeDecimalType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::match_each_decimal_value_type;
use vortex_array::match_each_native_ptype;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

pub fn take_canonical_array_non_nullable_indices(
    array: &ArrayRef,
    indices: &[usize],
) -> VortexResult<ArrayRef> {
    take_canonical_array(
        array,
        indices
            .iter()
            .map(|i| Some(*i))
            .collect::<Vec<_>>()
            .as_slice(),
    )
}

pub fn take_canonical_array(array: &ArrayRef, indices: &[Option<usize>]) -> VortexResult<ArrayRef> {
    let nullable: Nullability = indices.contains(&None).into();

    let validity = if array.dtype().is_nullable() || nullable == Nullability::Nullable {
        let validity_idx = array
            .validity()?
            .to_mask(array.len(), &mut LEGACY_SESSION.create_execution_ctx())?
            .to_bit_buffer();

        Validity::from_iter(
            indices
                .iter()
                .map(|i| i.is_some_and(|i| validity_idx.value(i))),
        )
    } else {
        Validity::NonNullable
    };

    let indices_non_opt = indices.iter().map(|i| i.unwrap_or(0)).collect::<Vec<_>>();
    let indices_slice_non_opt = indices_non_opt.as_slice();

    match array.dtype() {
        DType::Bool(_) => {
            let bool_array = array.to_bool();
            let vec_values = bool_array.to_bit_buffer().iter().collect::<Vec<_>>();
            Ok(BoolArray::new(
                indices_slice_non_opt
                    .iter()
                    .map(|i| vec_values[*i])
                    .collect(),
                validity,
            )
            .into_array())
        }
        DType::Primitive(p, _) => {
            let primitive_array = array.to_primitive();
            match_each_native_ptype!(p, |P| {
                Ok(take_primitive::<P>(
                    primitive_array,
                    validity,
                    indices_slice_non_opt,
                ))
            })
        }
        DType::Decimal(d, _) => {
            let decimal_array = array.to_decimal();

            match_each_decimal_value_type!(decimal_array.values_type(), |D| {
                Ok(take_decimal::<D>(
                    decimal_array,
                    d,
                    validity,
                    indices_slice_non_opt,
                ))
            })
        }
        DType::Utf8(_) | DType::Binary(_) => {
            let utf8 = array.to_varbinview();
            let values =
                utf8.with_iterator(|iter| iter.map(|v| v.map(|u| u.to_vec())).collect::<Vec<_>>());
            Ok(VarBinViewArray::from_iter(
                indices
                    .iter()
                    .map(|i| i.and_then(|idx| values[idx].clone())),
                array.dtype().clone().union_nullability(nullable),
            )
            .into_array())
        }
        DType::Struct(..) => {
            let struct_array = array.to_struct();
            let taken_children = struct_array
                .iter_unmasked_fields()
                .map(|c| take_canonical_array_non_nullable_indices(c, indices_slice_non_opt))
                .collect::<VortexResult<Vec<_>>>()?;

            StructArray::try_new(
                struct_array.names().clone(),
                taken_children,
                indices_slice_non_opt.len(),
                validity,
            )
            .map(|a| a.into_array())
        }
        DType::List(..) | DType::FixedSizeList(..) => {
            let mut builder = builder_with_capacity(
                &array.dtype().union_nullability(nullable),
                indices_slice_non_opt.len(),
            );
            let mut ctx = LEGACY_SESSION.create_execution_ctx();
            for idx in indices {
                if let Some(idx) = idx {
                    builder.append_scalar(
                        &array
                            .execute_scalar(*idx, &mut ctx)?
                            .cast(&array.dtype().union_nullability(nullable))
                            .vortex_expect("cannot cast scalar nullability"),
                    )?;
                } else {
                    builder.append_null()
                }
            }
            Ok(builder.finish())
        }
        d @ (DType::Null | DType::Extension(_) | DType::Variant(_)) => {
            unreachable!("DType {d} not supported for fuzzing")
        }
    }
}

fn take_primitive<T: NativePType>(
    primitive_array: PrimitiveArray,
    validity: Validity,
    indices: &[usize],
) -> ArrayRef {
    let vec_values = primitive_array.as_slice::<T>().to_vec();
    PrimitiveArray::new(
        indices
            .iter()
            .map(|i| vec_values[*i])
            .collect::<Buffer<T>>(),
        validity,
    )
    .into_array()
}

fn take_decimal<D: NativeDecimalType>(
    array: DecimalArray,
    decimal_type: &DecimalDType,
    validity: Validity,
    indices: &[usize],
) -> ArrayRef {
    let buf = array.buffer::<D>();
    let vec_values = buf.as_slice();
    DecimalArray::new(
        indices
            .iter()
            .map(|i| vec_values[*i])
            .collect::<Buffer<D>>(),
        *decimal_type,
        validity,
    )
    .into_array()
}
