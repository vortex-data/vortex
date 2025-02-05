use arrow_buffer::ArrowNativeType;
use vortex_array::accessor::ArrayAccessor;
use vortex_array::array::{BoolArray, ListArray, PrimitiveArray, StructArray, VarBinViewArray};
use vortex_array::validity::Validity;
use vortex_array::variants::{PrimitiveArrayTrait, StructArrayTrait};
use vortex_array::{Array, IntoArray, IntoArrayVariant};
use vortex_dtype::{match_each_native_ptype, DType, NativePType};
use vortex_error::VortexResult;

pub fn slice_canonical_array(array: &Array, start: usize, stop: usize) -> VortexResult<Array> {
    let validity = if array.dtype().is_nullable() {
        let bool_buff = array.validity_mask()?.to_boolean_buffer();

        Validity::from(bool_buff.slice(start, stop - start))
    } else {
        Validity::NonNullable
    };

    match array.dtype() {
        DType::Bool(_) => {
            let bool_array = array.clone().into_bool()?;
            let sliced_bools = bool_array.boolean_buffer().slice(start, stop - start);
            BoolArray::try_new(sliced_bools, validity).map(|a| a.into_array())
        }
        DType::Primitive(p, _) => {
            let primitive_array = array.clone().into_primitive()?;
            match_each_native_ptype!(p, |$P| {
                Ok(PrimitiveArray::new(primitive_array.buffer::<$P>().slice(start..stop), validity).into_array())
            })
        }
        DType::Utf8(_) | DType::Binary(_) => {
            let utf8 = array.clone().into_varbinview()?;
            let values =
                utf8.with_iterator(|iter| iter.map(|v| v.map(|u| u.to_vec())).collect::<Vec<_>>())?;
            Ok(VarBinViewArray::from_iter(
                values[start..stop].iter().cloned(),
                array.dtype().clone(),
            )
            .into_array())
        }
        DType::Struct(..) => {
            let struct_array = array.clone().into_struct()?;
            let sliced_children = struct_array
                .children()
                .map(|c| slice_canonical_array(&c, start, stop))
                .collect::<VortexResult<Vec<_>>>()?;
            StructArray::try_new(
                struct_array.names().clone(),
                sliced_children,
                stop - start,
                validity,
            )
            .map(|a| a.into_array())
        }
        DType::List(..) => {
            let list_array = array.clone().into_list()?;
            let offsets =
                slice_canonical_array(&list_array.offsets(), start, stop + 1)?.into_primitive()?;

            let elements = slice_canonical_array(
                &list_array.elements(),
                offsets.get_as_cast::<u64>(0) as usize,
                offsets.get_as_cast::<u64>(offsets.len() - 1) as usize,
            )?;
            let offsets = match_each_native_ptype!(offsets.ptype(), |$P| {
                shift_offsets::<$P>(offsets)
            })
            .into_array();
            ListArray::try_new(elements, offsets, validity).map(|a| a.into_array())
        }
        _ => unreachable!("Not a canonical array"),
    }
}

fn shift_offsets<O: NativePType + ArrowNativeType>(offsets: PrimitiveArray) -> PrimitiveArray {
    if offsets.is_empty() {
        return offsets;
    }
    let offsets: Vec<O> = offsets.as_slice().to_vec();
    let start = offsets[0];
    PrimitiveArray::from_iter(offsets.into_iter().map(|o| o - start).collect::<Vec<_>>())
}
