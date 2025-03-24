use vortex_array::accessor::ArrayAccessor;
use vortex_array::arrays::{BoolArray, ListArray, PrimitiveArray, StructArray, VarBinViewArray};
use vortex_array::validity::Validity;
use vortex_array::variants::{PrimitiveArrayTrait, StructArrayTrait};
use vortex_array::{Array, ArrayRef, ToCanonical};
use vortex_dtype::{
    DType, NativePType, Nullability, match_each_integer_ptype, match_each_native_ptype,
};
use vortex_error::VortexResult;

pub fn slice_canonical_array(
    array: &dyn Array,
    start: usize,
    stop: usize,
) -> VortexResult<ArrayRef> {
    let validity = if array.dtype().is_nullable() {
        let bool_buff = array.validity_mask()?.to_boolean_buffer();
        Validity::from(bool_buff.slice(start, stop - start))
    } else {
        Validity::NonNullable
    };

    match array.dtype() {
        DType::Bool(_) => {
            let bool_array = array.to_bool()?;
            let sliced_bools = bool_array.boolean_buffer().slice(start, stop - start);
            Ok(BoolArray::new(sliced_bools, validity).into_array())
        }
        DType::Primitive(p, _) => {
            let primitive_array = array.to_primitive()?;
            match_each_native_ptype!(p, |$P| {
                Ok(PrimitiveArray::new(primitive_array.buffer::<$P>().slice(start..stop), validity).into_array())
            })
        }
        DType::Utf8(_) | DType::Binary(_) => {
            let utf8 = array.to_varbinview()?;
            let values =
                utf8.with_iterator(|iter| iter.map(|v| v.map(|u| u.to_vec())).collect::<Vec<_>>())?;
            Ok(VarBinViewArray::from_iter(
                values[start..stop].iter().cloned(),
                array.dtype().clone(),
            )
            .into_array())
        }
        DType::Struct(..) => {
            let struct_array = array.to_struct()?;
            let sliced_children = struct_array
                .fields()
                .iter()
                .map(|c| slice_canonical_array(c, start, stop))
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
            let list_array = array.to_list()?;
            let offsets =
                slice_canonical_array(list_array.offsets(), start, stop + 1)?.to_primitive()?;

            let (start, end) = match_each_integer_ptype!(offsets.ptype(), |$P| {
                let offset_slice = offsets.as_slice::<$P>();
                (usize::try_from(offset_slice[0])?, usize::try_from(offset_slice[offsets.len() - 1])?)
            });

            let elements = slice_canonical_array(list_array.elements(), start, end)?;
            let offsets = match_each_integer_ptype!(offsets.ptype(), |$P| {
                shift_offsets(offsets.as_slice::<$P>())
            })
            .into_array();
            ListArray::try_new(elements, offsets, validity).map(|a| a.into_array())
        }
        d => unreachable!("DType {d} not supported for fuzzing"),
    }
}

fn shift_offsets<O: NativePType>(offsets: &[O]) -> PrimitiveArray {
    if offsets.is_empty() {
        return PrimitiveArray::empty::<O>(Nullability::NonNullable);
    }
    let start = offsets[0];
    PrimitiveArray::from_iter(
        offsets
            .iter()
            .copied()
            .map(|o| o - start)
            .collect::<Vec<_>>(),
    )
}
