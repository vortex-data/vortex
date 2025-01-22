use itertools::Itertools;
use vortex_dtype::Nullability::NonNullable;
use vortex_dtype::{DType, PType};
use vortex_error::VortexResult;

use crate::array::varbin::VarBinArray;
use crate::array::BinaryView;
use crate::builders::{ArrayBuilder, ArrayBuilderExt, ViewBuilder, ViewDType};
use crate::compute::try_cast;
use crate::validity::ArrayValidity;
use crate::{ArrayDType, IntoCanonical};

impl IntoCanonical for VarBinArray {
    fn into_canonical_builder(self, builder: &mut dyn ArrayBuilder) -> VortexResult<()> {
        match self.dtype() {
            DType::Utf8(_) => into_canonical_builder(self, builder.as_utf8_mut()),
            DType::Binary(_) => into_canonical_builder(self, builder.as_binary_mut()),
            _ => unreachable!(),
        }
    }
}

fn into_canonical_builder<V: ViewDType>(
    array: VarBinArray,
    builder: &mut ViewBuilder<V>,
) -> VortexResult<()> {
    // Append the entire buffer to the builder
    // TODO(ngates): this only works if we don't care about duplicate values.
    let buffer = array.bytes();
    let buffer_idx = builder.append_buffer(buffer.clone());

    // Then create views from the offsets
    let offsets =
        try_cast(array.offsets(), &DType::Primitive(PType::U32, NonNullable))?.into_primitive()?;
    let validity = array.logical_validity().to_null_buffer()?;

    offsets
        .as_slice::<u32>()
        .iter()
        .tuple_windows()
        .enumerate()
        .for_each(|(idx, (start, end))| {
            if validity.as_ref().map_or(true, |v| v.is_valid(idx)) {
                let len = *end - *start;
                let mut prefix = [0u8; 4];
                prefix.copy_from_slice(&buffer[*start as usize..][..len.min(4) as usize]);
                let view = BinaryView::new_view(len, prefix, buffer_idx, *start);
                unsafe { builder.push_view_unchecked(view) };
            } else {
                builder.append_null();
            }
        });

    Ok(())
}

//
// fn into_arrow(self) -> VortexResult<ArrayRef> {
//     // Specialized implementation of `into_arrow` for VarBin since it has a direct
//     // Arrow representation.
//     varbin_to_arrow(&self)
// }
//
// fn into_arrow_with_data_type(self, data_type: &DataType) -> VortexResult<ArrayRef> {
//     let array_ref = self.into_arrow()?;
//
//     Ok(if array_ref.data_type() != data_type {
//         arrow_cast::cast(array_ref.as_ref(), data_type)?
//     } else {
//         array_ref
//     })
// }

#[cfg(test)]
mod test {
    use rstest::rstest;
    use vortex_dtype::{DType, Nullability};

    use crate::array::varbin::builder::VarBinBuilder;
    use crate::validity::ArrayValidity;
    use crate::{ArrayDType, IntoArrayData};

    #[rstest]
    #[case(DType::Utf8(Nullability::Nullable))]
    #[case(DType::Binary(Nullability::Nullable))]
    fn test_canonical_varbin(#[case] dtype: DType) {
        let mut varbin = VarBinBuilder::<i32>::with_capacity(10);
        varbin.push_null();
        varbin.push_null();
        // inlined value
        varbin.push_value("123456789012".as_bytes());
        // non-inlinable value
        varbin.push_value("1234567890123".as_bytes());
        let varbin = varbin.finish(dtype.clone());

        let canonical = varbin
            .into_array()
            .into_canonical()
            .unwrap()
            .into_varbinview()
            .unwrap();
        assert_eq!(canonical.dtype(), &dtype);

        assert!(!canonical.is_valid(0));
        assert!(!canonical.is_valid(1));

        // First value is inlined (12 bytes)
        assert!(canonical.views()[2].is_inlined());
        assert_eq!(canonical.bytes_at(2).as_slice(), "123456789012".as_bytes());

        // Second value is not inlined (13 bytes)
        assert!(!canonical.views()[3].is_inlined());
        assert_eq!(canonical.bytes_at(3).as_slice(), "1234567890123".as_bytes());
    }
}
