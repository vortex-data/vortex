use arrow_array::builder::make_view;
use vortex_buffer::{buffer, Buffer, BufferMut};
use vortex_dtype::{match_each_native_ptype, DType, Nullability};
use vortex_error::VortexResult;

use crate::array::constant::ConstantArray;
use crate::array::{BinaryView, VarBinViewArray};
use crate::builders::{ArrayBuilder, ArrayBuilderExt};
use crate::validity::Validity;
use crate::{ArrayDType, ArrayLen, IntoCanonical};

impl IntoCanonical for ConstantArray {
    fn into_canonical_builder(self, builder: &mut dyn ArrayBuilder) -> VortexResult<()> {
        match self.dtype() {
            DType::Null => builder.as_null_mut().append_nulls(self.len()),
            DType::Bool(_) => {
                let builder = builder.as_bool_mut();
                match self.scalar().as_bool().value() {
                    None => builder.append_nulls(self.len()),
                    Some(b) => builder.append_values(b, self.len()),
                }
            }
            DType::Primitive(ptype, _) => {
                match_each_native_ptype!(ptype, |$P| {
                    let builder = builder.as_primitive_mut::<$P>();
                    match self.scalar().as_primitive().typed_value::<$P>() {
                        None => builder.append_nulls(self.len()),
                        Some(v) => builder.append_values(v, self.len()),
                    }
                })
            }
            DType::Utf8(_) => {
                let builder = builder.as_utf8_mut();
                match self.scalar().as_utf8().value() {
                    None => builder.append_nulls(self.len()),
                    Some(v) => builder.append_values(v, self.len()),
                }
            }
            DType::Binary(_) => {
                let builder = builder.as_binary_mut();
                match self.scalar().as_binary().value() {
                    None => builder.append_nulls(self.len()),
                    Some(v) => builder.append_values(v, self.len()),
                }
            }
            DType::Struct(..) => {
                let builder = builder.as_struct_mut();
                let s = self.scalar();
                for (field_builder, field) in builder.field_builders().zip(s.as_struct().fields()) {
                    field_builder.append_scalar(&field)?;
                }
            }
            DType::List(..) => {
                todo!()
            }
            DType::Extension(_) => {
                todo!()
            }
        }
        Ok(())
    }
}

#[allow(dead_code)]
fn canonical_byte_view(
    scalar_bytes: Option<&[u8]>,
    dtype: &DType,
    len: usize,
) -> VortexResult<VarBinViewArray> {
    match scalar_bytes {
        None => {
            let views = buffer![BinaryView::from(0_u128); len];

            VarBinViewArray::try_new(views, Vec::new(), dtype.clone(), Validity::AllInvalid)
        }
        Some(scalar_bytes) => {
            // Create a view to hold the scalar bytes.
            // If the scalar cannot be inlined, allocate a single buffer large enough to hold it.
            let view = BinaryView::from(make_view(scalar_bytes, 0, 0));
            let mut buffers = Vec::new();
            if scalar_bytes.len() >= BinaryView::MAX_INLINED_SIZE {
                buffers.push(Buffer::copy_from(scalar_bytes));
            }

            // Clone our constant view `len` times.
            // TODO(aduffy): switch this out for a ConstantArray once we
            //   add u128 PType, see https://github.com/spiraldb/vortex/issues/1110
            let mut views = BufferMut::with_capacity_aligned(len, align_of::<u128>().into());
            for _ in 0..len {
                views.push(view);
            }

            let validity = if dtype.nullability() == Nullability::NonNullable {
                Validity::NonNullable
            } else {
                Validity::AllValid
            };

            VarBinViewArray::try_new(views.freeze(), buffers, dtype.clone(), validity)
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_dtype::{DType, Nullability};
    use vortex_scalar::Scalar;

    use crate::array::ConstantArray;
    use crate::compute::scalar_at;
    use crate::stats::{ArrayStatistics as _, StatsSet};
    use crate::{ArrayLen, IntoArrayData as _};

    #[test]
    fn test_canonicalize_null() {
        let const_null = ConstantArray::new(Scalar::null(DType::Null), 42);
        let actual = const_null
            .into_array()
            .into_canonical()
            .unwrap()
            .into_null()
            .unwrap();
        assert_eq!(actual.len(), 42);
        assert_eq!(scalar_at(actual, 33).unwrap(), Scalar::null(DType::Null));
    }

    #[test]
    fn test_canonicalize_const_str() {
        let const_array = ConstantArray::new("four".to_string(), 4);

        // Check all values correct.
        let canonical = const_array
            .into_array()
            .into_canonical()
            .unwrap()
            .into_varbinview()
            .unwrap();

        assert_eq!(canonical.len(), 4);

        for i in 0..=3 {
            assert_eq!(scalar_at(&canonical, i).unwrap(), "four".into(),);
        }
    }

    #[test]
    fn test_canonicalize_propagates_stats() {
        let scalar = Scalar::bool(true, Nullability::NonNullable);
        let const_array = ConstantArray::new(scalar.clone(), 4).into_array();
        let stats = const_array.statistics().to_set();

        let canonical = const_array.into_canonical().unwrap();
        let canonical_stats = canonical.statistics().to_set();

        assert_eq!(canonical_stats, StatsSet::constant(&scalar, 4));
        assert_eq!(canonical_stats, stats);
    }
}
