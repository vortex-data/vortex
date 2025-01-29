use arrow_array::builder::make_view;
use arrow_buffer::BooleanBuffer;
use vortex_buffer::{buffer, Buffer, BufferMut};
use vortex_dtype::{match_each_native_ptype, DType, Nullability};
use vortex_error::{vortex_bail, VortexExpect, VortexResult};
use vortex_scalar::{BinaryScalar, BoolScalar, ExtScalar, Utf8Scalar};

use crate::array::constant::ConstantArray;
use crate::array::primitive::PrimitiveArray;
use crate::array::{
    BinaryView, BoolArray, ConstantEncoding, ExtensionArray, NullArray, VarBinViewArray,
};
use crate::arrow::IntoArrowArray;
use crate::validity::Validity;
use crate::vtable::CanonicalVTable;
use crate::{ArrayDType, ArrayLen, Canonical, IntoArrayData, IntoCanonical};

impl CanonicalVTable<ConstantArray> for ConstantEncoding {
    fn into_canonical(&self, array: ConstantArray) -> VortexResult<Canonical> {
        let scalar = &array.scalar();

        let validity = match array.dtype().nullability() {
            Nullability::NonNullable => Validity::NonNullable,
            Nullability::Nullable => match scalar.is_null() {
                true => Validity::AllInvalid,
                false => Validity::AllValid,
            },
        };

        Ok(match array.dtype() {
            DType::Null => Canonical::Null(NullArray::new(array.len())),
            DType::Bool(..) => Canonical::Bool(BoolArray::try_new(
                if BoolScalar::try_from(scalar)?.value().unwrap_or_default() {
                    BooleanBuffer::new_set(array.len())
                } else {
                    BooleanBuffer::new_unset(array.len())
                },
                validity,
            )?),
            DType::Primitive(ptype, ..) => {
                match_each_native_ptype!(ptype, |$P| {
                    Canonical::Primitive(PrimitiveArray::new(
                        if scalar.is_valid() {
                            Buffer::full(
                                $P::try_from(scalar)
                                    .vortex_expect("Couldn't unwrap scalar to primitive"),
                                array.len(),
                            )
                        } else {
                            Buffer::zeroed(array.len())
                        },
                        validity,
                    ))
                })
            }
            DType::Utf8(_) => {
                let value = Utf8Scalar::try_from(scalar)?.value();
                let const_value = value.as_ref().map(|v| v.as_bytes());
                Canonical::VarBinView(canonical_byte_view(
                    const_value,
                    array.dtype(),
                    array.len(),
                )?)
            }
            DType::Binary(_) => {
                let value = BinaryScalar::try_from(scalar)?.value();
                let const_value = value.as_ref().map(|v| v.as_slice());
                Canonical::VarBinView(canonical_byte_view(
                    const_value,
                    array.dtype(),
                    array.len(),
                )?)
            }
            DType::Struct(..) => vortex_bail!("Unsupported scalar type {}", array.dtype()),
            DType::List(..) => vortex_bail!("Unsupported scalar type {}", array.dtype()),
            DType::Extension(ext_dtype) => {
                let s = ExtScalar::try_from(scalar)?;

                let storage_scalar = s.storage();
                let storage_array = ConstantArray::new(storage_scalar, array.len()).into_array();
                ExtensionArray::new(ext_dtype.clone(), storage_array).into_canonical()?
            }
        })
    }
}

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
    use enum_iterator::all;
    use vortex_dtype::half::f16;
    use vortex_dtype::{DType, Nullability, PType};
    use vortex_scalar::Scalar;

    use crate::array::ConstantArray;
    use crate::canonical::IntoArrayVariant;
    use crate::compute::scalar_at;
    use crate::stats::{ArrayStatistics as _, Stat, StatsSet};
    use crate::{ArrayDType, ArrayLen, IntoArrayData as _, IntoCanonical};

    #[test]
    fn test_canonicalize_null() {
        let const_null = ConstantArray::new(Scalar::null(DType::Null), 42);
        let actual = const_null.into_canonical().unwrap().into_null().unwrap();
        assert_eq!(actual.len(), 42);
        assert_eq!(scalar_at(actual, 33).unwrap(), Scalar::null(DType::Null));
    }

    #[test]
    fn test_canonicalize_const_str() {
        let const_array = ConstantArray::new("four".to_string(), 4);

        // Check all values correct.
        let canonical = const_array.into_varbinview().unwrap();

        assert_eq!(canonical.len(), 4);

        for i in 0..=3 {
            assert_eq!(scalar_at(&canonical, i).unwrap(), "four".into());
        }
    }

    #[test]
    fn test_canonicalize_propagates_stats() {
        let scalar = Scalar::bool(true, Nullability::NonNullable);
        let const_array = ConstantArray::new(scalar.clone(), 4).into_array();
        let stats = const_array.statistics().to_set();

        let canonical = const_array.into_canonical().unwrap();
        let canonical_stats = canonical.statistics().to_set();

        let reference = StatsSet::constant(scalar, 4);
        for stat in all::<Stat>() {
            let canonical_stat = canonical_stats
                .get(stat)
                .cloned()
                .map(|sv| Scalar::new(stat.dtype(canonical.dtype()), sv));
            let reference_stat = reference
                .get(stat)
                .cloned()
                .map(|sv| Scalar::new(stat.dtype(canonical.dtype()), sv));
            let original_stat = stats
                .get(stat)
                .cloned()
                .map(|sv| Scalar::new(stat.dtype(canonical.dtype()), sv));
            assert_eq!(canonical_stat, reference_stat);
            assert_eq!(canonical_stat, original_stat);
        }
    }

    #[test]
    fn test_canonicalize_scalar_values() {
        let f16_scalar = Scalar::primitive(f16::from_f32(5.722046e-6), Nullability::NonNullable);
        let scalar = Scalar::new(
            DType::Primitive(PType::F16, Nullability::NonNullable),
            Scalar::primitive(96u8, Nullability::NonNullable).into_value(),
        );
        let const_array = ConstantArray::new(scalar.clone(), 1).into_array();
        let canonical_const = const_array.into_primitive().unwrap();
        assert_eq!(scalar_at(&canonical_const, 0).unwrap(), scalar);
        assert_eq!(scalar_at(&canonical_const, 0).unwrap(), f16_scalar);
    }
}
