use arrow_buffer::BooleanBuffer;
use vortex_buffer::{Buffer, BufferMut, buffer};
use vortex_dtype::{DType, Nullability, PType, match_each_native_ptype};
use vortex_error::{VortexExpect, VortexResult};
use vortex_scalar::{
    BinaryScalar, BoolScalar, DecimalValue, ExtScalar, ListScalar, Scalar, ScalarValue,
    StructScalar, Utf8Scalar,
};

use crate::array::ArrayCanonicalImpl;
use crate::arrays::constant::ConstantArray;
use crate::arrays::primitive::PrimitiveArray;
use crate::arrays::{
    BinaryView, BoolArray, DecimalArray, ExtensionArray, ListArray, NullArray, StructArray,
    VarBinViewArray, precision_to_storage_size,
};
use crate::builders::{ArrayBuilderExt, builder_with_capacity};
use crate::validity::Validity;
use crate::{Array, Canonical, IntoArray, match_each_decimal_value, match_each_decimal_value_type};

impl ArrayCanonicalImpl for ConstantArray {
    fn _to_canonical(&self) -> VortexResult<Canonical> {
        let scalar = self.scalar();

        let validity = match self.dtype().nullability() {
            Nullability::NonNullable => Validity::NonNullable,
            Nullability::Nullable => match scalar.is_null() {
                true => Validity::AllInvalid,
                false => Validity::AllValid,
            },
        };

        Ok(match self.dtype() {
            DType::Null => Canonical::Null(NullArray::new(self.len())),
            DType::Bool(..) => Canonical::Bool(BoolArray::new(
                if BoolScalar::try_from(scalar)?.value().unwrap_or_default() {
                    BooleanBuffer::new_set(self.len())
                } else {
                    BooleanBuffer::new_unset(self.len())
                },
                validity,
            )),
            DType::Primitive(ptype, ..) => {
                match_each_native_ptype!(ptype, |$P| {
                    Canonical::Primitive(PrimitiveArray::new(
                        if scalar.is_valid() {
                            Buffer::full(
                                $P::try_from(scalar)
                                    .vortex_expect("Couldn't unwrap scalar to primitive"),
                                self.len(),
                            )
                        } else {
                            Buffer::zeroed(self.len())
                        },
                        validity,
                    ))
                })
            }
            DType::Decimal(decimal_type, ..) => {
                let size = precision_to_storage_size(decimal_type);
                let decimal = scalar.as_decimal();
                let Some(value) = decimal.decimal_value() else {
                    let all_null = match_each_decimal_value_type!(size, |$D| {
                       DecimalArray::new(
                                Buffer::<$D>::zeroed(self.len()),
                                *decimal_type,
                                Validity::AllInvalid,
                            )
                    });
                    return Ok(Canonical::Decimal(all_null));
                };

                let decimal_array = match_each_decimal_value!(value, |$V| {
                   DecimalArray::new(
                        Buffer::full(*$V, self.len()),
                        *decimal_type,
                        Validity::AllValid,
                    )
                });
                Canonical::Decimal(decimal_array)
            }
            DType::Utf8(_) => {
                let value = Utf8Scalar::try_from(scalar)?.value();
                let const_value = value.as_ref().map(|v| v.as_bytes());
                Canonical::VarBinView(canonical_byte_view(const_value, self.dtype(), self.len())?)
            }
            DType::Binary(_) => {
                let value = BinaryScalar::try_from(scalar)?.value();
                let const_value = value.as_ref().map(|v| v.as_slice());
                Canonical::VarBinView(canonical_byte_view(const_value, self.dtype(), self.len())?)
            }
            DType::Struct(struct_dtype, _) => {
                let value = StructScalar::try_from(scalar)?;
                let fields = value.fields().map(|fields| {
                    fields
                        .into_iter()
                        .map(|s| ConstantArray::new(s, self.len()).into_array())
                        .collect::<Vec<_>>()
                });
                Canonical::Struct(StructArray::try_new_with_dtype(
                    fields.unwrap_or_default(),
                    struct_dtype.clone(),
                    self.len(),
                    validity,
                )?)
            }
            DType::List(..) => {
                let value = ListScalar::try_from(scalar)?;
                Canonical::List(canonical_list_array(
                    value.elements(),
                    value.element_dtype(),
                    value.dtype().nullability(),
                    self.len(),
                )?)
            }
            DType::Extension(ext_dtype) => {
                let s = ExtScalar::try_from(scalar)?;

                let storage_scalar = s.storage();
                let storage_self = ConstantArray::new(storage_scalar, self.len()).into_array();
                Canonical::Extension(ExtensionArray::new(ext_dtype.clone(), storage_self))
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
            let view = BinaryView::make_view(scalar_bytes, 0, 0);
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

            VarBinViewArray::try_new(
                views.freeze(),
                buffers,
                dtype.clone(),
                Validity::from(dtype.nullability()),
            )
        }
    }
}

fn canonical_list_array(
    values: Option<Vec<Scalar>>,
    element_dtype: &DType,
    list_nullability: Nullability,
    len: usize,
) -> VortexResult<ListArray> {
    match values {
        None => ListArray::try_new(
            Canonical::empty(element_dtype).into_array(),
            ConstantArray::new(
                Scalar::new(
                    DType::Primitive(PType::U64, Nullability::NonNullable),
                    ScalarValue::from(0),
                ),
                len + 1,
            )
            .into_array(),
            Validity::AllInvalid,
        ),
        Some(vs) => {
            let mut elements_builder = builder_with_capacity(element_dtype, len * vs.len());
            for _ in 0..len {
                for v in &vs {
                    elements_builder.append_scalar(v)?;
                }
            }
            let offsets = if vs.is_empty() {
                Buffer::zeroed(len + 1)
            } else {
                (0..=len * vs.len())
                    .step_by(vs.len())
                    .map(|i| i as u64)
                    .collect::<Buffer<_>>()
            };

            ListArray::try_new(
                elements_builder.finish(),
                offsets.into_array(),
                Validity::from(list_nullability),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use enum_iterator::all;
    use vortex_dtype::half::f16;
    use vortex_dtype::{DType, Nullability, PType};
    use vortex_scalar::Scalar;

    use crate::array::Array;
    use crate::arrays::ConstantArray;
    use crate::canonical::ToCanonical;
    use crate::compute::scalar_at;
    use crate::stats::{Stat, StatsProviderExt, StatsSet};

    #[test]
    fn test_canonicalize_null() {
        let const_null = ConstantArray::new(Scalar::null(DType::Null), 42);
        let actual = const_null.to_null().unwrap();
        assert_eq!(actual.len(), 42);
        assert_eq!(scalar_at(&actual, 33).unwrap(), Scalar::null(DType::Null));
    }

    #[test]
    fn test_canonicalize_const_str() {
        let const_array = ConstantArray::new("four".to_string(), 4);

        // Check all values correct.
        let canonical = const_array.to_varbinview().unwrap();

        assert_eq!(canonical.len(), 4);

        for i in 0..=3 {
            assert_eq!(scalar_at(&canonical, i).unwrap(), "four".into());
        }
    }

    #[test]
    fn test_canonicalize_propagates_stats() {
        let scalar = Scalar::bool(true, Nullability::NonNullable);
        let const_array = ConstantArray::new(scalar.clone(), 4).into_array();
        let stats = const_array.statistics().to_owned();

        let canonical = const_array.to_canonical().unwrap();
        let canonical_stats = canonical.as_ref().statistics().to_owned();

        let reference = StatsSet::constant(scalar, 4);
        for stat in all::<Stat>() {
            let canonical_stat =
                canonical_stats.get_scalar(stat, &stat.dtype(canonical.as_ref().dtype()).unwrap());
            let reference_stat =
                reference.get_scalar(stat, &stat.dtype(canonical.as_ref().dtype()).unwrap());
            let original_stat =
                stats.get_scalar(stat, &stat.dtype(canonical.as_ref().dtype()).unwrap());
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
        let canonical_const = const_array.to_primitive().unwrap();
        assert_eq!(scalar_at(&canonical_const, 0).unwrap(), scalar);
        assert_eq!(scalar_at(&canonical_const, 0).unwrap(), f16_scalar);
    }

    #[test]
    fn test_canonicalize_lists() {
        let list_scalar = Scalar::list(
            Arc::new(DType::Primitive(PType::U64, Nullability::NonNullable)),
            vec![1u64.into(), 2u64.into()],
            Nullability::NonNullable,
        );
        let const_array = ConstantArray::new(list_scalar, 2).into_array();
        let canonical_const = const_array.to_list().unwrap();
        assert_eq!(
            canonical_const
                .elements()
                .to_primitive()
                .unwrap()
                .as_slice::<u64>(),
            [1u64, 2, 1, 2]
        );
        assert_eq!(
            canonical_const
                .offsets()
                .to_primitive()
                .unwrap()
                .as_slice::<u64>(),
            [0u64, 2, 4]
        );
    }

    #[test]
    fn test_canonicalize_empty_list() {
        let list_scalar = Scalar::list(
            Arc::new(DType::Primitive(PType::U64, Nullability::NonNullable)),
            vec![],
            Nullability::NonNullable,
        );
        let const_array = ConstantArray::new(list_scalar, 2).into_array();
        let canonical_const = const_array.to_list().unwrap();
        assert!(
            canonical_const
                .elements()
                .to_primitive()
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            canonical_const
                .offsets()
                .to_primitive()
                .unwrap()
                .as_slice::<u64>(),
            [0u64, 0, 0]
        );
    }

    #[test]
    fn test_canonicalize_null_list() {
        let list_scalar = Scalar::null(DType::List(
            Arc::new(DType::Primitive(PType::U64, Nullability::NonNullable)),
            Nullability::Nullable,
        ));
        let const_array = ConstantArray::new(list_scalar, 2).into_array();
        let canonical_const = const_array.to_list().unwrap();
        assert!(
            canonical_const
                .elements()
                .to_primitive()
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            canonical_const
                .offsets()
                .to_primitive()
                .unwrap()
                .as_slice::<u64>(),
            [0u64, 0, 0]
        );
    }
}
