// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Conversion logic from this "legacy" scalar crate to Vortex Vector scalars.

use std::ops::Deref;
use std::sync::Arc;

use itertools::Itertools;
use vortex_buffer::Buffer;
use vortex_dtype::match_each_decimal_value_type;
use vortex_dtype::match_each_native_ptype;
use vortex_dtype::DType;
use vortex_dtype::DecimalType;
use vortex_dtype::NativeDecimalType;
use vortex_dtype::PTypeDowncastExt;
use vortex_dtype::PrecisionScale;
use vortex_error::vortex_ensure;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_vector::binaryview::BinaryScalar;
use vortex_vector::binaryview::StringScalar;
use vortex_vector::bool::BoolScalar;
use vortex_vector::decimal::DScalar;
use vortex_vector::fixed_size_list::FixedSizeListScalar;
use vortex_vector::fixed_size_list::FixedSizeListVector;
use vortex_vector::listview::ListViewScalar;
use vortex_vector::listview::ListViewVector;
use vortex_vector::listview::ListViewVectorMut;
use vortex_vector::null::NullScalar;
use vortex_vector::primitive::PScalar;
use vortex_vector::primitive::PVector;
use vortex_vector::struct_::StructScalar;
use vortex_vector::struct_::StructVector;
use vortex_vector::VectorMut;
use vortex_vector::VectorMutOps;
use vortex_vector::VectorOps;

use crate::DecimalValue;
use crate::Scalar;

impl Scalar {
    /// Convert the `vortex-scalar` [`Scalar`] into a `vortex-vector` [`vortex_vector::Scalar`].
    pub fn to_vector_scalar(&self) -> vortex_vector::Scalar {
        match self.dtype() {
            DType::Null => NullScalar.into(),
            DType::Bool(_) => BoolScalar::new(self.as_bool().value()).into(),
            DType::Primitive(ptype, _) => {
                match_each_native_ptype!(ptype, |T| {
                    PScalar::new(self.as_primitive().typed_value::<T>()).into()
                })
            }
            DType::Decimal(dec_dtype, _) => {
                let dscalar = self.as_decimal();
                let dec_type = DecimalType::smallest_decimal_value_type(dec_dtype);
                match_each_decimal_value_type!(dec_type, |D| {
                    let ps = PrecisionScale::<D>::new(dec_dtype.precision(), dec_dtype.scale());
                    DScalar::maybe_new(
                        ps,
                        dscalar
                            .decimal_value()
                            .map(|d| d.cast::<D>().vortex_expect("Failed to cast decimal value")),
                    )
                    .vortex_expect("Failed to create decimal scalar")
                    .into()
                })
            }
            DType::Utf8(_) => StringScalar::new(self.as_utf8().value()).into(),
            DType::Binary(_) => BinaryScalar::new(self.as_binary().value()).into(),
            DType::List(elems_dtype, _) => {
                let lscalar = self.as_list();
                match lscalar.elements() {
                    None => {
                        let mut list_view = ListViewVectorMut::with_capacity(elems_dtype, 1);
                        list_view.append_nulls(1);
                        ListViewScalar::new(list_view.freeze()).into()
                    }
                    Some(elements) => {
                        // If the list elements are non-null, we convert each one accordingly
                        // and append it to the new list view.
                        let mut new_elements =
                            VectorMut::with_capacity(elems_dtype, elements.len());
                        for element in &elements {
                            let element_scalar = element.to_vector_scalar();
                            new_elements.append_scalars(&element_scalar, 1);
                        }

                        let offsets =
                            PVector::<u64>::new(Buffer::from_iter([0]), Mask::new_true(1));
                        let sizes = PVector::<u64>::new(
                            Buffer::from_iter([elements.len() as u64]),
                            Mask::new_true(1),
                        );

                        // Create the length-1 vector holding the list scalar.
                        let list_view_vector = ListViewVector::new(
                            Arc::new(new_elements.freeze()),
                            offsets.into(),
                            sizes.into(),
                            Mask::new_true(1),
                        );

                        ListViewScalar::new(list_view_vector).into()
                    }
                }
            }
            DType::FixedSizeList(elems_dtype, size, _) => {
                let lscalar = self.as_list();
                match lscalar.elements() {
                    None => {
                        let mut elements = VectorMut::with_capacity(elems_dtype, *size as usize);
                        elements.append_zeros(*size as usize);

                        FixedSizeListScalar::new(FixedSizeListVector::new(
                            Arc::new(elements.freeze()),
                            *size,
                            Mask::new_false(1),
                        ))
                        .into()
                    }
                    Some(element_scalars) => {
                        let mut elements = VectorMut::with_capacity(elems_dtype, *size as usize);
                        for element_scalar in &element_scalars {
                            elements.append_scalars(&element_scalar.to_vector_scalar(), 1);
                        }
                        FixedSizeListScalar::new(FixedSizeListVector::new(
                            Arc::new(elements.freeze()),
                            *size,
                            Mask::new_true(1),
                        ))
                        .into()
                    }
                }
            }
            DType::Struct(fields, _) => {
                let scalar = self.as_struct();

                match scalar.fields() {
                    None => {
                        // Null struct scalar, we still need a length-1 vector for each field.
                        let fields = fields
                            .fields()
                            .map(|dtype| {
                                let mut field_vec = VectorMut::with_capacity(&dtype, 1);
                                field_vec.append_zeros(1);
                                field_vec.freeze()
                            })
                            .collect();
                        StructScalar::new(StructVector::new(Arc::new(fields), Mask::new_false(1)))
                            .into()
                    }
                    Some(field_scalars) => {
                        let fields = field_scalars
                            .map(|scalar| {
                                let mut field_vec = VectorMut::with_capacity(scalar.dtype(), 1);
                                field_vec.append_scalars(&scalar.to_vector_scalar(), 1);
                                field_vec.freeze()
                            })
                            .collect();
                        StructScalar::new(StructVector::new(Arc::new(fields), Mask::new_true(1)))
                            .into()
                    }
                }
            }
            DType::Extension(_) => self.as_extension().storage().to_vector_scalar(),
        }
    }

    /// Convert a `vortex-vector` [`vortex_vector::Scalar`] into a `vortex-scalar` [`Scalar`].
    pub fn from_vector_scalar(scalar: vortex_vector::Scalar, dtype: &DType) -> VortexResult<Self> {
        Ok(match dtype {
            DType::Null => Scalar::null(DType::Null),
            DType::Bool(n) => match scalar.as_bool().value() {
                None => {
                    vortex_ensure!(
                        n.is_nullable(),
                        "Cannot create null scalar for non-nullable dtype"
                    );
                    Scalar::null(dtype.clone())
                }
                Some(b) => Scalar::bool(b, *n),
            },
            DType::Primitive(ptype, n) => {
                match_each_native_ptype!(ptype, |T| {
                    let pscalar = scalar.into_primitive().downcast::<T>();
                    match pscalar.value() {
                        None => {
                            vortex_ensure!(
                                n.is_nullable(),
                                "Cannot create null scalar for non-nullable dtype"
                            );
                            Scalar::null(dtype.clone())
                        }
                        Some(v) => Scalar::primitive(v, *n),
                    }
                })
            }
            DType::Decimal(dec_type, n) => {
                let dec_scalar = scalar.into_decimal();
                match_each_decimal_value_type!(
                    DecimalType::smallest_decimal_value_type(dec_type),
                    |D| {
                        let dscalar = <D as NativeDecimalType>::downcast(dec_scalar);
                        match dscalar.value() {
                            None => {
                                vortex_ensure!(
                                    n.is_nullable(),
                                    "Cannot create null scalar for non-nullable dtype"
                                );
                                Scalar::null(dtype.clone())
                            }
                            Some(v) => Scalar::decimal(DecimalValue::from(v), *dec_type, *n),
                        }
                    }
                )
            }
            DType::Utf8(n) => match scalar.as_string().value() {
                None => {
                    vortex_ensure!(
                        n.is_nullable(),
                        "Cannot create null scalar for non-nullable dtype"
                    );
                    Scalar::null(dtype.clone())
                }
                Some(s) => Scalar::utf8(s.clone(), *n),
            },
            DType::Binary(n) => match scalar.as_binary().value() {
                None => {
                    vortex_ensure!(
                        n.is_nullable(),
                        "Cannot create null scalar for non-nullable dtype"
                    );
                    Scalar::null(dtype.clone())
                }
                Some(b) => Scalar::binary(b.clone(), *n),
            },
            DType::List(elem_dtype, n) => {
                let elements = scalar.as_list().value().elements();
                Scalar::list(
                    elem_dtype.clone(),
                    (0..elements.len())
                        .map(|idx| elements.scalar_at(idx))
                        .map(|scalar| Scalar::from_vector_scalar(scalar, elem_dtype.deref()))
                        .try_collect()?,
                    *n,
                )
            }
            DType::FixedSizeList(elem_dtype, size, n) => {
                let scalar = scalar.into_fixed_size_list();
                let elements = scalar.value().elements();
                vortex_ensure!(scalar.value().element_size() == *size);

                Scalar::fixed_size_list(
                    DType::FixedSizeList(elem_dtype.clone(), *size, *n),
                    (0..*size as usize)
                        .map(|idx| {
                            Scalar::from_vector_scalar(elements.scalar_at(idx), elem_dtype.deref())
                        })
                        .try_collect()?,
                    *n,
                )
            }
            DType::Struct(fields, n) => Scalar::struct_(
                DType::Struct(fields.clone(), *n),
                fields
                    .fields()
                    .zip(scalar.into_struct().fields())
                    .map(|(field_dtype, field_scalar)| {
                        Scalar::from_vector_scalar(field_scalar, &field_dtype)
                    })
                    .try_collect()?,
            ),
            DType::Extension(ext_dtype) => Scalar::extension(
                ext_dtype.clone(),
                Scalar::from_vector_scalar(scalar, ext_dtype.storage_dtype())?,
            ),
        })
    }
}
