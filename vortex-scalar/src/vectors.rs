// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Conversion logic from this "legacy" scalar crate to Vortex Vector scalars.

use std::sync::Arc;

use vortex_buffer::Buffer;
use vortex_dtype::{
    DType, DecimalType, PrecisionScale, match_each_decimal_value_type, match_each_native_ptype,
};
use vortex_error::VortexExpect;
use vortex_mask::Mask;
use vortex_vector::binaryview::{BinaryScalar, StringScalar};
use vortex_vector::bool::BoolScalar;
use vortex_vector::decimal::DScalar;
use vortex_vector::fixed_size_list::{FixedSizeListScalar, FixedSizeListVector};
use vortex_vector::listview::{ListViewScalar, ListViewVector, ListViewVectorMut};
use vortex_vector::null::NullScalar;
use vortex_vector::primitive::{PScalar, PVector};
use vortex_vector::struct_::{StructScalar, StructVector};
use vortex_vector::{VectorMut, VectorMutOps};

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
                        StructScalar::new(StructVector::new(Arc::new(fields), Mask::new_false(1)))
                            .into()
                    }
                }
            }
            DType::Extension(_) => self.as_extension().storage().to_vector_scalar(),
        }
    }
}
