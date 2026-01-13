// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Conversion from Canonical arrays to Vectors.

use std::sync::Arc;

use vortex_buffer::Buffer;
use vortex_dtype::BigCast;
use vortex_dtype::DType;
use vortex_dtype::PrecisionScale;
use vortex_dtype::match_each_decimal_value_type;
use vortex_dtype::match_each_native_ptype;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_vector::Vector;
use vortex_vector::binaryview::BinaryVector;
use vortex_vector::binaryview::StringVector;
use vortex_vector::bool::BoolVector;
use vortex_vector::decimal::DVector;
use vortex_vector::fixed_size_list::FixedSizeListVector;
use vortex_vector::listview::ListViewVector;
use vortex_vector::null::NullVector;
use vortex_vector::primitive::PVector;
use vortex_vector::struct_::StructVector;

use crate::ArrayRef;
use crate::Canonical;
use crate::Executable;
use crate::ExecutionCtx;

impl Executable for Vector {
    fn execute(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        let canonical = array.execute::<Canonical>(ctx)?;
        canonical.to_vector(ctx)
    }
}

impl Canonical {
    /// Convert a Canonical array to a Vector.
    ///
    /// This is the reverse of `VectorIntoArray` - it takes a fully materialized
    /// canonical array and converts it into the corresponding vector type.
    /// TODO(joe): move over the execute_mask
    pub fn to_vector(self, ctx: &mut ExecutionCtx) -> VortexResult<Vector> {
        Ok(match self {
            Canonical::Null(a) => Vector::Null(NullVector::new(a.len())),
            Canonical::Bool(a) => {
                Vector::Bool(BoolVector::new(a.bit_buffer().clone(), a.validity_mask()))
            }
            Canonical::Primitive(a) => {
                let ptype = a.ptype();
                let validity = a.validity_mask();
                match_each_native_ptype!(ptype, |T| {
                    let buffer = a.as_slice::<T>();
                    Vector::Primitive(PVector::<T>::new(buffer.to_vec().into(), validity).into())
                })
            }
            Canonical::Decimal(a) => {
                // Match on the storage type first to read the buffer
                match_each_decimal_value_type!(a.values_type(), |D| {
                    // Use the smallest type that can represent the precision/scale.
                    // The array may store values in a smaller type (if values fit), but
                    // DVector requires a PrecisionScale that matches its type parameter.
                    let min_value_type =
                        DecimalType::smallest_decimal_value_type(&a.decimal_dtype());
                    match_each_decimal_value_type!(min_value_type, |E| {
                        let decimal_dtype = a.decimal_dtype();
                        let buffer = a.buffer::<D>();
                        let validity_mask = a.validity_mask();

                        // Copy from D to E, possibly widening, possibly narrowing
                        let values = Buffer::<E>::from_trusted_len_iter(buffer.iter().map(|d| {
                            <E as BigCast>::from(*d).vortex_expect("Decimal cast failed")
                        }));

                        // SAFETY: values came from a valid DecimalArray with the same precision/scale
                        Vector::Decimal(
                            unsafe {
                                DVector::<E>::new_unchecked(
                                    PrecisionScale::new_unchecked(
                                        decimal_dtype.precision(),
                                        decimal_dtype.scale(),
                                    ),
                                    values,
                                    validity_mask,
                                )
                            }
                            .into(),
                        )
                    })
                })
            }
            Canonical::VarBinView(a) => {
                let validity = a.validity_mask();
                match a.dtype() {
                    DType::Utf8(_) => {
                        let views = a.views().clone();
                        // Convert Arc<[ByteBuffer]> to Arc<Box<[ByteBuffer]>>
                        let buffers: Box<[_]> = a.buffers().iter().cloned().collect();
                        Vector::String(unsafe {
                            StringVector::new_unchecked(views, Arc::new(buffers), validity)
                        })
                    }
                    DType::Binary(_) => {
                        let views = a.views().clone();
                        // Convert Arc<[ByteBuffer]> to Arc<Box<[ByteBuffer]>>
                        let buffers: Box<[_]> = a.buffers().iter().cloned().collect();
                        Vector::Binary(unsafe {
                            BinaryVector::new_unchecked(views, Arc::new(buffers), validity)
                        })
                    }
                    _ => unreachable!("VarBinView must be Utf8 or Binary"),
                }
            }
            Canonical::List(a) => {
                let (elements, offsets, sizes, validity) = a.into_parts();

                let validity = validity.to_array(offsets.len()).execute::<Mask>(ctx)?;
                let elements_vector = elements.execute::<Vector>(ctx)?;
                let offsets = offsets.execute::<Canonical>(ctx)?.into_primitive();
                let sizes = sizes.execute::<Canonical>(ctx)?.into_primitive();
                let offsets_ptype = offsets.ptype();
                let sizes_ptype = sizes.ptype();

                match_each_native_ptype!(offsets_ptype, |O| {
                    match_each_native_ptype!(sizes_ptype, |S| {
                        let offsets_vec = PVector::<O>::new(
                            offsets.as_slice::<O>().to_vec().into(),
                            offsets.validity_mask(),
                        );
                        let sizes_vec = PVector::<S>::new(
                            sizes.as_slice::<S>().to_vec().into(),
                            sizes.validity_mask(),
                        );
                        Vector::List(unsafe {
                            ListViewVector::new_unchecked(
                                Arc::new(elements_vector),
                                offsets_vec.into(),
                                sizes_vec.into(),
                                validity,
                            )
                        })
                    })
                })
            }
            Canonical::FixedSizeList(a) => {
                let validity = a.validity_mask();
                let list_size = a.list_size();
                let elements_vector = a.elements().clone().execute::<Vector>(ctx)?;
                Vector::FixedSizeList(unsafe {
                    FixedSizeListVector::new_unchecked(
                        Arc::new(elements_vector),
                        list_size,
                        validity,
                    )
                })
            }
            Canonical::Struct(a) => {
                let validity = a.validity_mask();
                let mut fields = Vec::with_capacity(a.fields().len());
                for f in a.fields().iter().cloned() {
                    fields.push(f.execute::<Vector>(ctx)?);
                }
                let fields: Box<[Vector]> = fields.into_boxed_slice();
                Vector::Struct(StructVector::new(Arc::new(fields), validity))
            }
            Canonical::Extension(a) => {
                // For extension arrays, convert the underlying storage
                a.storage().clone().execute::<Vector>(ctx)?
            }
        })
    }
}
