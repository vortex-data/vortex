// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Conversion from Canonical arrays to Vectors.

use std::sync::Arc;

use vortex_dtype::DType;
use vortex_dtype::PrecisionScale;
use vortex_dtype::match_each_decimal_value_type;
use vortex_dtype::match_each_native_ptype;
use vortex_error::VortexResult;
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

use crate::Canonical;
use crate::ExecutionCtx;
use crate::VectorExecutor;

impl Canonical {
    /// Convert a Canonical array to a Vector.
    ///
    /// This is the reverse of `VectorIntoArray` - it takes a fully materialized
    /// canonical array and converts it into the corresponding vector type.
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
                let values_type = a.values_type();
                let dec_dtype = a.decimal_dtype();
                let validity = a.validity_mask();
                match_each_decimal_value_type!(values_type, |D| {
                    let buffer = a.buffer::<D>();
                    let ps = PrecisionScale::<D>::new(dec_dtype.precision(), dec_dtype.scale());
                    Vector::Decimal(DVector::<D>::new(ps, buffer, validity).into())
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
                let validity = a.validity_mask();
                let elements_vector = a.elements().execute(ctx)?.to_vector(ctx)?;
                let offsets = a.offsets().execute(ctx)?.into_primitive();
                let sizes = a.sizes().execute(ctx)?.into_primitive();
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
                let elements_vector = a.elements().execute(ctx)?.to_vector(ctx)?;
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
                for f in a.fields().iter() {
                    fields.push(f.execute(ctx)?.to_vector(ctx)?);
                }
                let fields: Box<[Vector]> = fields.into_boxed_slice();
                Vector::Struct(StructVector::new(Arc::new(fields), validity))
            }
            Canonical::Extension(a) => {
                // For extension arrays, convert the underlying storage
                a.storage().execute(ctx)?.to_vector(ctx)?
            }
        })
    }
}
