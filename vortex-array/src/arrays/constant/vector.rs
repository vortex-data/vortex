// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_dtype::DecimalType;
use vortex_dtype::PrecisionScale;
use vortex_dtype::match_each_decimal_value_type;
use vortex_dtype::match_each_native_ptype;
use vortex_error::VortexExpect;
use vortex_scalar::BinaryScalar;
use vortex_scalar::BoolScalar;
use vortex_scalar::DecimalScalar;
use vortex_scalar::PrimitiveScalar;
use vortex_scalar::Scalar;
use vortex_scalar::Utf8Scalar;
use vortex_vector::VectorMut;
use vortex_vector::VectorMutOps;
use vortex_vector::binaryview::BinaryVectorMut;
use vortex_vector::binaryview::StringVectorMut;
use vortex_vector::bool::BoolVectorMut;
use vortex_vector::decimal::DVectorMut;
use vortex_vector::decimal::DecimalVectorMut;
use vortex_vector::null::NullVectorMut;
use vortex_vector::primitive::PVectorMut;
use vortex_vector::primitive::PrimitiveVectorMut;

pub(super) fn to_vector(scalar: Scalar, len: usize) -> VectorMut {
    match scalar.dtype() {
        DType::Null => NullVectorMut::new(len).into(),
        DType::Bool(_) => to_vector_bool(scalar.as_bool(), len).into(),
        DType::Primitive(..) => to_vector_primitive(scalar.as_primitive(), len).into(),
        DType::Decimal(..) => to_vector_decimal(scalar.as_decimal(), len).into(),
        DType::Utf8(_) => to_vector_utf8(scalar.as_utf8(), len).into(),
        DType::Binary(_) => to_vector_binary(scalar.as_binary(), len).into(),
        DType::List(..) => unimplemented!("List constant vectorization"),
        DType::FixedSizeList(..) => unimplemented!("FixedSizeList constant vectorization"),
        DType::Struct(..) => unimplemented!("Struct constant vectorization"),
        DType::Extension(_) => to_vector(scalar.as_extension().storage(), len),
    }
}

fn to_vector_bool(scalar: BoolScalar, len: usize) -> BoolVectorMut {
    let mut vec = BoolVectorMut::with_capacity(len);
    match scalar.value() {
        Some(v) => vec.append_values(v, len),
        None => vec.append_nulls(len),
    }
    vec
}

fn to_vector_primitive(scalar: PrimitiveScalar, len: usize) -> PrimitiveVectorMut {
    match_each_native_ptype!(scalar.ptype(), |T| {
        let mut vec = PVectorMut::<T>::with_capacity(len);
        match scalar.typed_value::<T>() {
            Some(v) => vec.append_values(v, len),
            None => vec.append_nulls(len),
        }
        vec.into()
    })
}

fn to_vector_decimal(scalar: DecimalScalar, len: usize) -> DecimalVectorMut {
    let decimal_dtype = scalar
        .dtype()
        .as_decimal_opt()
        .vortex_expect("Decimal scalar must have a decimal type");
    let decimal_type = DecimalType::smallest_decimal_value_type(decimal_dtype);

    match_each_decimal_value_type!(decimal_type, |D| {
        let ps = PrecisionScale::<D>::new(decimal_dtype.precision(), decimal_dtype.scale());
        let mut vec = DVectorMut::<D>::with_capacity(ps, len);
        match scalar.decimal_value() {
            Some(v) => vec
                .try_append_n(v.cast::<D>().vortex_expect("known to fit"), len)
                .vortex_expect("known to fit"),
            None => vec.append_nulls(len),
        }
        vec.into()
    })
}

fn to_vector_utf8(scalar: Utf8Scalar, len: usize) -> StringVectorMut {
    let mut vec = StringVectorMut::with_capacity(len);
    match scalar.value() {
        Some(v) => vec.append_values(v.as_ref(), len),
        None => vec.append_nulls(len),
    }
    vec
}

fn to_vector_binary(scalar: BinaryScalar, len: usize) -> BinaryVectorMut {
    let mut vec = BinaryVectorMut::with_capacity(len);
    match scalar.value() {
        Some(v) => vec.append_values(v.as_ref(), len),
        None => vec.append_nulls(len),
    }
    vec
}
