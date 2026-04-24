// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::sync::Arc;

use arrow_schema::DataType;
use arrow_schema::DataType::*;
use arrow_schema::Field;
use itertools::Itertools as _;
use noodles_vcf::header::record::value::Map;
use noodles_vcf::header::record::value::map::Info;
use noodles_vcf::header::record::value::map::info::Number;
use noodles_vcf::header::record::value::map::info::Type;
use noodles_vcf::variant::record::info::field::value::Array;
use noodles_vcf::variant::record::info::field::value::Value;
use noodles_vcf::variant::record::info::field::value::array::Values;
use noodles_vcf::variant::record::samples::series::value::Array as EntryArray;
use noodles_vcf::variant::record::samples::series::value::Value as EntryValue;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;

use crate::statpopgen::builder::InfoArrayBuilder;

pub fn builder_from_info(info: &Map<Info>) -> InfoArrayBuilder {
    match (info.number(), info.ty()) {
        (Number::Count(1), Type::Integer) => InfoArrayBuilder::Integer(Default::default()),
        (Number::Count(1), Type::Float) => InfoArrayBuilder::Float(Default::default()),
        (Number::Count(0), Type::Flag) => InfoArrayBuilder::Flag(Default::default()),
        (Number::Count(1), Type::Character) => todo!(),
        (Number::Count(1), Type::String) => InfoArrayBuilder::String(Default::default()),
        (_, Type::Integer) => InfoArrayBuilder::ListInteger(Default::default()),
        (_, Type::Float) => InfoArrayBuilder::ListFloat(Default::default()),
        (_, Type::Flag) => todo!(),
        (_, Type::Character) => todo!(),
        (_, Type::String) => InfoArrayBuilder::ListString(Default::default()),
    }
}

pub fn list(x: DataType) -> DataType {
    List(Arc::new(Field::new("item", x, true)))
}

pub fn data_type_from_info(info: &Map<Info>) -> DataType {
    match (info.number(), info.ty()) {
        (Number::Count(1), Type::Integer) => Int32,
        (Number::Count(1), Type::Float) => Float32,
        (Number::Count(0), Type::Flag) => Boolean,
        (Number::Count(1), Type::Character) => todo!(),
        (Number::Count(1), Type::String) => Utf8,
        (_, Type::Integer) => list(Int32),
        (_, Type::Float) => list(Float32),
        (_, Type::Flag) => todo!(),
        (_, Type::Character) => todo!(),
        (_, Type::String) => list(Utf8),
    }
}

pub fn value_int32(v: Option<Value>) -> VortexResult<Option<i32>> {
    Ok(match v {
        None => None,

        Some(Value::Integer(x)) => Some(x),
        _ => vortex_bail!("expected int32 {:?}", v),
    })
}

pub fn value_float32(v: Option<Value>) -> VortexResult<Option<f32>> {
    Ok(match v {
        None => None,
        Some(Value::Float(x)) => Some(x),
        _ => vortex_bail!("expected f64 {:?}", v),
    })
}

pub fn value_string(v: Option<Value>) -> VortexResult<Option<Cow<str>>> {
    Ok(match v {
        None => None,
        Some(Value::String(x)) => Some(x),
        _ => vortex_bail!("expected string {:?}", v),
    })
}

pub fn value_boolean(v: Option<Value>) -> VortexResult<bool> {
    Ok(match v {
        None => false,
        Some(Value::Flag) => true,
        _ => vortex_bail!("expected bool {:?}", v),
    })
}

pub fn value_list_int32<'a>(
    v: Option<Value<'a>>,
) -> VortexResult<Option<Box<dyn Values<'a, i32> + 'a>>> {
    Ok(match v {
        None => None,
        Some(Value::Array(a)) => match a {
            Array::Integer(values) => Some(values),
            v => vortex_bail!("expected int32 {:?}", v),
        },
        v => vortex_bail!("expected int32 {:?}", v),
    })
}

pub fn value_list_float32<'a>(
    v: Option<Value<'a>>,
) -> VortexResult<Option<Box<dyn Values<'a, f32> + 'a>>> {
    Ok(match v {
        None => None,
        Some(Value::Array(a)) => match a {
            Array::Float(values) => Some(values),
            v => vortex_bail!("expected int32 {:?}", v),
        },
        v => vortex_bail!("expected f64 {:?}", v),
    })
}

pub fn value_list_string<'a>(
    v: Option<Value<'a>>,
) -> VortexResult<Option<Box<dyn Values<'a, Cow<'a, str>> + 'a>>> {
    Ok(match v {
        None => None,
        Some(Value::Array(a)) => match a {
            Array::String(values) => Some(values),
            v => vortex_bail!("expected int32 {:?}", v),
        },
        v => vortex_bail!("expected string {:?}", v),
    })
}

pub fn parse_genotype(gt: Option<EntryValue>) -> VortexResult<Option<u64>> {
    let Some(gt) = gt else {
        return Ok(None);
    };
    let EntryValue::Genotype(gt) = gt else {
        vortex_bail!("expected genotype {:?}", gt)
    };
    match gt
        .iter()
        .process_results(|iter| iter.map(|x| x.0).collect::<Vec<_>>())?[..]
    {
        [None, None] => Ok(None),
        [Some(l), Some(r)] => Ok(Some(l as u64 + r as u64)),
        _ => vortex_bail!("wtf {:?}", gt),
    }
}

pub fn parse_int32_format(x: Option<EntryValue>) -> VortexResult<Option<i32>> {
    let Some(x) = x else {
        return Ok(None);
    };
    let EntryValue::Integer(x) = x else {
        vortex_bail!("expected int32 {:?}", x)
    };
    Ok(Some(x))
}

pub fn parse_pgt_format(x: Option<EntryValue>) -> VortexResult<Option<i32>> {
    let Some(x) = x else {
        return Ok(None);
    };
    // DK: bioinfomatics is a dumpster fire
    Ok(match x {
        EntryValue::String(x) if x == "./." || x == "." => None,
        EntryValue::String(x) if x == "0|0" => Some(0),
        EntryValue::String(x) if x == "0|1" => Some(1),
        EntryValue::String(x) if x == "1|0" => Some(2),
        EntryValue::String(x) if x == "1|1" => Some(3),
        _ => vortex_bail!("expected biallelic phased genotype {:?}", x),
    })
}

pub fn parse_string_format<'a>(x: Option<EntryValue<'a>>) -> VortexResult<Option<Cow<'a, str>>> {
    let Some(x) = x else {
        return Ok(None);
    };
    match x {
        EntryValue::String(x) => Ok(Some(x)),
        _ => vortex_bail!("expected string {:?}", x),
    }
}

pub fn parse_list_int32_format(x: Option<EntryValue>) -> VortexResult<Option<Vec<Option<i32>>>> {
    let Some(x) = x else {
        return Ok(None);
    };
    match x {
        EntryValue::Array(x) => match x {
            EntryArray::Integer(values) => Ok(Some(
                values
                    .iter()
                    .map(|x| x.expect("no io errors"))
                    .collect::<Vec<_>>(),
            )),
            _ => vortex_bail!("expected list int32 {:?}", x),
        },
        _ => vortex_bail!("expected list list int32 {:?}", x),
    }
}
