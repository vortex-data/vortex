// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An implementation of an ST_Contains expression type as a ScalarFn.
//!
//! The Vectors don't seem to be complete enough to use ScalarFn for non-trivial things
//! so for now this is unused.

use crate::expr::functions::{ArgName, Arity, EmptyOptions, ExecutionArgs, FunctionId, VTable};
use geo::Contains;
use geo_types::Geometry;
use geozero::geo_types::GeoWriter;
use geozero::{GeozeroGeometry, wkb};
use std::ops::BitAnd;
use vortex_buffer::BitBuffer;
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_ensure, vortex_err};
use vortex_vector::bool::{BoolScalar, BoolVector};
use vortex_vector::{Datum, VectorOps};
use vortex_vector::{Scalar, Vector};

pub struct STContains;

impl VTable for STContains {
    type Options = EmptyOptions;

    fn id(&self) -> FunctionId {
        FunctionId::from("vortex.geo.contains")
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(2)
    }

    fn arg_name(&self, _options: &Self::Options, arg_idx: usize) -> ArgName {
        match arg_idx {
            0 => ArgName::new_ref("GeomA"),
            1 => ArgName::new_ref("GeomB"),
            _ => unreachable!("ST_Contains must be called with exactly 2 arguments"),
        }
    }

    fn return_dtype(&self, options: &Self::Options, arg_types: &[DType]) -> VortexResult<DType> {
        // The result is a Bool column with the nullability of its arguments
        let result_nullability = arg_types[0].nullability() | arg_types[1].nullability();
        Ok(DType::Bool(result_nullability))
    }

    fn execute(&self, _options: &Self::Options, args: &ExecutionArgs) -> VortexResult<Datum> {
        // Force each element to perform a datum operation here.
        // The inner option must be a Literal
        let geoma = args.input_datums(0);
        let geomb = args.input_datums(1);

        vortex_ensure!(
            args.input_type(0).is_binary() && args.input_type(1).is_binary(),
            "Arguments to ST_Contains must be binary"
        );

        // If we have two values, compare them.
        match (geoma, geomb) {
            (Datum::Scalar(geoma_scalar), Datum::Scalar(geomb_scalar)) => {
                let geoma = geoma_scalar
                    .as_binary()
                    .value()
                    .ok_or_else(|| vortex_err!("literal argument to ST_Contains cannot be NULL"))?;

                let geomb = geoma_scalar
                    .as_binary()
                    .value()
                    .ok_or_else(|| vortex_err!("literal argument to ST_Contains cannot be NULL"))?;

                let contain = parse_wkb(&geoma).contains(&parse_wkb(&geomb));

                Ok(Datum::Scalar(Scalar::Bool(BoolScalar::new(Some(contain)))))
            }
            (Datum::Scalar(geoma_scalar), Datum::Vector(geomb_vector)) => {
                let geoma = geoma_scalar
                    .as_binary()
                    .value()
                    .ok_or_else(|| vortex_err!("literal argument to ST_Contains cannot be NULL"))?;

                let geoma = parse_wkb(&geoma);

                let geomb_bin = geomb_vector.as_binary();
                let matches = BitBuffer::collect_bool(geomb_bin.len(), |index| {
                    match geomb_bin.get_ref(index) {
                        None => false,
                        Some(geomb_buf) => {
                            let geomb = parse_wkb(geomb_buf);
                            geoma.contains(&geomb)
                        }
                    }
                });
                Ok(Datum::Vector(Vector::Bool(BoolVector::new(
                    matches,
                    geomb_vector.validity().clone(),
                ))))
            }
            (Datum::Vector(geoma_vector), Datum::Scalar(geomb_scalar)) => {
                let geomb = geomb_scalar
                    .as_binary()
                    .value()
                    .ok_or_else(|| vortex_err!("literal argument to ST_Contains cannot be NULL"))?;

                let geomb = parse_wkb(&geomb);

                let geoma_bin = geoma_vector.as_binary();
                let matches = BitBuffer::collect_bool(geoma_bin.len(), |index| {
                    match geoma_bin.get_ref(index) {
                        None => false,
                        Some(geoma_buf) => {
                            let geoma = parse_wkb(geoma_buf);
                            geoma.contains(&geomb)
                        }
                    }
                });
                Ok(Datum::Vector(Vector::Bool(BoolVector::new(
                    matches,
                    geoma_vector.validity().clone(),
                ))))
            }
            (Datum::Vector(geoma_vector), Datum::Vector(geomb_vector)) => {
                vortex_ensure!(
                    geoma_vector.len() == geomb_vector.len(),
                    "ST_Contains input vectors must have same length"
                );

                let geoma_bin = geoma_vector.as_binary();
                let geomb_bin = geomb_vector.as_binary();

                let matches = BitBuffer::collect_bool(geoma_bin.len(), |index| {
                    let wkb_a = geoma_bin.get_ref(index);
                    let wkb_b = geomb_bin.get_ref(index);

                    match (wkb_a, wkb_b) {
                        (Some(a), Some(b)) => contains_wkb_slow(a, b),
                        _ => false,
                    }
                });

                let validity = geoma_bin.validity().bitand(geomb_bin.validity());

                Ok(Datum::Vector(Vector::Bool(BoolVector::new(
                    matches, validity,
                ))))
            }
        }
    }
}

fn contains_wkb_slow(left: &[u8], right: &[u8]) -> bool {
    let left_geom = parse_wkb(left);
    let right_geom = parse_wkb(right);

    left_geom.contains(&right_geom)
}

fn parse_wkb(wkb: &[u8]) -> Geometry {
    let mut writer = GeoWriter::new();
    wkb::Wkb(wkb)
        .process_geom(&mut writer)
        .expect("wkb parsing left");
    writer.take_geometry().expect("wkb should yield geometry")
}
