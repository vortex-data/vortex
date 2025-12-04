// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An implementation of an ST_Contains expression type.

// use crate::expr::functions::{ArgName, Arity, EmptyOptions, ExecutionArgs, FunctionId, VTable};
use crate::accessor::ArrayAccessor;
use crate::arrays::{BoolArray, ConstantArray};
use crate::expr::traversal::Node;
use crate::expr::{ChildName, ExprId, ExpressionView, Literal, VTable};
use crate::vtable::ValidityHelper;
use crate::{Array, ArrayRef, IntoArray, ToCanonical};
use geo::{Centroid, Contains};
use geo_types::Geometry;
use geozero::geo_types::GeoWriter;
use geozero::{GeozeroGeometry, wkb};
use std::fmt::Formatter;
use vortex_buffer::BitBuffer;
use vortex_dtype::{DType, Nullability};
use vortex_error::{VortexResult, vortex_ensure};
use vortex_scalar::Scalar;

pub struct STContains;

impl VTable for STContains {
    type Instance = ();

    fn id(&self) -> ExprId {
        ExprId::from("vortex.geo.contains")
    }

    fn validate(&self, expr: &ExpressionView<Self>) -> VortexResult<()> {
        vortex_ensure!(
            expr.children_count() == 2,
            "ST_Contains expression must have exactly 2 children"
        );

        let _lhs = &expr.children()[0];
        let _rhs = &expr.children()[1];

        // TODO(aduffy): do other checks on the lhs/rhs

        Ok(())
    }

    fn child_name(&self, _instance: &Self::Instance, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::new_ref("geom_a"),
            1 => ChildName::new_ref("geom_b"),
            _ => unreachable!("child_name called with invalid child_idx"),
        }
    }

    fn fmt_sql(&self, expr: &ExpressionView<Self>, f: &mut Formatter<'_>) -> std::fmt::Result {
        let lhs = &expr.children()[0];
        let rhs = &expr.children()[1];

        write!(f, "ST_CONTAINS(")?;
        lhs.fmt_sql(f)?;
        write!(f, ", ")?;
        rhs.fmt_sql(f)?;
        write!(f, ")")
    }

    fn return_dtype(&self, expr: &ExpressionView<Self>, scope: &DType) -> VortexResult<DType> {
        let lhs = &expr.children()[0];
        let rhs = &expr.children()[1];
        let nullability =
            lhs.return_dtype(scope)?.nullability() | rhs.return_dtype(scope)?.nullability();
        Ok(DType::Bool(nullability))
    }

    fn evaluate(&self, expr: &ExpressionView<Self>, scope: &ArrayRef) -> VortexResult<ArrayRef> {
        let lhs = &expr.children()[0];
        let rhs = &expr.children()[1];

        match (lhs.as_opt::<Literal>(), rhs.as_opt::<Literal>()) {
            (Some(l), Some(r)) => {
                // Both are literals
                let len = scope.len();

                let l_v = l.data().as_binary().value();
                let r_v = r.data().as_binary().value();
                let constant = match (l_v, r_v) {
                    (Some(wkb_l), Some(wkb_r)) => {
                        let geom_l = parse_wkb(&wkb_l);
                        let geom_r = parse_wkb(&wkb_r);
                        Scalar::bool(geom_l.contains(&geom_r), Nullability::NonNullable)
                    }
                    _ => Scalar::null(DType::Bool(Nullability::Nullable)),
                };

                Ok(ConstantArray::new(constant, len).into_array())
            }
            (Some(l), None) => {
                // lhs is literal, rhs is an array that we need to iterate over.
                let rhs = rhs.evaluate(scope)?;
                let len = rhs.len();

                let Some(wkb_l) = l.data().as_binary().value() else {
                    return Ok(ConstantArray::new(
                        Scalar::null(DType::Bool(Nullability::Nullable)),
                        len,
                    )
                    .into_array());
                };

                let geom_l = parse_wkb(&wkb_l);

                let rhs = rhs.to_varbinview();
                let validity = rhs.validity().clone();

                rhs.with_iterator(|iter| {
                    let matches = iter
                        .map(|rhs_value| match rhs_value {
                            None => false,
                            Some(wkb_r) => {
                                let geom_r = parse_wkb(wkb_r);
                                // Get centroid of the geometry
                                let _centroid =  geom_r.centroid();
                                geom_l.contains(&geom_r)
                            }
                        })
                        .collect::<BitBuffer>();

                    Ok(BoolArray::from_bit_buffer(matches, validity).into_array())
                })
            }
            (None, Some(r)) => {
                // rhs is literal, lhs is an array that we need to iterate over
                let lhs = lhs.evaluate(scope)?;
                let len = lhs.len();

                let Some(wkb_r) = r.data().as_binary().value() else {
                    return Ok(ConstantArray::new(
                        Scalar::null(DType::Bool(Nullability::Nullable)),
                        len,
                    )
                    .into_array());
                };

                let geom_r = parse_wkb(&wkb_r);

                let lhs = lhs.to_varbinview();
                let validity = lhs.validity().clone();
                lhs.with_iterator(|iter| {
                    let matches = iter
                        .map(|v| match v {
                            None => false,
                            Some(wkb_l) => {
                                let geom_l = parse_wkb(wkb_l);
                                geom_l.contains(&geom_r)
                            }
                        })
                        .collect::<BitBuffer>();

                    Ok(BoolArray::from_bit_buffer(matches, validity).into_array())
                })
            }
            (None, None) => {
                // lhs and rhs are both arrays, we need to zip/iterate them both.
                let lhs = lhs.evaluate(scope)?.to_varbinview();
                let rhs = rhs.evaluate(scope)?.to_varbinview();

                // And the validities together.
                let validity = lhs.validity().clone().and(rhs.validity().clone());

                let len = rhs.len();

                // TODO(aduffy): hoist validity checking
                let matches = BitBuffer::collect_bool(len, |index| {
                    if lhs.is_invalid(index) || rhs.is_invalid(index) {
                        return false;
                    }

                    let l_v = lhs.bytes_at(index);
                    let r_v = rhs.bytes_at(index);

                    let geom_l = parse_wkb(&l_v);
                    let geom_r = parse_wkb(&r_v);

                    geom_l.contains(&geom_r)
                });

                Ok(BoolArray::from_bit_buffer(matches, validity).into_array())
            }
        }
    }
}

fn parse_wkb(wkb: &[u8]) -> Geometry {
    let mut writer = GeoWriter::new();
    wkb::Wkb(wkb)
        .process_geom(&mut writer)
        .expect("wkb parsing left");
    writer.take_geometry().expect("wkb should yield geometry")
}
