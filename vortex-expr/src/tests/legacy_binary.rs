// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Loading expressions written with the legacy `vortex.binary` function.

use std::sync::LazyLock;

use prost::Message;
use rstest::rstest;
use vortex_array::proto::expr as pb;
use vortex_array::proto::expr::binary_opts::BinaryOp;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::Expression;
use crate::proto::ExprSerializeProtoExt;
use crate::root;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

/// Each operator of the legacy `vortex.binary` function loads as the function that now owns
/// it, and serializes again with the id of that function.
#[rstest]
#[case(BinaryOp::Eq, "vortex.compare", "=")]
#[case(BinaryOp::NotEq, "vortex.compare", "!=")]
#[case(BinaryOp::Gt, "vortex.compare", ">")]
#[case(BinaryOp::Gte, "vortex.compare", ">=")]
#[case(BinaryOp::Lt, "vortex.compare", "<")]
#[case(BinaryOp::Lte, "vortex.compare", "<=")]
#[case(BinaryOp::And, "vortex.boolean", "and")]
#[case(BinaryOp::Or, "vortex.boolean", "or")]
#[case(BinaryOp::Add, "vortex.numeric", "+")]
#[case(BinaryOp::Sub, "vortex.numeric", "-")]
#[case(BinaryOp::Mul, "vortex.numeric", "*")]
#[case(BinaryOp::Div, "vortex.numeric", "/")]
fn legacy_binary_loads_as_the_new_function(
    #[case] op: BinaryOp,
    #[case] expected_id: &str,
    #[case] expected_operator: &str,
) -> VortexResult<()> {
    let operand = root().serialize_proto()?;
    let legacy = pb::Expr {
        id: "vortex.binary".to_string(),
        metadata: Some(pb::BinaryOpts { op: op.into() }.encode_to_vec()),
        children: vec![operand.clone(), operand],
    };

    let expr = Expression::from_proto(&legacy, &SESSION)?;
    let scalar_fn = expr.as_scalar().vortex_expect("a scalar function");
    assert_eq!(scalar_fn.id().as_ref(), expected_id);
    assert_eq!(scalar_fn.options().to_string(), expected_operator);

    let written = expr.serialize_proto()?;
    assert_eq!(written.id, expected_id);
    assert_eq!(Expression::from_proto(&written, &SESSION)?, expr);

    Ok(())
}
