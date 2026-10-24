// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::slice;

use prost::Message;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::ScalarFnArray;
use crate::arrays::StructArray;
use crate::arrays::struct_::StructArrayExt;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::FieldName;
use crate::dtype::Nullability;
use crate::proto::expr as pb;
use crate::scalar::Scalar;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::EmptyOptions;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ReduceNode;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::builtin::mask::Mask;
use crate::scalar_fn::builtin::pack::Pack;
use crate::scalar_fn::display::ExprDisplay;

#[derive(Clone)]
pub struct GetItem;

impl GetItem {
    /// Creates a lazy projection of `field_name` from `input`.
    ///
    /// # Errors
    ///
    /// Returns an error if `input` is not a struct or does not contain `field_name`.
    pub fn try_new(
        input: ArrayRef,
        field_name: impl Into<FieldName>,
    ) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(GetItem.bind(field_name.into()), [input])
    }

    /// Read `field_name` out of `struct_node`, without a `get_item` node where possible.
    ///
    /// Rules that read a field out of their own child use this instead of building the
    /// `get_item` node themselves. The optimizer visits children before their parent, so a
    /// `get_item` that a rule introduces is never simplified again. Removing it here keeps the
    /// output of the rule a fixed point.
    pub fn apply_now<T: ReduceNode>(struct_node: &T, field_name: &FieldName) -> VortexResult<T> {
        if let Some(field) = Self::maybe_apply_now(struct_node, field_name)? {
            return Ok(field);
        }

        struct_node.new_node(
            GetItem.bind(field_name.clone()),
            slice::from_ref(struct_node),
        )
    }

    /// Read `field_name` straight out of a `pack` node, which holds each field as a child.
    ///
    /// Returns `None` if `struct_node` is not a `pack`, or does not hold `field_name`.
    fn maybe_apply_now<T: ReduceNode>(
        struct_node: &T,
        field_name: &FieldName,
    ) -> VortexResult<Option<T>> {
        let Some(pack) = struct_node
            .scalar_fn()
            .and_then(|scalar_fn| scalar_fn.as_opt::<Pack>())
        else {
            return Ok(None);
        };
        let Some(idx) = pack.names.find(field_name) else {
            return Ok(None);
        };

        let field = struct_node.child(idx);
        if !pack.nullability.is_nullable() {
            return Ok(Some(field));
        }

        // A nullable pack intersects its validity into the field, so mask the field with an
        // all-true constant to make its dtype nullable.
        Ok(Some(struct_node.new_node(
            Mask.bind(EmptyOptions),
            &[field, struct_node.new_constant(true.into())],
        )?))
    }
}

impl ScalarFnVTable for GetItem {
    type Options = FieldName;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.get_item");
        *ID
    }

    fn serialize(&self, instance: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            pb::GetItemOpts {
                path: instance.to_string(),
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(
        &self,
        _metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        let opts = pb::GetItemOpts::decode(_metadata)?;
        Ok(FieldName::from(opts.path))
    }

    fn arity(&self, _field_name: &FieldName) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _instance: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("input"),
            _ => unreachable!("Invalid child index {} for GetItem expression", child_idx),
        }
    }

    fn fmt_sql(
        &self,
        field_name: &FieldName,
        expr: &dyn ExprDisplay,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        Display::fmt(expr.display_child(0), f)?;
        write!(f, ".{}", field_name)
    }

    fn return_dtype(&self, field_name: &FieldName, arg_dtypes: &[DType]) -> VortexResult<DType> {
        let struct_dtype = &arg_dtypes[0];
        let field_dtype = struct_dtype
            .as_struct_fields_opt()
            .and_then(|st| st.field(field_name))
            .ok_or_else(|| {
                vortex_err!("Couldn't find the {} field in the input scope", field_name)
            })?;

        // Match here to avoid cloning the dtype if nullability doesn't need to change
        if matches!(
            (struct_dtype.nullability(), field_dtype.nullability()),
            (Nullability::Nullable, Nullability::NonNullable)
        ) {
            return Ok(field_dtype.with_nullability(Nullability::Nullable));
        }

        Ok(field_dtype)
    }

    fn execute(
        &self,
        field_name: &FieldName,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let input = args.get(0)?;
        if let Some(constant) = input.as_opt::<Constant>() {
            let dtype = self.return_dtype(field_name, &[input.dtype().clone()])?;
            let scalar = if constant.scalar().is_null() {
                Scalar::null(dtype)
            } else {
                constant
                    .scalar()
                    .as_struct()
                    .field(field_name)
                    .ok_or_else(|| {
                        vortex_err!(
                            "Field '{}' missing from constant struct array {}",
                            field_name,
                            input.dtype()
                        )
                    })?
                    .cast(&dtype)?
            };

            return Ok(ConstantArray::new(scalar, input.len()).into_array());
        }

        let input = input.execute::<StructArray>(ctx)?;
        let field = input.unmasked_field_by_name(field_name).cloned()?;

        match input.dtype().nullability() {
            Nullability::NonNullable => Ok(field),
            Nullability::Nullable => field.mask(input.validity()?.to_array(input.len())),
        }
    }

    fn reduce<T: ReduceNode>(&self, field_name: &FieldName, node: &T) -> VortexResult<Option<T>> {
        Self::maybe_apply_now(&node.child(0), field_name)
    }

    fn is_strict(&self, _field_name: &FieldName) -> bool {
        true
    }

    fn is_infallible(&self, _field_name: &FieldName) -> bool {
        // If this type-checks, it is infallible.
        true
    }
}
