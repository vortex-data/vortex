// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod kernel;

use std::fmt::Display;
use std::fmt::Formatter;

pub use kernel::*;
use prost::Message;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_proto::expr as pb;
use vortex_session::VortexSession;

use crate::AnyColumnar;
use crate::ArrayRef;
use crate::ArrayView;
use crate::CanonicalView;
use crate::ColumnarView;
use crate::ExecutionCtx;
use crate::arrays::Bool;
use crate::arrays::Constant;
use crate::arrays::Decimal;
use crate::arrays::Extension;
use crate::arrays::FixedSizeList;
use crate::arrays::ListView;
use crate::arrays::Null;
use crate::arrays::Primitive;
use crate::arrays::Struct;
use crate::arrays::VarBinView;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::expr::StatsCatalog;
use crate::expr::cast;
use crate::expr::expression::Expression;
use crate::expr::lit;
use crate::expr::stats::Stat;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ReduceCtx;
use crate::scalar_fn::ReduceNode;
use crate::scalar_fn::ReduceNodeRef;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;

/// A cast expression that converts values to a target data type.
#[derive(Clone)]
pub struct Cast;

/// How a cast matches up values between the input and target types.
///
/// Most relevant when casting between struct types, where matching fields by name allows
/// reordering and schema evolution, while matching by position requires the source and target
/// struct to have the same field order.
///
/// By-name is the default cast behavior and does not need to be requested explicitly. By-position
/// must always be opted into via [`CastOptions::by_position`].
#[derive(Copy, Clone, Hash, Eq, PartialEq, Debug)]
pub enum CastMode {
    /// Match fields by name. This is the implicit default of [`cast`](crate::expr::cast).
    ByName,
    /// Match fields by position. Requires explicit [`CastOptions::by_position`].
    ByPosition,
}

impl CastMode {
    /// A short string identifier for this mode, used in displays.
    pub fn name(&self) -> &'static str {
        match self {
            CastMode::ByPosition => "by_position",
            CastMode::ByName => "by_name",
        }
    }
}

impl From<CastMode> for pb::CastMode {
    fn from(mode: CastMode) -> Self {
        match mode {
            CastMode::ByPosition => pb::CastMode::ByPosition,
            CastMode::ByName => pb::CastMode::ByName,
        }
    }
}

impl From<pb::CastMode> for CastMode {
    fn from(mode: pb::CastMode) -> Self {
        match mode {
            pb::CastMode::ByPosition => CastMode::ByPosition,
            pb::CastMode::ByName => CastMode::ByName,
        }
    }
}

/// Options controlling the semantics of a cast operation.
///
/// The target data type is passed separately alongside these options; `CastOptions` captures
/// only knobs that tweak *how* the cast is performed.
///
/// `CastOptions` intentionally has no `Default`: by-name is already the default behavior of the
/// plain [`cast`](crate::expr::cast) helpers, so callers that pass options must be explicit about
/// which mode they want — typically [`CastOptions::by_position`] when they need positional
/// semantics.
#[derive(Copy, Clone, Hash, Eq, PartialEq, Debug)]
pub struct CastOptions {
    mode: CastMode,
}

impl CastOptions {
    /// Create a new [`CastOptions`] with the given matching mode.
    pub fn new(mode: CastMode) -> Self {
        Self { mode }
    }

    /// Options that match struct fields by name. This is the same behavior as
    /// [`cast`](crate::expr::cast), and is exposed so callers can be explicit about the cast
    /// mode (e.g. in tests) or pass [`CastOptions`] generically.
    pub fn by_name() -> Self {
        Self::new(CastMode::ByName)
    }

    /// Options that match struct fields by position. Positional cast must always be requested
    /// explicitly; there is no default-options path that selects this mode.
    pub fn by_position() -> Self {
        Self::new(CastMode::ByPosition)
    }

    /// The field-matching mode of this cast.
    pub fn mode(&self) -> CastMode {
        self.mode
    }
}

impl Display for CastOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.mode.name())
    }
}

/// Combined options stored on a `vortex.cast` [`ScalarFnVTable`] instance: the target
/// data type plus the [`CastOptions`] controlling how the cast is performed.
#[derive(Clone, Hash, Eq, PartialEq, Debug)]
pub struct CastFnOptions {
    target: DType,
    options: CastOptions,
}

impl CastFnOptions {
    /// Create a new [`CastFnOptions`] from a target type and cast options.
    pub fn new(target: DType, options: CastOptions) -> Self {
        Self { target, options }
    }

    /// The target data type of this cast.
    pub fn target(&self) -> &DType {
        &self.target
    }

    /// The [`CastOptions`] that govern how the cast is performed.
    pub fn options(&self) -> &CastOptions {
        &self.options
    }
}

impl Display for CastFnOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        // Omit the mode for the implicit default (by-name) to keep the common case terse.
        match self.options.mode {
            CastMode::ByName => write!(f, "{}", self.target),
            CastMode::ByPosition => write!(f, "{}: {}", self.options, self.target),
        }
    }
}

impl ScalarFnVTable for Cast {
    type Options = CastFnOptions;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::new("vortex.cast")
    }

    fn serialize(&self, instance: &CastFnOptions) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            pb::CastOpts {
                target: Some((&instance.target).try_into()?),
                mode: pb::CastMode::from(instance.options.mode()) as i32,
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(
        &self,
        metadata: &[u8],
        session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        let proto = pb::CastOpts::decode(metadata)?;
        let mode = CastMode::from(pb::CastMode::try_from(proto.mode).map_err(|_| {
            vortex_err!("Unknown cast mode value {} in Cast expression", proto.mode)
        })?);
        let target = DType::from_proto(
            proto
                .target
                .as_ref()
                .ok_or_else(|| vortex_err!("Missing target dtype in Cast expression"))?,
            session,
        )?;
        Ok(CastFnOptions::new(target, CastOptions::new(mode)))
    }

    fn arity(&self, _options: &CastFnOptions) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _instance: &CastFnOptions, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("input"),
            _ => unreachable!("Invalid child index {} for Cast expression", child_idx),
        }
    }

    fn fmt_sql(
        &self,
        instance: &CastFnOptions,
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "cast(")?;
        expr.children()[0].fmt_sql(f)?;
        write!(f, " as {}", instance)?;
        write!(f, ")")
    }

    fn return_dtype(&self, instance: &CastFnOptions, _arg_dtypes: &[DType]) -> VortexResult<DType> {
        Ok(instance.target.clone())
    }

    fn execute(
        &self,
        instance: &CastFnOptions,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let input = args.get(0)?;

        let Some(columnar) = input.as_opt::<AnyColumnar>() else {
            return input
                .execute::<ArrayRef>(ctx)?
                .cast_opts(instance.target.clone(), instance.options);
        };

        match columnar {
            ColumnarView::Canonical(canonical) => {
                match cast_canonical(canonical, &instance.target, &instance.options, ctx)? {
                    Some(result) => Ok(result),
                    None => vortex_bail!(
                        "No CastKernel to cast canonical array {} from {} to {}",
                        canonical.to_array_ref().encoding_id(),
                        canonical.to_array_ref().dtype(),
                        instance.target,
                    ),
                }
            }
            ColumnarView::Constant(constant) => {
                match cast_constant(constant, &instance.target, &instance.options)? {
                    Some(result) => Ok(result),
                    None => vortex_bail!(
                        "No CastReduce to cast constant array from {} to {}",
                        constant.dtype(),
                        instance.target,
                    ),
                }
            }
        }
    }

    fn reduce(
        &self,
        instance: &CastFnOptions,
        node: &dyn ReduceNode,
        _ctx: &dyn ReduceCtx,
    ) -> VortexResult<Option<ReduceNodeRef>> {
        // Collapse node if child is already the target type
        let child = node.child(0);
        if child.node_dtype()? == instance.target {
            return Ok(Some(child));
        }
        Ok(None)
    }

    fn stat_expression(
        &self,
        instance: &CastFnOptions,
        expr: &Expression,
        stat: Stat,
        catalog: &dyn StatsCatalog,
    ) -> Option<Expression> {
        match stat {
            Stat::IsConstant
            | Stat::IsSorted
            | Stat::IsStrictSorted
            | Stat::NaNCount
            | Stat::Sum
            | Stat::UncompressedSizeInBytes => expr.child(0).stat_expression(stat, catalog),
            Stat::Max | Stat::Min => {
                // We cast min/max to the new type
                expr.child(0)
                    .stat_expression(stat, catalog)
                    .map(|x| cast(x, instance.target.clone()))
            }
            Stat::NullCount => {
                // if !expr.data().is_nullable() {
                // NOTE(ngates): we should decide on the semantics here. In theory, the null
                //  count of something cast to non-nullable will be zero. But if we return
                //  that we know this to be zero, then a pruning predicate may eliminate data
                //  that would otherwise have caused the cast to error.
                // return Some(lit(0u64));
                // }
                None
            }
        }
    }

    fn validity(
        &self,
        instance: &CastFnOptions,
        expression: &Expression,
    ) -> VortexResult<Option<Expression>> {
        Ok(Some(if instance.target.is_nullable() {
            expression.child(0).validity()?
        } else {
            lit(true)
        }))
    }

    // This might apply a nullability
    fn is_null_sensitive(&self, _instance: &CastFnOptions) -> bool {
        true
    }
}

/// Cast a canonical array to the target dtype by dispatching to the appropriate
/// [`CastReduce`] or [`CastKernel`] for each canonical encoding.
fn cast_canonical(
    canonical: CanonicalView<'_>,
    dtype: &DType,
    options: &CastOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    match canonical {
        CanonicalView::Null(a) => <Null as CastReduce>::cast(a, dtype, options),
        CanonicalView::Bool(a) => <Bool as CastReduce>::cast(a, dtype, options),
        CanonicalView::Primitive(a) => <Primitive as CastKernel>::cast(a, dtype, options, ctx),
        CanonicalView::Decimal(a) => <Decimal as CastKernel>::cast(a, dtype, options, ctx),
        CanonicalView::VarBinView(a) => <VarBinView as CastReduce>::cast(a, dtype, options),
        CanonicalView::List(a) => <ListView as CastReduce>::cast(a, dtype, options),
        CanonicalView::FixedSizeList(a) => <FixedSizeList as CastReduce>::cast(a, dtype, options),
        CanonicalView::Struct(a) => <Struct as CastKernel>::cast(a, dtype, options, ctx),
        CanonicalView::Extension(a) => <Extension as CastReduce>::cast(a, dtype, options),
        CanonicalView::Variant(_) => {
            vortex_bail!("Variant arrays don't support casting")
        }
    }
}

/// Cast a constant array by dispatching to its [`CastReduce`] implementation.
fn cast_constant(
    array: ArrayView<Constant>,
    dtype: &DType,
    options: &CastOptions,
) -> VortexResult<Option<ArrayRef>> {
    <Constant as CastReduce>::cast(array, dtype, options)
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect as _;

    use crate::IntoArray;
    use crate::arrays::StructArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::Expression;
    use crate::expr::cast;
    use crate::expr::get_item;
    use crate::expr::root;
    use crate::expr::test_harness;

    #[test]
    fn dtype() {
        let dtype = test_harness::struct_dtype();
        assert_eq!(
            cast(root(), DType::Bool(Nullability::NonNullable))
                .return_dtype(&dtype)
                .unwrap(),
            DType::Bool(Nullability::NonNullable)
        );
    }

    #[test]
    fn replace_children() {
        let expr = cast(root(), DType::Bool(Nullability::Nullable));
        expr.with_children(vec![root()])
            .vortex_expect("operation should succeed in test");
    }

    #[test]
    fn evaluate() {
        let test_array = StructArray::from_fields(&[
            ("a", buffer![0i32, 1, 2].into_array()),
            ("b", buffer![4i64, 5, 6].into_array()),
        ])
        .unwrap()
        .into_array();

        let expr: Expression = cast(
            get_item("a", root()),
            DType::Primitive(PType::I64, Nullability::NonNullable),
        );
        let result = test_array.apply(&expr).unwrap();

        assert_eq!(
            result.dtype(),
            &DType::Primitive(PType::I64, Nullability::NonNullable)
        );
    }

    #[test]
    fn test_display() {
        let expr = cast(
            get_item("value", root()),
            DType::Primitive(PType::I64, Nullability::NonNullable),
        );
        assert_eq!(expr.to_string(), "cast($.value as i64)");

        let expr2 = cast(root(), DType::Bool(Nullability::Nullable));
        assert_eq!(expr2.to_string(), "cast($ as bool?)");
    }
}
