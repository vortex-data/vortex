// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Bind abstract stat expressions to concrete pruning stat sources.

use std::fmt;
use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::registry::CachedId;

use crate::aggregate_fn::AggregateFnRef;
use crate::dtype::DType;
use crate::dtype::Field;
use crate::dtype::FieldPath;
use crate::expr::BoundExpr;
use crate::expr::lit;
use crate::expr::placeholder::Placeholder;
use crate::expr::placeholder::PlaceholderId;
use crate::expr::placeholder::PlaceholderRef;
use crate::expr::stats::Stat;
use crate::expr::traversal::NodeExt;
use crate::expr::traversal::Transformed;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::get_item::GetItem;
use crate::scalar_fn::fns::stat::StatFn;

/// A reference to a statistic of a stored column supplied by the evaluation context.
#[derive(Clone, Debug)]
pub struct StatRef {
    payload: (FieldPath, Stat),
    dtype: DType,
    display_name: Arc<str>,
}

impl StatRef {
    /// Creates a stat reference placeholder.
    pub fn new(path: FieldPath, stat: Stat, dtype: DType) -> Self {
        let display_name = stat_ref_display_name(&path, stat);
        Self {
            payload: (path, stat),
            dtype,
            display_name,
        }
    }

    /// Returns the stored column path this reference addresses.
    pub fn path(&self) -> &FieldPath {
        &self.payload.0
    }

    /// Returns the statistic this reference addresses.
    pub fn stat(&self) -> Stat {
        self.payload.1
    }
}

impl Placeholder for StatRef {
    type Payload = (FieldPath, Stat);

    fn id(&self) -> PlaceholderId {
        static ID: CachedId = CachedId::new("vortex.stat_ref");
        *ID
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn display_name(&self) -> &str {
        &self.display_name
    }

    fn payload(&self) -> &Self::Payload {
        &self.payload
    }
}

/// Creates an expression placeholder for a stored column statistic.
pub fn stat_ref(path: FieldPath, stat: Stat, dtype: DType) -> BoundExpr {
    BoundExpr::Placeholder(PlaceholderRef::new(StatRef::new(path, stat, dtype)))
}

/// A target that can bind abstract `vortex.stat` calls to a concrete stats source.
pub trait StatBinder {
    /// Bind a legacy stat slot for a pure provenance chain.
    ///
    /// `Ok(None)` means the stat is unavailable for this binder.
    fn bind_stat(
        &mut self,
        path: &FieldPath,
        stat: Stat,
        stat_dtype: &DType,
    ) -> VortexResult<Option<BoundExpr>>;

    /// Bind an aggregate-backed stat call for a pure provenance chain.
    ///
    /// The default maps aggregates with legacy [`Stat`] slots to [`Self::bind_stat`].
    /// Aggregates such as all-null and all-NaN have no legacy slot and must be handled by
    /// concrete binders when supported.
    fn bind_aggregate(
        &mut self,
        path: &FieldPath,
        aggregate_fn: &AggregateFnRef,
        stat_dtype: &DType,
    ) -> VortexResult<Option<BoundExpr>> {
        let Some(stat) = Stat::from_aggregate_fn(aggregate_fn) else {
            return Ok(None);
        };
        self.bind_stat(path, stat, stat_dtype)
    }

    /// Replacement for an unavailable stat.
    ///
    /// The nullable typed-null literal preserves pruning's three-valued semantics: unknown stats
    /// are inconclusive rather than a proof.
    fn missing_stat(&mut self, dtype: DType) -> VortexResult<Option<BoundExpr>> {
        Ok(Some(lit(Scalar::null(dtype.as_nullable()))))
    }
}

/// Bind all abstract `vortex.stat` calls in the boolean predicate `predicate`.
///
/// Only pure provenance chains, `Root` or nested `GetItem` over `Root`, are eligible for binding.
/// Computed inputs are replaced with [`StatBinder::missing_stat`].
///
/// Returns `Ok(None)` when the bound predicate constant-folds to anything other than the literal
/// `true` — the predicate can never prune, so callers should skip evaluation entirely (it does
/// NOT mean "prune"). A surviving expression — including the literal `true`, which proves every
/// scope prunable — is returned as `Ok(Some(_))` for evaluation.
pub fn bind_stats(
    predicate: BoundExpr,
    binder: &mut impl StatBinder,
) -> VortexResult<Option<BoundExpr>> {
    let lowered = predicate
        .transform_down(|expr| bind_stat_expr(expr, binder))
        .map(Transformed::into_inner)?
        .optimize_recursive()?;

    if contains_root(&lowered) {
        return Err(vortex_err!(
            "Stats binding leaked a Root expression into the bound pruning predicate"
        ));
    }

    if let Some(scalar) = lowered.as_literal() {
        let value = scalar
            .as_bool_opt()
            .ok_or_else(|| {
                vortex_err!(
                    "bind_stats requires a boolean predicate, but it folded to a {} literal",
                    scalar.dtype()
                )
            })?
            .value();
        return Ok((value == Some(true)).then_some(lowered));
    }

    Ok(Some(lowered))
}

fn bind_stat_expr(
    expr: BoundExpr,
    binder: &mut impl StatBinder,
) -> VortexResult<Transformed<BoundExpr>> {
    let Some(options) = expr.as_opt::<StatFn>() else {
        return Ok(Transformed::no(expr));
    };
    let options = options.clone();
    let input = expr.child(0);
    let stat_dtype = expr.dtype().clone();

    let Some(path) = provenance_path(input) else {
        let replacement = binder
            .missing_stat(stat_dtype.clone())?
            .unwrap_or_else(|| null_stat(&stat_dtype));
        return Ok(Transformed::yes(replacement));
    };

    let replacement = match binder.bind_aggregate(&path, options.aggregate_fn(), &stat_dtype)? {
        Some(expr) => Some(expr),
        None => binder.missing_stat(stat_dtype.clone())?,
    };

    Ok(Transformed::yes(
        replacement.unwrap_or_else(|| null_stat(&stat_dtype)),
    ))
}

fn provenance_path(expr: &BoundExpr) -> Option<FieldPath> {
    match expr {
        BoundExpr::Root(_) => Some(FieldPath::root()),
        BoundExpr::Call(call) if call.function().is::<GetItem>() => {
            let field = call.function().as_::<GetItem>().clone();
            Some(provenance_path(call.child(0))?.push(field))
        }
        BoundExpr::Literal(_) | BoundExpr::Placeholder(_) | BoundExpr::Call(_) => None,
    }
}

fn null_stat(dtype: &DType) -> BoundExpr {
    lit(Scalar::null(dtype.as_nullable()))
}

fn contains_root(expr: &BoundExpr) -> bool {
    expr.is_root() || expr.children().iter().any(contains_root)
}

fn stat_ref_display_name(path: &FieldPath, stat: Stat) -> Arc<str> {
    let mut out = String::from("stat_ref(");
    if path.is_root() {
        out.push('$');
    } else {
        for (idx, field) in path.parts().iter().enumerate() {
            if idx > 0 {
                out.push('.');
            }
            match field {
                Field::Name(name) => out.push_str(name.as_ref()),
                Field::ElementType => out.push_str("[]"),
            }
        }
    }
    out.push('.');
    out.push_str(stat.name());
    out.push(')');
    out.into()
}

impl fmt::Display for StatRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.display_name())
    }
}

#[cfg(test)]
mod tests {
    use std::hash::BuildHasher;
    use std::hash::RandomState;

    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;

    use super::StatBinder;
    use super::bind_stats;
    use super::stat_ref;
    use crate::dtype::DType;
    use crate::dtype::Field;
    use crate::dtype::FieldPath;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::StructFields;
    use crate::expr::BoundExpr;
    use crate::expr::checked_add;
    use crate::expr::get_item;
    use crate::expr::gt;
    use crate::expr::lit;
    use crate::expr::root;
    use crate::expr::stats::Stat;
    use crate::stats::stat;

    /// Records every (path, stat) request; binds them all as StatRef placeholders.
    #[derive(Default)]
    struct RecordingBinder {
        requests: Vec<(FieldPath, Stat)>,
    }

    impl StatBinder for RecordingBinder {
        fn bind_stat(
            &mut self,
            path: &FieldPath,
            stat: Stat,
            stat_dtype: &DType,
        ) -> VortexResult<Option<BoundExpr>> {
            self.requests.push((path.clone(), stat));
            Ok(Some(stat_ref(path.clone(), stat, stat_dtype.clone())))
        }
    }

    fn max_stat(input: BoundExpr) -> BoundExpr {
        stat(
            input,
            Stat::Max
                .aggregate_fn()
                .vortex_expect("max has an aggregate fn"),
        )
    }

    /// Nested GetItem chains must produce the provenance path in root-to-leaf order.
    #[test]
    fn nested_provenance_path_ordering() -> VortexResult<()> {
        let scope = DType::Struct(
            StructFields::from_iter([(
                "a",
                DType::Struct(
                    StructFields::from_iter([(
                        "b",
                        DType::Primitive(PType::I32, Nullability::NonNullable),
                    )]),
                    Nullability::NonNullable,
                ),
            )]),
            Nullability::NonNullable,
        );
        let input = get_item("b", get_item("a", root(scope)));
        let predicate = gt(max_stat(input), lit(0i32));

        let mut binder = RecordingBinder::default();
        bind_stats(predicate, &mut binder)?;

        assert_eq!(
            binder.requests,
            vec![(
                FieldPath::from_iter([Field::Name("a".into()), Field::Name("b".into())]),
                Stat::Max
            )]
        );
        Ok(())
    }

    /// Computed stat inputs must never reach the binder (SOUNDNESS RULE).
    #[test]
    fn computed_input_never_binds() -> VortexResult<()> {
        let scope = DType::Struct(
            StructFields::from_iter([(
                "a",
                DType::Primitive(PType::I32, Nullability::NonNullable),
            )]),
            Nullability::NonNullable,
        );
        let computed = checked_add(get_item("a", root(scope)), lit(1i32));
        let predicate = gt(max_stat(computed), lit(10i32));

        let mut binder = RecordingBinder::default();
        bind_stats(predicate, &mut binder)?;

        assert!(binder.requests.is_empty());
        Ok(())
    }

    /// A binder that leaks a Root into its bound expression must produce a clear error.
    #[test]
    fn leaked_root_errors() -> VortexResult<()> {
        struct RootLeakingBinder;
        impl StatBinder for RootLeakingBinder {
            fn bind_stat(
                &mut self,
                _path: &FieldPath,
                _stat: Stat,
                stat_dtype: &DType,
            ) -> VortexResult<Option<BoundExpr>> {
                Ok(Some(root(stat_dtype.clone())))
            }
        }

        let scope = DType::Primitive(PType::I32, Nullability::NonNullable);
        let predicate = gt(max_stat(root(scope)), lit(0i32));
        let result = bind_stats(predicate, &mut RootLeakingBinder);
        let err = result.err().ok_or_else(|| vortex_err!("expected error"))?;
        assert!(err.to_string().contains("leaked a Root"), "{err}");
        Ok(())
    }

    /// Non-boolean predicates that fold to a literal must error, not panic.
    #[test]
    fn non_boolean_literal_errors() -> VortexResult<()> {
        let result = bind_stats(lit(5i32), &mut RecordingBinder::default());
        let err = result.err().ok_or_else(|| vortex_err!("expected error"))?;
        assert!(err.to_string().contains("boolean predicate"), "{err}");
        Ok(())
    }

    /// Stat references with equal dtypes but different (path, stat) payloads must differ in
    /// both equality and hash — placeholder identity is payload-sensitive.
    #[test]
    fn stat_ref_payload_identity() {
        let dtype = DType::Primitive(PType::I32, Nullability::Nullable);
        let a_min = stat_ref(FieldPath::from_name("a"), Stat::Min, dtype.clone());
        let a_max = stat_ref(FieldPath::from_name("a"), Stat::Max, dtype.clone());
        let b_max = stat_ref(FieldPath::from_name("b"), Stat::Max, dtype.clone());
        let a_max2 = stat_ref(FieldPath::from_name("a"), Stat::Max, dtype);

        assert_ne!(a_min, a_max);
        assert_ne!(a_max, b_max);
        assert_eq!(a_max, a_max2);

        let hasher = RandomState::new();
        assert_eq!(hasher.hash_one(&a_max), hasher.hash_one(&a_max2));
        assert_ne!(hasher.hash_one(&a_min), hasher.hash_one(&a_max));
    }
}
