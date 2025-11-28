// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub mod rewrite;

use crate::array::evaluate::ArrayEvaluator;
use crate::arrays::{
    BoolMaskedValidityRule, BoolVTable, ChunkedVTable, ConstantVTable, DecimalMaskedValidityRule,
    DecimalVTable, ExprOptimizationRule, ExprVTable, ExtensionVTable, FixedSizeListVTable,
    ListVTable, ListViewVTable, MaskedVTable, NullVTable, PrimitiveMaskedValidityRule,
    PrimitiveVTable, StructExprPartitionRule, StructVTable, VarBinVTable, VarBinViewVTable,
};
use crate::expr;
use crate::transform::{AnyArrayParent, ArrayOptimizer, ArrayParentReduceRule, ArrayReduceRule};
use crate::vtable::{ArrayVTable, ArrayVTableExt, VTable};
pub use rewrite::ArrayRewriteRuleRegistry;
use vortex_session::registry::Registry;
use vortex_session::{Ref, SessionExt};

pub type ArrayRegistry = Registry<ArrayVTable>;

#[derive(Debug)]
pub struct ArraySession {
    /// The set of registered array encodings.
    registry: ArrayRegistry,

    /// The set of registered rewrite rules.
    rewrite_rules: ArrayRewriteRuleRegistry,

    /// An evaluator for apply expressions to arrays.
    /// This temporary until expressions becomes a closed grammar with lazy evaluation.
    evaluator: ArrayEvaluator,
}

impl ArraySession {
    pub fn registry(&self) -> &ArrayRegistry {
        &self.registry
    }

    pub fn rewrite_rules(&self) -> &ArrayRewriteRuleRegistry {
        &self.rewrite_rules
    }

    /// Register a new array encoding, replacing any existing encoding with the same ID.
    pub fn register(&self, encoding: ArrayVTable) {
        self.registry.register(encoding)
    }

    /// Register many array encodings, replacing any existing encodings with the same ID.
    pub fn register_many(&self, encodings: impl IntoIterator<Item = ArrayVTable>) {
        self.registry.register_many(encodings);
    }

    /// Register a reduce rule for a specific array encoding
    pub fn register_reduce_rule<V, R>(&self, encoding: &V, rule: R)
    where
        V: VTable,
        R: 'static + ArrayReduceRule<V>,
    {
        self.rewrite_rules
            .register_reduce_rule::<V, R>(encoding, rule);
    }

    /// Register a parent reduce rule for specific child and parent types
    pub fn register_parent_rule<Child, Parent, R>(
        &self,
        child_encoding: &Child,
        parent_encoding: &Parent,
        rule: R,
    ) where
        Child: VTable,
        Parent: VTable,
        R: 'static + ArrayParentReduceRule<Child, Parent>,
    {
        self.rewrite_rules.register_parent_rule::<Child, Parent, R>(
            child_encoding,
            parent_encoding,
            rule,
        );
    }

    /// Register a parent reduce rule that matches any parent type
    pub fn register_any_parent_rule<Child, R>(&self, child_encoding: &Child, rule: R)
    where
        Child: VTable,
        R: 'static + ArrayParentReduceRule<Child, AnyArrayParent>,
    {
        self.rewrite_rules
            .register_any_parent_rule::<Child, R>(child_encoding, rule);
    }

    /// Create an ArrayOptimizer using this session's rules
    pub fn optimizer(&self, expr_optimizer: expr::transform::ExprOptimizer) -> ArrayOptimizer {
        ArrayOptimizer::new(self.rewrite_rules.clone(), expr_optimizer)
    }
}

impl Default for ArraySession {
    fn default() -> Self {
        let encodings = ArrayRegistry::default();

        // Register the canonical encodings.
        encodings.register_many([
            NullVTable.as_vtable(),
            BoolVTable.as_vtable(),
            PrimitiveVTable.as_vtable(),
            DecimalVTable.as_vtable(),
            VarBinViewVTable.as_vtable(),
            ListViewVTable.as_vtable(),
            FixedSizeListVTable.as_vtable(),
            StructVTable.as_vtable(),
            ExtensionVTable.as_vtable(),
        ]);

        // Register the utility encodings.
        encodings.register_many([
            ChunkedVTable.as_vtable(),
            ConstantVTable.as_vtable(),
            MaskedVTable.as_vtable(),
            ListVTable.as_vtable(),
            VarBinVTable.as_vtable(),
        ]);

        let session = Self {
            registry: encodings,
            rewrite_rules: ArrayRewriteRuleRegistry::default(),
            evaluator: ArrayEvaluator::default(),
        };

        session.register_parent_rule::<BoolVTable, MaskedVTable, BoolMaskedValidityRule>(
            &BoolVTable,
            &MaskedVTable,
            BoolMaskedValidityRule,
        );

        session.register_parent_rule::<PrimitiveVTable, MaskedVTable, PrimitiveMaskedValidityRule>(
            &PrimitiveVTable,
            &MaskedVTable,
            PrimitiveMaskedValidityRule,
        );

        session.register_parent_rule::<DecimalVTable, MaskedVTable, DecimalMaskedValidityRule>(
            &DecimalVTable,
            &MaskedVTable,
            DecimalMaskedValidityRule,
        );

        session.register_parent_rule::<StructVTable, ExprVTable, StructExprPartitionRule>(
            &StructVTable,
            &ExprVTable,
            StructExprPartitionRule,
        );

        session.register_reduce_rule::<ExprVTable, ExprOptimizationRule>(
            &ExprVTable,
            ExprOptimizationRule,
        );

        session
    }
}

/// Session data for Vortex arrays.
pub trait ArraySessionExt: SessionExt {
    /// Returns the array encoding registry.
    fn arrays(&self) -> Ref<'_, ArraySession> {
        self.get::<ArraySession>()
    }
}

impl<S: SessionExt> ArraySessionExt for S {}
