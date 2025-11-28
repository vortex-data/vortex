// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::Ref;
use vortex_session::SessionExt;
use vortex_session::registry::Registry;

use crate::array::evaluate::ArrayEvaluator;
use crate::arrays::BoolMaskedValidityRule;
use crate::arrays::BoolVTable;
use crate::arrays::ChunkedVTable;
use crate::arrays::ConstantVTable;
use crate::arrays::DecimalMaskedValidityRule;
use crate::arrays::DecimalVTable;
use crate::arrays::ExprVTable;
use crate::arrays::ExtensionVTable;
use crate::arrays::FixedSizeListVTable;
use crate::arrays::ListVTable;
use crate::arrays::ListViewVTable;
use crate::arrays::MaskedVTable;
use crate::arrays::NullVTable;
use crate::arrays::PrimitiveMaskedValidityRule;
use crate::arrays::PrimitiveVTable;
use crate::arrays::StructExprPartitionRule;
use crate::arrays::StructVTable;
use crate::arrays::VarBinVTable;
use crate::arrays::VarBinViewVTable;
use crate::optimizer::ArrayOptimizer;
use crate::optimizer::rules::AnyArrayParent;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::optimizer::rules::ArrayReduceRule;
use crate::vtable::ArrayVTable;
use crate::vtable::ArrayVTableExt;
use crate::vtable::VTable;

pub type ArrayRegistry = Registry<ArrayVTable>;

#[derive(Debug)]
pub struct ArraySession {
    /// The set of registered array encodings.
    registry: ArrayRegistry,

    /// The array optimizer containing rules.
    optimizer: ArrayOptimizer,

    /// An evaluator for apply expressions to arrays.
    /// This temporary until expressions becomes a closed grammar with lazy evaluation.
    evaluator: ArrayEvaluator,
}

impl ArraySession {
    pub fn registry(&self) -> &ArrayRegistry {
        &self.registry
    }

    pub fn optimizer(&self) -> &ArrayOptimizer {
        &self.optimizer
    }

    pub fn optimizer_mut(&mut self) -> &mut ArrayOptimizer {
        &mut self.optimizer
    }

    pub fn evaluator(&self) -> &ArrayEvaluator {
        &self.evaluator
    }

    pub fn evaluator_mut(&mut self) -> &mut ArrayEvaluator {
        &mut self.evaluator
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
    pub fn register_reduce_rule<V, R>(&mut self, encoding: &V, rule: R)
    where
        V: VTable,
        R: 'static + ArrayReduceRule<V>,
    {
        self.optimizer.register_reduce_rule::<V, R>(encoding, rule);
    }

    /// Register a parent reduce rule for specific child and parent types
    pub fn register_parent_rule<Child, Parent, R>(
        &mut self,
        child_encoding: &Child,
        parent_encoding: &Parent,
        rule: R,
    ) where
        Child: VTable,
        Parent: VTable,
        R: 'static + ArrayParentReduceRule<Child, Parent>,
    {
        self.optimizer.register_parent_rule::<Child, Parent, R>(
            child_encoding,
            parent_encoding,
            rule,
        );
    }

    /// Register a parent reduce rule that matches any parent type
    pub fn register_any_parent_rule<Child, R>(&mut self, child_encoding: &Child, rule: R)
    where
        Child: VTable,
        R: 'static + ArrayParentReduceRule<Child, AnyArrayParent>,
    {
        self.optimizer
            .register_any_parent_rule::<Child, R>(child_encoding, rule);
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

        let mut session = Self {
            registry: encodings,
            optimizer: ArrayOptimizer::default(),
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
