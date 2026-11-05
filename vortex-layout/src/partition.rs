// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! We have Plans which tell us how data is stored and how to read it. An
//! Expression is a subset of a Plan which tells us how to read the data but
//! doesn't know how the data is stored.
//!
//! Given a plan graph and an expression graph, we want to know which
//! expressions can be pushed down to plan's children. For this we need to know
//! which expressions operate on plan's disjoint slots (children).
//!
//! So, for an expression the rule (property) is "Given a plan and an
//! expression graph, if you see this expression, it needs the following slots
//! from a plan". Let's look at an example.
//!
//! If we have a non-nullable PackPlan(a: .., b: .., c: ..) and an Expression
//! $.a > 5 && $.b < 1, we know that $.a > 5 and $.b < 1 are disjoint over the
//! plan, and we never read "c". We know this because $.a and $.b (get_item)
//! have a rule for a PackPlan which returns us one field.
//!
//! Thus we partition the expression into two parts: $.a > 5 references child
//! plan for "a" of PackPlan, and we rewrite the expression over this plan to $
//! > 5, same for $.b < 1.
//!
//! $.a > 5 over pack(a: child_plan) => $ > 5 over child_plan
//!
//! We also need a way to combine these parts back into an Expression. If we
//! call these partitions 0 and 1, the combiner Expression is $.0 && $.1.
//!
//! $.a > 5 && $.b < 1 over pack(a: ..., b: ..., c: ...) =>
//! {0: $ > 5, 1: $ < 1} and a combiner $.0 && $.1
//!
//! An invariant that must hold in rules is "combiner expression references
//! exactly the slots it has requested to read from the plan".
//!
//! See register_pack_partition_rules and register_row_idx_partition_rules for
//! examples.

use std::any::Any;
use std::cell::RefCell;
use std::fmt;
use std::hash::Hash;
use std::sync::LazyLock;

use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
use vortex_array::expr as unbound;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::Expression;
use vortex_array::expr::bound::pack;
use vortex_array::expr::traversal::NodeExt;
use vortex_array::expr::traversal::Transformed;
use vortex_array::expr::traversal::TraversalOrder;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnRef;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::fns::get_item::GetItem;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::ArcSwapMap;
use vortex_session::SessionExt;
use vortex_session::SessionVar;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_map::HashMap;
use vortex_utils::aliases::hash_set::HashSet;

use crate::plan::physical::PlanId;
use crate::plan::physical::PlanVTable;
use crate::plan::physical::register_pack_partition_rules;
use crate::plan::physical::register_row_idx_partition_rules;

/// A PartitionedPlan is a plan which has some logical meaning for its slots
/// and allows partitioning an Expression over itself.
pub trait PartitionedPlan: PlanVTable {
    /// Each plan has slots which are numbers. This is the logical type for
    /// these numbers.
    /// For example, Pack has slots 0...n, first one is a validity.
    /// This type for Pack is therefore Slot { Validity, Field(usize) }.
    type Slot: Copy + Into<PlanSlot>;

    fn slot_dtype(scope: &DType, slot: Self::Slot) -> VortexResult<DType>;

    // You likely want to write a PartitionRule instead of overriding this
    // method
    fn partition(
        &self,
        scope: &DType,
        expr: &BoundExpression,
        session: &VortexSession,
    ) -> VortexResult<PartitionedExpression> {
        partition_impl(scope, expr, self.id(), session)
    }
}

pub trait PartitionRule: Send + Sync + 'static {
    type Plan: PartitionedPlan;
    /// For a plan and an expression, answer which plan's slots does this
    /// expression use.
    fn apply(
        &self,
        slots: &Slots<'_, Self::Plan>,
        node: &BoundExpression,
        // Return type is unbound because its type is only known after traversing
        // the whole tree
    ) -> VortexResult<Option<Expression>>;
}

/// Session state for partitioning expression over plans
#[derive(Clone)]
pub struct PartitionSession {
    /// Rules for expression partitioning. $ stores None.
    rules: ArcSwapMap<(PlanId, Option<ScalarFnId>), &'static dyn DynPartitionRule>,
}

impl PartitionSession {
    /// Register a rule which allows to partition $ over a plan.
    /// For example, if you have an expression $ < 5 && $ > 1, this rule will
    /// run on each of $.
    pub fn register_root_rule<Rule: PartitionRule>(&self, plan: Rule::Plan, rule: &'static Rule) {
        self.rules.insert((plan.id(), None), rule);
    }

    /// Register a rule which allows to partition a scalar function over a plan
    pub fn register_rule<Rule: PartitionRule>(
        &self,
        plan: Rule::Plan,
        scalar_fn: impl ScalarFnVTable,
        rule: &'static Rule,
    ) {
        self.rules.insert((plan.id(), Some(scalar_fn.id())), rule);
    }

    fn rule(
        &self,
        plan: PlanId,
        scalar_fn: Option<ScalarFnId>,
    ) -> Option<&'static dyn DynPartitionRule> {
        self.rules.get(&(plan, scalar_fn))
    }
}

/// Struct { i: dtype } for i in dtypes
fn positional_scope(dtypes: Vec<DType>) -> DType {
    DType::Struct(
        StructFields::new((0..dtypes.len()).map(positional_name).collect(), dtypes),
        Nullability::NonNullable,
    )
}

/// Plan's child slot index
pub type PlanSlot = usize;

pub struct Slots<'a, Plan: PartitionedPlan> {
    scope: &'a DType,
    reads: RefCell<HashMap<PlanSlot, Plan::Slot>>,
}

impl<'a, Plan: PartitionedPlan> Slots<'a, Plan> {
    fn new(scope: &'a DType) -> Self {
        Self {
            scope,
            reads: RefCell::new(HashMap::default()),
        }
    }

    /// Plan's type
    pub fn scope(&self) -> &DType {
        self.scope
    }

    pub fn read(&self, slot: Plan::Slot) -> Expression {
        let position = slot.into();
        self.reads.borrow_mut().entry(position).or_insert(slot);
        unbound::get_item(positional_name(position), unbound::root())
    }

    fn into_reads(self) -> VortexResult<HashMap<PlanSlot, DType>> {
        self.reads
            .into_inner()
            .into_iter()
            .map(|(position, slot)| Ok((position, Plan::slot_dtype(self.scope, slot)?)))
            .collect()
    }
}

pub enum PartitionedExpression {
    /// Expression doesn't reference plan's children
    Any(BoundExpression),
    /// Expression references one of plan's children
    One(PlanSlot, BoundExpression),
    /// Expression references many plan's children
    Many(Partitions),
}

pub struct Partitions {
    /// Combine "partitions" back into an Expression
    pub combiner: BoundExpression,
    pub partitions: Vec<Partition>,
}

pub struct Partition {
    pub slot: PlanSlot,
    pub expr: BoundExpression,
}

struct RuleOutput {
    expr: Expression,
    reads: HashMap<PlanSlot, DType>,
}

trait DynPartitionRule: Send + Sync {
    fn run(&self, scope: &DType, node: &BoundExpression) -> VortexResult<Option<RuleOutput>>;
}

impl<Rule: PartitionRule> DynPartitionRule for Rule {
    fn run(&self, scope: &DType, node: &BoundExpression) -> VortexResult<Option<RuleOutput>> {
        let slots = Slots::<Rule::Plan>::new(scope);
        let Some(expr) = self.apply(&slots, node)? else {
            return Ok(None);
        };
        Ok(Some(RuleOutput {
            expr,
            reads: slots.into_reads()?,
        }))
    }
}

fn partition_impl(
    scope: &DType,
    expr: &BoundExpression,
    source: PlanId,
    session: &VortexSession,
) -> VortexResult<PartitionedExpression> {
    let rules = session.get::<PartitionSession>();
    let mut partitioner = Partitioner {
        scope,
        source,
        rules: &rules,
        parts: Ordered::default(),
        substituted: false,
    };
    let root = match partitioner.visit(expr)? {
        Analyzed::None(expr) => return Ok(PartitionedExpression::Any(expr)),
        Analyzed::One(slot, expr) => return Ok(PartitionedExpression::One(slot, expr)),
        Analyzed::Many(root) => root,
    };

    let mut partitions = Vec::with_capacity(partitioner.parts.items.len());
    let mut dtypes = Vec::with_capacity(partitions.capacity());
    let mut flattened = HashSet::default();
    for (i, part) in partitioner.parts.items.into_iter().enumerate() {
        if part.exprs.items.len() == 1 {
            flattened.insert(positional_name(i));
        }
        let expr = pack_by_position(part.exprs.items);
        dtypes.push(expr.dtype().clone());
        partitions.push(Partition {
            slot: part.slot,
            expr,
        });
    }

    let root = if flattened.is_empty() {
        root
    } else {
        flatten_refs(root, &flattened)?
    };
    let mut combiner = root.bind(&positional_scope(dtypes))?;
    if partitioner.substituted {
        combiner = combiner.optimize_recursive()?;
    }
    Ok(PartitionedExpression::Many(Partitions {
        combiner,
        partitions,
    }))
}

fn pack_by_position(mut exprs: Vec<BoundExpression>) -> BoundExpression {
    if exprs.len() == 1 {
        return exprs.pop().vortex_expect("a single child");
    }
    let names = exprs
        .into_iter()
        .enumerate()
        .map(|(sub, expr)| (positional_name(sub), expr));
    pack(names, Nullability::NonNullable)
}

enum Analyzed {
    /// References no children
    None(BoundExpression),
    /// References one child
    One(PlanSlot, BoundExpression),
    /// References many children and is therefore replaced with a new $
    Many(Expression),
}

#[derive(Clone, Copy)]
enum Refs {
    None,
    One(PlanSlot),
    Many,
}

impl Refs {
    fn with(self, child: &Analyzed) -> Self {
        match (self, child) {
            (Refs::Many, _) | (_, Analyzed::Many(_)) => Refs::Many,
            (refs, Analyzed::None(_)) => refs,
            (Refs::None, Analyzed::One(slot, _)) => Refs::One(*slot),
            (Refs::One(current), Analyzed::One(slot, _)) if current == *slot => self,
            (Refs::One(_), Analyzed::One(..)) => Refs::Many,
        }
    }
}

struct Partitioner<'r> {
    scope: &'r DType,
    source: PlanId,
    rules: &'r PartitionSession,
    parts: Ordered<PlanSlot, Part>,
    substituted: bool,
}

struct Part {
    slot: PlanSlot,
    exprs: Ordered<BoundExpression, BoundExpression>,
}

impl Partitioner<'_> {
    fn visit(&mut self, node: &BoundExpression) -> VortexResult<Analyzed> {
        if let Some(output) = self.apply_rule(node)? {
            return self.replaced(output);
        }
        let scalar_fn = node
            .as_scalar()
            .ok_or_else(|| vortex_err!("$ has no partition rule"))?;

        let mut refs = Refs::None;
        let mut children = Vec::with_capacity(node.children().len());
        for child in node.children() {
            let analyzed = self.visit(child)?;
            refs = refs.with(&analyzed);
            children.push(analyzed);
        }
        match refs {
            Refs::None => Ok(Analyzed::None(bound(scalar_fn, children)?)),
            Refs::One(slot) => Ok(Analyzed::One(slot, bound(scalar_fn, children)?)),
            Refs::Many => Ok(Analyzed::Many(self.unbound(scalar_fn, children)?)),
        }
    }

    fn unbound(
        &mut self,
        scalar_fn: &ScalarFnRef,
        children: Vec<Analyzed>,
    ) -> VortexResult<Expression> {
        let children = children
            .into_iter()
            .map(|child| match child {
                Analyzed::None(expr) => to_unbound(&expr),
                Analyzed::One(slot, expr) => Ok(self.cut(slot, expr)),
                Analyzed::Many(expr) => Ok(expr),
            })
            .collect::<VortexResult<Vec<_>>>()?;
        Expression::try_new(scalar_fn.clone(), children)
    }

    fn apply_rule(&self, node: &BoundExpression) -> VortexResult<Option<RuleOutput>> {
        if !node.children().iter().all(BoundExpression::is_root) {
            return Ok(None);
        }
        let scalar_fn = node.as_scalar().map(|scalar_fn| scalar_fn.id());
        let rule = self.rules.rule(self.source, scalar_fn);
        rule.map_or(Ok(None), |rule| rule.run(self.scope, node))
    }

    fn replaced(&mut self, output: RuleOutput) -> VortexResult<Analyzed> {
        let RuleOutput { expr, reads } = output;
        if reads.len() > 1 {
            self.substituted = true;
            return Ok(Analyzed::Many(self.cut_slots(expr, &reads)?));
        }
        match reads.into_iter().next() {
            None => Ok(Analyzed::None(bind_part(&expr, None)?)),
            Some((slot, dtype)) => Ok(Analyzed::One(slot, bind_part(&expr, Some(&dtype))?)),
        }
    }

    fn cut_slots(
        &mut self,
        expr: Expression,
        reads: &HashMap<PlanSlot, DType>,
    ) -> VortexResult<Expression> {
        let cut = expr.transform_down(|node| {
            let Some(slot) = slot_of_read(&node)? else {
                vortex_ensure!(!node.is_root(), "Slots must be read");
                return Ok(Transformed::no(node));
            };
            let dtype = reads
                .get(&slot)
                .ok_or_else(|| vortex_err!("Slot {slot} was not read"))?;
            Ok(replace(
                self.cut(slot, BoundExpression::new_root(dtype.clone())),
            ))
        })?;
        Ok(cut.into_inner())
    }

    fn cut(&mut self, slot: PlanSlot, expr: BoundExpression) -> Expression {
        let part = self.parts.position(slot, || Part {
            slot,
            exprs: Ordered::default(),
        });
        let sub = self.parts.items[part]
            .exprs
            .position(expr.clone(), move || expr);
        unbound::get_item(
            positional_name(sub),
            unbound::get_item(positional_name(part), unbound::root()),
        )
    }
}

fn slot_of_read(expr: &Expression) -> VortexResult<Option<PlanSlot>> {
    let Some(name) = expr.as_opt::<GetItem>() else {
        return Ok(None);
    };
    if !expr.child(0).is_root() {
        return Ok(None);
    }
    name.as_ref()
        .parse()
        .map(Some)
        .map_err(|_| vortex_err!("{name} is not a slot read"))
}

fn bound(scalar_fn: &ScalarFnRef, children: Vec<Analyzed>) -> VortexResult<BoundExpression> {
    let exprs = children
        .into_iter()
        .map(|child| match child {
            Analyzed::None(expr) | Analyzed::One(_, expr) => Ok(expr),
            Analyzed::Many(_) => Err(vortex_err!("multiple slot referencing child")),
        })
        .collect::<VortexResult<Vec<_>>>()?;
    BoundExpression::try_new(scalar_fn.clone(), exprs)
}

fn replace(value: Expression) -> Transformed<Expression> {
    Transformed {
        value,
        order: TraversalOrder::Skip,
        changed: true,
    }
}

fn bind_part(expr: &Expression, root: Option<&DType>) -> VortexResult<BoundExpression> {
    if slot_of_read(expr)?.is_some() {
        let dtype = root.ok_or_else(|| vortex_err!("Part reads a slot it does not have"))?;
        return Ok(BoundExpression::new_root(dtype.clone()));
    }
    let scalar_fn = expr
        .as_scalar()
        .ok_or_else(|| vortex_err!("Slots must be read"))?;
    let children = expr
        .children()
        .iter()
        .map(|child| bind_part(child, root))
        .collect::<VortexResult<Vec<_>>>()?;
    BoundExpression::try_new(scalar_fn.clone(), children)
}

const NAMES_LEN: usize = 100;
static NAMES: LazyLock<[FieldName; NAMES_LEN]> =
    LazyLock::new(|| std::array::from_fn(|position| FieldName::from(format!("{position}"))));
pub(crate) fn positional_name(slot: PlanSlot) -> FieldName {
    let name = NAMES.get(slot);
    name.cloned().unwrap_or_else(|| format!("{slot}").into())
}

struct Ordered<K, V> {
    index: HashMap<K, usize>,
    items: Vec<V>,
}

impl<K, V> Default for Ordered<K, V> {
    fn default() -> Self {
        Self {
            index: HashMap::default(),
            items: Vec::new(),
        }
    }
}

impl<K: Hash + Eq, V> Ordered<K, V> {
    fn position(&mut self, key: K, value: impl FnOnce() -> V) -> usize {
        let items = &mut self.items;
        *self.index.entry(key).or_insert_with(|| {
            items.push(value());
            items.len() - 1
        })
    }
}

fn to_unbound(expr: &BoundExpression) -> VortexResult<Expression> {
    let scalar_fn = expr
        .as_scalar()
        .ok_or_else(|| vortex_err!("Expression without slots references $"))?;
    let children = expr
        .children()
        .iter()
        .map(to_unbound)
        .collect::<VortexResult<Vec<_>>>()?;
    Expression::try_new(scalar_fn.clone(), children)
}

/// References produced in cut() have two levels. If reference is used only
/// once, we can remove one level of indirection here. $.0.x -> $.x
fn flatten_refs(root: Expression, flattened: &HashSet<FieldName>) -> VortexResult<Expression> {
    let flat = root.transform_down(|node| match flat_ref(&node, flattened) {
        Some(inner) => Ok(replace(inner.clone())),
        None => Ok(Transformed::no(node)),
    })?;
    Ok(flat.into_inner())
}

fn flat_ref<'a>(node: &'a Expression, flattened: &HashSet<FieldName>) -> Option<&'a Expression> {
    let name = node.as_opt::<GetItem>()?;
    if *name != positional_name(0) {
        return None;
    }
    let inner = node.child(0);
    let part = inner.as_opt::<GetItem>()?;
    (inner.child(0).is_root() && flattened.contains(part)).then_some(inner)
}

impl Default for PartitionSession {
    fn default() -> Self {
        let this = Self {
            rules: ArcSwapMap::default(),
        };
        register_pack_partition_rules(&this);
        register_row_idx_partition_rules(&this);
        this
    }
}

impl fmt::Debug for PartitionSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PartitionSession").finish_non_exhaustive()
    }
}

impl SessionVar for PartitionSession {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
