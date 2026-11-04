// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! We have Plans which tell us how data is stored and how to read it. An
//! Expression is a subset of a Plan which tells us how to read the data but
//! doesn't know how the data is stored.
//!
//! Given an Expression graph and a Plan, we want to know which Expressions in
//! this graph are disjoint i.e. operate over non-overlapping slots (children)
//! of a Plan. There may be particular Expressions which provide us with this
//! information.
//!
//! For example, if we have a PackPlan(a: ..., b: ..., c: ...) and an Expression
//! $.a > 5 && $.b < 1, we know that $.a > 5 (or, formally, gt(get_item("a", $),
//! 5)) and $.b < 1 don't overlap, and we never read "c". We know this because
//! get_item(x, $) tell us parts of $ it needs.
//!
//! Thus we partition the Expression into two parts: $.a > 5 references child
//! plan for "a" of PackPlan, and we rewrite the expression over this plan to $
//! > 5, same for $.b < 1.
//!
//!     $.a > 5 over pack(a: child_plan) => $ > 5 over child_plan
//!
//! Now we also need a way to combine these parts back into an Expression. If we
//! call these partitions 0 and 1, the combiner Expression is $.0 && $.1.
//!
//!     $.a > 5 && $.b < 1 over pack(a: c1, b: c2) => {0: $ > 5, 1: $ < 1} and a
//!     combiner $.0 && $.1
//!
//! You can register your plan as a PartitionedPlan with a set of its slots
//! (children) that can be referenced in Expressions. Referencing child slots of
//! a Plan with positional indices is cumbersome so there is a convenience trait
//! PartitionedPlanSlot.
//!
//!     For example, for a PackPlan the first slot is always the validity slot,
//!     and i'th field is referenced by i + 1. PartitionedPlanSlot { Validity,
//!     Field(usize) } is easier to read for a programmer.
//!
//! Then you can write a PartitionRule for this Plan which would make an
//! Expression the partition point if it's found in expression graph. In this
//! rule you say "if I find this Expression, it needs the following input slots
//! from this Plan" via NodeNeeds.
//!
//!     For example, get_item("x", $) over a PackPlan needs only field x and its
//!     corresponding slot. select("a, b", $) over the same plan would need
//!     slots for a and b.
//!
//! See register_pack_partition_rules and register_row_idx_partition_rules for
//! examples.

use std::any::Any;
use std::borrow::Cow;
use std::fmt;
use std::marker::PhantomData;
use std::sync::LazyLock;

use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
use vortex_array::expr as unbound;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::Expression;
use vortex_array::expr::bound::get_item;
use vortex_array::expr::bound::pack;
use vortex_array::scalar_fn::ScalarFnId;
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

pub trait PartitionRule<Plan: PartitionedPlan>: Send + Sync {
    /// Apply a rule to partition an expression with a plan
    fn apply(&self, node: &BoundExpression) -> VortexResult<Option<NodeNeeds<Plan::Slot>>>;
}

pub trait PartitionedPlan: PlanVTable {
    type Slot: PartitionedPlanSlot;

    // You likely want to write a PartitionRule instead of overriding this
    // method
    fn partition(
        &self,
        expr: &BoundExpression,
        session: &VortexSession,
    ) -> VortexResult<PartitionedExpression> {
        partition_impl(expr, self.id(), session)
    }
}

/// Plan's child slot index
pub type PlanSlot = usize;

/// Plan's child slot, a convenience trait.
/// For example, for Pack you can write PartitionedPlanSlot =
/// Slot { Validity, Field(i) } and read Slot::Validity through
/// NodeNeeds instead of a positional index.
pub trait PartitionedPlanSlot: Into<PlanSlot> + Copy + PartialEq {}

/// Slots a node needs and an expression that combines reads from them.
/// Reads are addressed by slot.
pub struct NodeNeeds<Slot = PlanSlot> {
    slots: Vec<Slot>,
    scope: BoundExpression,
    combiner: BoundExpression,
}

impl<Slot: PartitionedPlanSlot> NodeNeeds<Slot> {
    /// Needs "slots" and combines them with the scope itself
    pub fn new(slots: impl IntoIterator<Item = (Slot, DType)>) -> Self {
        let (slots, mut dtypes): (Vec<Slot>, Vec<DType>) = slots.into_iter().unzip();
        let scope = if dtypes.len() == 1 {
            BoundExpression::new_root(dtypes.remove(0))
        } else {
            BoundExpression::new_root(positional_scope(dtypes))
        };
        let combiner = scope.clone();
        Self {
            slots,
            scope,
            combiner,
        }
    }

    /// Read "slot" from the scope
    pub fn get(&self, slot: Slot) -> VortexResult<BoundExpression> {
        let position = self
            .slots
            .iter()
            .position(|candidate| *candidate == slot)
            .ok_or_else(|| vortex_err!("Slot {} is not in the scope", slot.into()))?;
        if self.slots.len() == 1 {
            return Ok(self.scope.clone());
        }
        Ok(get_item(positional_name(position), self.scope.clone()))
    }

    pub fn with_combiner(self, combiner: BoundExpression) -> Self {
        Self { combiner, ..self }
    }
}

/// Result of partitioning an expression over a plan
pub enum PartitionedExpression {
    /// If None, expression references none of plan's chidlren.
    /// If Some, expression references one of plan's children
    One(Option<PlanSlot>, BoundExpression),
    /// Expression references multiple plan's children
    Many(Partitions),
}

pub struct Partitions {
    /// Expression that combines partitions
    pub combiner: BoundExpression,
    pub partitions: Vec<Partition>,
}

pub struct Partition {
    pub slot: PlanSlot,
    pub expr: BoundExpression,
}

pub trait DynPartitionRule: Send + Sync {
    fn apply(&self, node: &BoundExpression) -> VortexResult<Option<NodeNeeds>>;
}

pub struct PartitionRuleAdapter<Plan, Rule> {
    rule: Rule,
    _plan: PhantomData<fn() -> Plan>,
}

/// Session state for partitioning expression over plans
#[derive(Clone)]
pub struct PartitionSession {
    /// Rules for expression partitioning. Root expression ($) stores None.
    rules: ArcSwapMap<(PlanId, Option<ScalarFnId>), &'static dyn DynPartitionRule>,
}

impl PartitionSession {
    /// Register a rule which allows to partition $ over a plan.
    /// For example, if you have an expression $ < 5 && $ > 1, this rule will
    /// run on each of $.
    pub fn register_root_rule<Plan: PartitionedPlan, Rule: PartitionRule<Plan>>(
        &self,
        plan: &Plan,
        rule: &'static PartitionRuleAdapter<Plan, Rule>,
    ) {
        self.rules.insert((plan.id(), None), rule);
    }

    /// Register a rule which allows to partition a scalar function over a plan
    pub fn register_rule<Plan: PartitionedPlan, Rule: PartitionRule<Plan>>(
        &self,
        plan: &Plan,
        scalar_fn: ScalarFnId,
        rule: &'static PartitionRuleAdapter<Plan, Rule>,
    ) {
        self.rules.insert((plan.id(), Some(scalar_fn)), rule);
    }

    pub(crate) fn rule(
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

pub(crate) fn scope_of(expr: &BoundExpression) -> Option<&DType> {
    if expr.is_root() {
        return Some(expr.dtype());
    }
    expr.children().iter().find_map(scope_of)
}

fn partition_impl(
    expr: &BoundExpression,
    source: PlanId,
    session: &VortexSession,
) -> VortexResult<PartitionedExpression> {
    let rules = session.get::<PartitionSession>();
    let mut order = linearize(expr, source, &rules)?;
    let mut parts = Parts::default();
    let mut results: Vec<Option<Analyzed>> = Vec::with_capacity(order.len());
    for i in 0..order.len() {
        let analyzed = match order[i].touches.take() {
            Some(touches) => replaced(&mut parts, touches)?,
            None => analyze(i, &order, &mut results, &mut parts)?,
        };
        results.push(Some(analyzed));
    }

    let root = match results.pop().flatten().vortex_expect("root not traversed") {
        Analyzed::One(slot, expr) => {
            return Ok(PartitionedExpression::One(slot, expr.into_owned()));
        }
        Analyzed::Root(root) => root,
    };

    let mut partitions = Vec::with_capacity(parts.parts.len());
    let mut flattened = HashSet::default();
    for (i, part) in parts.parts.into_iter().enumerate() {
        let mut exprs = part.exprs.into_iter();
        let expr = if exprs.len() == 1 {
            flattened.insert(positional_name(i));
            exprs.next().vortex_expect("a single child")
        } else {
            let names = exprs
                .enumerate()
                .map(|(sub, expr)| (positional_name(sub), expr));
            pack(names, Nullability::NonNullable)
        };
        partitions.push(Partition {
            slot: part.slot,
            expr,
        });
    }

    let dtypes: Vec<DType> = partitions
        .iter()
        .map(|partition| partition.expr.dtype().clone())
        .collect();
    let root = if flattened.is_empty() {
        root
    } else {
        flatten_refs(&root, &flattened)?
    };
    let mut combiner = root.bind(&positional_scope(dtypes))?;
    if parts.substituted {
        combiner = combiner.optimize_recursive()?;
    }
    Ok(PartitionedExpression::Many(Partitions {
        combiner,
        partitions,
    }))
}

struct Entry<'a> {
    node: &'a BoundExpression,
    size: usize,
    touches: Option<NodeNeeds>,
}

fn linearize<'a>(
    expr: &'a BoundExpression,
    source: PlanId,
    rules: &PartitionSession,
) -> VortexResult<Vec<Entry<'a>>> {
    let mut order = Vec::new();
    let mut stack = vec![(expr, 0usize, 0usize)];
    while let Some((node, child, mark)) = stack.pop() {
        if child == 0 {
            if let Some(touches) = apply_rule(rules, source, node)? {
                order.push(Entry {
                    node,
                    size: 1,
                    touches: Some(touches),
                });
                continue;
            }
            vortex_ensure!(!node.is_root());
        }
        if child < node.children().len() {
            stack.push((node, child + 1, mark));
            stack.push((&node.children()[child], 0, order.len()));
        } else {
            order.push(Entry {
                node,
                size: order.len() - mark + 1,
                touches: None,
            });
        }
    }
    Ok(order)
}

enum Analyzed<'a> {
    /// References at most one field from children.
    One(Option<PlanSlot>, Cow<'a, BoundExpression>),
    /// References many fields from children and is therefore replaced with a
    /// new root expression.
    Root(Expression),
}

fn analyze<'a>(
    i: usize,
    order: &[Entry<'a>],
    results: &mut [Option<Analyzed<'a>>],
    parts: &mut Parts,
) -> VortexResult<Analyzed<'a>> {
    let node = order[i].node;
    let scalar_fn = node.as_scalar().vortex_expect("not a scalar function");

    let mut slot = None;
    let mut many = false;
    let mut changed = false;
    let mut children = Vec::with_capacity(node.children().len());
    let mut next = i;
    for _ in 0..node.children().len() {
        let index = next - 1;
        next -= order[index].size;
        let analyzed = results[index].take().vortex_expect("children are analyzed");
        match &analyzed {
            Analyzed::One(child_slot, expr) => {
                changed |= matches!(expr, Cow::Owned(_));
                match (slot, *child_slot) {
                    (None, child_slot) => slot = child_slot,
                    (Some(current), Some(child_slot)) if current != child_slot => many = true,
                    _ => {}
                }
            }
            Analyzed::Root(_) => many = true,
        }
        children.push(analyzed);
    }
    children.reverse();

    if many {
        let children = children
            .into_iter()
            .map(|child| match child {
                Analyzed::One(None, expr) => to_unbound(&expr, None),
                Analyzed::One(Some(slot), expr) => Ok(parts.cut(slot, expr.into_owned())),
                Analyzed::Root(expr) => Ok(expr),
            })
            .collect::<VortexResult<Vec<_>>>()?;
        let expr = Expression::try_new(scalar_fn.clone(), children)?;
        return Ok(Analyzed::Root(expr));
    }

    if !changed {
        return Ok(Analyzed::One(slot, Cow::Borrowed(node)));
    }

    let exprs = children
        .into_iter()
        .map(|child| match child {
            Analyzed::One(_, expr) => Ok(expr.into_owned()),
            Analyzed::Root(_) => Err(vortex_err!("multiple slot referencing child")),
        })
        .collect::<VortexResult<Vec<_>>>()?;
    let expr = Cow::Owned(BoundExpression::try_new(scalar_fn.clone(), exprs)?);
    Ok(Analyzed::One(slot, expr))
}

fn replaced<'a>(parts: &mut Parts, touches: NodeNeeds) -> VortexResult<Analyzed<'a>> {
    let NodeNeeds {
        slots, combiner, ..
    } = touches;
    if slots.len() <= 1 {
        let idx = slots.first().copied();
        let combiner = Cow::Owned(combiner);
        return Ok(Analyzed::One(idx, combiner));
    }
    let fields = scope_of(&combiner)
        .and_then(DType::as_struct_fields_opt)
        .ok_or_else(|| vortex_err!("Non-existent field"))?;
    vortex_ensure!(fields.nfields() == slots.len());
    parts.substituted = true;

    let elements =
        slots
            .into_iter()
            .zip(fields.fields())
            .enumerate()
            .map(|(position, (slot, dtype))| {
                let read = parts.cut(slot, BoundExpression::new_root(dtype));
                (positional_name(position), read)
            });
    let expr = unbound::pack(elements, Nullability::NonNullable);
    Ok(Analyzed::Root(to_unbound(&combiner, Some(&expr))?))
}

const NAMES_LEN: usize = 100;
static NAMES: LazyLock<[FieldName; NAMES_LEN]> =
    LazyLock::new(|| std::array::from_fn(|position| FieldName::from(format!("{position}"))));
pub(crate) fn positional_name(slot: PlanSlot) -> FieldName {
    let name = NAMES.get(slot);
    name.cloned().unwrap_or_else(|| format!("{slot}").into())
}

fn apply_rule(
    rules: &PartitionSession,
    source: PlanId,
    node: &BoundExpression,
) -> VortexResult<Option<NodeNeeds>> {
    let rule = rules.rule(source, node.as_scalar().map(|scalar_fn| scalar_fn.id()));
    match rule {
        Some(rule) => rule.apply(node),
        None => Ok(None),
    }
}

struct PartBuild {
    slot: PlanSlot,
    positions: HashMap<BoundExpression, usize>,
    exprs: Vec<BoundExpression>,
}

impl PartBuild {
    fn push_dedup(&mut self, expr: BoundExpression) -> usize {
        let exprs = &mut self.exprs;
        *self.positions.entry(expr.clone()).or_insert_with(|| {
            exprs.push(expr);
            exprs.len() - 1
        })
    }
}

#[derive(Default)]
struct Parts {
    index: HashMap<PlanSlot, usize>,
    parts: Vec<PartBuild>,
    substituted: bool,
}

impl Parts {
    fn cut(&mut self, slot: PlanSlot, expr: BoundExpression) -> Expression {
        let part = *self.index.entry(slot).or_insert_with(|| {
            self.parts.push(PartBuild {
                slot,
                positions: HashMap::default(),
                exprs: Vec::new(),
            });
            self.parts.len() - 1
        });
        let sub = self.parts[part].push_dedup(expr);
        unbound::get_item(
            positional_name(sub),
            unbound::get_item(positional_name(part), unbound::root()),
        )
    }
}

fn to_unbound(expr: &BoundExpression, scope: Option<&Expression>) -> VortexResult<Expression> {
    let Some(scalar_fn) = expr.as_scalar() else {
        return scope.cloned().ok_or_else(|| vortex_err!("scope reference"));
    };
    let children = expr
        .children()
        .iter()
        .map(|child| to_unbound(child, scope))
        .collect::<VortexResult<Vec<_>>>()?;
    Expression::try_new(scalar_fn.clone(), children)
}

/// References produced in cut() have two levels. If reference is used only
/// once, we can remove one level of indirection here. $.0.x -> $.x
fn flatten_refs(node: &Expression, flattened: &HashSet<FieldName>) -> VortexResult<Expression> {
    if let Some(name) = node.as_opt::<GetItem>()
        && *name == positional_name(0)
        && let Some(child_name) = node.child(0).as_opt::<GetItem>()
        && node.child(0).child(0).is_root()
        && flattened.contains(child_name)
    {
        return Ok(node.child(0).clone());
    }
    if node.children().is_empty() {
        return Ok(node.clone());
    }
    let children = node
        .children()
        .iter()
        .map(|child| flatten_refs(child, flattened))
        .collect::<VortexResult<Vec<_>>>()?;
    node.clone().with_children(children)
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

impl<Plan, Rule> PartitionRuleAdapter<Plan, Rule> {
    pub const fn new(rule: Rule) -> Self {
        Self {
            rule,
            _plan: PhantomData,
        }
    }
}

impl<Plan, Rule> DynPartitionRule for PartitionRuleAdapter<Plan, Rule>
where
    Plan: PartitionedPlan,
    Rule: PartitionRule<Plan>,
{
    fn apply(&self, node: &BoundExpression) -> VortexResult<Option<NodeNeeds>> {
        let rule = self.rule.apply(node)?.map(|touches| NodeNeeds {
            slots: touches.slots.into_iter().map(Into::into).collect(),
            scope: touches.scope,
            combiner: touches.combiner,
        });
        Ok(rule)
    }
}
