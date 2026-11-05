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
//! An invariant that must hold in rules is "the combiner expressions
//! references exactly the slots it has requested to read from the plan".
//!
//! See register_pack_partition_rules and register_row_idx_partition_rules for
//! examples.

use std::any::Any;
use std::borrow::Cow;
use std::cell::RefCell;
use std::fmt;
use std::sync::LazyLock;

use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
use vortex_array::expr as unbound;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::Expression;
use vortex_array::expr::bound::pack;
use vortex_array::scalar_fn::ScalarFnId;
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
    /// Each plan has physical slots. This is the logical type for its slot
    /// numbers.
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

    /// Slot's type
    pub fn dtype(&self, slot: Plan::Slot) -> VortexResult<DType> {
        Plan::slot_dtype(self.scope, slot)
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
    let mut order = linearize(expr, scope, source, &rules)?;
    let mut parts = Parts::default();
    let mut results: Vec<Option<Analyzed>> = Vec::with_capacity(order.len());
    for i in 0..order.len() {
        let analyzed = match order[i].touches.take() {
            Some(output) => replaced(&mut parts, output)?,
            None => analyze(i, &order, &mut results, &mut parts)?,
        };
        results.push(Some(analyzed));
    }

    let root = match results.pop().flatten().vortex_expect("root not traversed") {
        Analyzed::None(expr) => {
            return Ok(PartitionedExpression::Any(expr.into_owned()));
        }
        Analyzed::One(slot, expr) => {
            return Ok(PartitionedExpression::One(slot, expr.into_owned()));
        }
        Analyzed::Many(root) => root,
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
    touches: Option<RuleOutput>,
}

fn linearize<'a>(
    expr: &'a BoundExpression,
    scope: &DType,
    source: PlanId,
    rules: &PartitionSession,
) -> VortexResult<Vec<Entry<'a>>> {
    let mut order = Vec::new();
    let mut stack = vec![(expr, 0usize, 0usize)];
    while let Some((node, size, mark)) = stack.pop() {
        if size == 0 {
            if let Some(touches) = apply_rule(rules, scope, source, node)? {
                order.push(Entry {
                    node,
                    size: 1,
                    touches: Some(touches),
                });
                continue;
            }
            vortex_ensure!(!node.is_root());
        }
        if size < node.children().len() {
            stack.push((node, size + 1, mark));
            stack.push((&node.children()[size], 0, order.len()));
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
    /// References no children
    None(Cow<'a, BoundExpression>),
    /// References one child
    One(PlanSlot, Cow<'a, BoundExpression>),
    /// References many children and is therefore replaced with a new $
    Many(Expression),
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
            Analyzed::None(expr) => changed |= matches!(expr, Cow::Owned(_)),
            Analyzed::One(child_slot, expr) => {
                changed |= matches!(expr, Cow::Owned(_));
                match slot {
                    None => slot = Some(*child_slot),
                    Some(current) if current != *child_slot => many = true,
                    Some(_) => {}
                }
            }
            Analyzed::Many(_) => many = true,
        }
        children.push(analyzed);
    }
    children.reverse();

    if many {
        let children = children
            .into_iter()
            .map(|child| match child {
                Analyzed::None(expr) => to_unbound(&expr),
                Analyzed::One(slot, expr) => Ok(parts.cut(slot, expr.into_owned())),
                Analyzed::Many(expr) => Ok(expr),
            })
            .collect::<VortexResult<Vec<_>>>()?;
        let expr = Expression::try_new(scalar_fn.clone(), children)?;
        return Ok(Analyzed::Many(expr));
    }

    let expr = if changed {
        let exprs = children
            .into_iter()
            .map(|child| match child {
                Analyzed::None(expr) | Analyzed::One(_, expr) => Ok(expr.into_owned()),
                Analyzed::Many(_) => Err(vortex_err!("multiple slot referencing child")),
            })
            .collect::<VortexResult<Vec<_>>>()?;
        Cow::Owned(BoundExpression::try_new(scalar_fn.clone(), exprs)?)
    } else {
        Cow::Borrowed(node)
    };
    Ok(match slot {
        None => Analyzed::None(expr),
        Some(slot) => Analyzed::One(slot, expr),
    })
}

fn replaced<'a>(parts: &mut Parts, output: RuleOutput) -> VortexResult<Analyzed<'a>> {
    let RuleOutput { expr, reads } = output;
    if reads.len() > 1 {
        parts.substituted = true;
        return Ok(Analyzed::Many(cut_slots(&expr, &reads, parts)?));
    }
    match reads.into_iter().next() {
        None => Ok(Analyzed::None(Cow::Owned(bind_part(&expr, None)?))),
        Some((slot, dtype)) => {
            let part = bind_part(&expr, Some(&dtype))?;
            Ok(Analyzed::One(slot, Cow::Owned(part)))
        }
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

fn read_dtype(reads: &HashMap<PlanSlot, DType>, slot: PlanSlot) -> VortexResult<DType> {
    reads
        .get(&slot)
        .cloned()
        .ok_or_else(|| vortex_err!("Slot {slot} was not read"))
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

fn cut_slots(
    expr: &Expression,
    reads: &HashMap<PlanSlot, DType>,
    parts: &mut Parts,
) -> VortexResult<Expression> {
    if let Some(slot) = slot_of_read(expr)? {
        let read = BoundExpression::new_root(read_dtype(reads, slot)?);
        return Ok(parts.cut(slot, read));
    }
    vortex_ensure!(!expr.is_root(), "Slots must be read");
    if expr.children().is_empty() {
        return Ok(expr.clone());
    }
    let children = expr
        .children()
        .iter()
        .map(|child| cut_slots(child, reads, parts))
        .collect::<VortexResult<Vec<_>>>()?;
    expr.clone().with_children(children)
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
    scope: &DType,
    source: PlanId,
    node: &BoundExpression,
) -> VortexResult<Option<RuleOutput>> {
    let scalar_fn = match node.as_scalar() {
        Some(_) if !node.children().iter().all(BoundExpression::is_root) => return Ok(None),
        scalar_fn => scalar_fn.map(|scalar_fn| scalar_fn.id()),
    };
    match rules.rule(source, scalar_fn) {
        Some(rule) => rule.run(scope, node),
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
