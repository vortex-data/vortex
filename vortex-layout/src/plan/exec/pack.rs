// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::plan::PackPlan;
use crate::plan::exec::ExecNode;
use crate::plan::exec::NodeState;
use crate::plan::exec::StepCx;
use crate::plan::exec::selection::Selection;
use crate::plan::exec::selection::join;

/// Zips its fields into struct arrays as their rows arrive.
///
/// Every field runs over the same rows and selection, so the fields' outputs have the same
/// length and line up row for row. Whenever every port has rows available, the node takes as
/// many rows as the shortest port holds from each of them and emits one struct. Fields chunked
/// alike therefore stream one struct per chunk; fields chunked differently are sliced at the
/// boundaries they share.
pub(crate) struct PackNode {
    plan: PackPlan,
    selection: Selection,
    ports: usize,
}

impl PackNode {
    pub(crate) fn new(plan: PackPlan, selection: Selection) -> Self {
        let ports = plan.children().len();
        Self {
            plan,
            selection,
            ports,
        }
    }

    /// Builds a struct of `len` rows from `arrays`, one per port, the last being the validity
    /// when the struct is nullable.
    fn assemble(&self, mut arrays: Vec<ArrayRef>, len: usize) -> VortexResult<StructArray> {
        let validity = if self.plan.dtype().is_nullable() {
            Validity::Array(
                arrays
                    .pop()
                    .ok_or_else(|| vortex_err!("Nullable Pack is missing its validity port"))?,
            )
        } else {
            Validity::NonNullable
        };
        StructArray::try_new_with_dtype(arrays, self.plan.fields().clone(), len, validity)
    }

    fn field_dtype(&self, port: usize) -> DType {
        self.plan
            .fields()
            .field_by_index(port)
            .unwrap_or(DType::Bool(Nullability::NonNullable))
    }
}

impl ExecNode for PackNode {
    fn start(&mut self, cx: &mut StepCx<'_>) -> VortexResult<NodeState> {
        if self.ports == 0 {
            // A struct with no fields still has rows, so no child can carry them.
            let len = self.selection.mask().true_count();
            cx.emit(self.assemble(Vec::new(), len)?.into_array());
            return Ok(NodeState::Done);
        }
        if self.selection.mask().all_false() {
            return Ok(NodeState::Done);
        }
        for port in 0..self.ports {
            cx.spawn(
                port,
                self.plan.child_required(port)?,
                self.selection.rows().clone(),
                self.selection.mask().clone(),
            );
        }
        Ok(NodeState::Wait)
    }

    fn compute(&mut self, cx: &mut StepCx<'_>) -> VortexResult<NodeState> {
        loop {
            let len = cx
                .inputs()
                .iter()
                .map(|input| input.available())
                .min()
                .unwrap_or(0);
            if len == 0 {
                break;
            }
            let mut arrays = Vec::with_capacity(self.ports);
            for port in 0..self.ports {
                let dtype = self.field_dtype(port);
                arrays.push(join(&dtype, cx.input(port).take(len)?)?);
            }
            cx.emit(self.assemble(arrays, len)?.into_array());
        }
        if cx.all_finished() {
            return Ok(NodeState::Done);
        }
        Ok(NodeState::Wait)
    }
}
