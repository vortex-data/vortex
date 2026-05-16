mod ids;
mod output;

pub use ids::*;
pub use output::*;

use crate::ChannelBuffer;
use crate::ChannelSpec;
use crate::OperatorNode;
use crate::ResourceSpec;

#[derive(Default)]
pub struct OperatorGraph {
    nodes: Vec<OperatorNode>,
    channels: Vec<ChannelSpec>,
    resources: Vec<ResourceSpec>,
}

impl OperatorGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_operator(&mut self, node: OperatorNode) -> OperatorId {
        let id = OperatorId::from_index(self.nodes.len());
        self.nodes.push(node);
        id
    }

    pub fn add_resource(&mut self, resource: ResourceSpec) {
        self.resources.push(resource);
    }

    pub fn connect(&mut self, from: OperatorId, to: Vec<InputPortRef>, buffer: ChannelBuffer) {
        let label = format!("channel_{}", from.index());
        self.connect_named(label, from, to, buffer);
    }

    pub fn connect_named(
        &mut self,
        label: impl Into<String>,
        from: OperatorId,
        to: Vec<InputPortRef>,
        buffer: ChannelBuffer,
    ) {
        self.channels.push(ChannelSpec::single_producer(
            label, from, to, buffer,
        ));
    }

    /// Multi-producer wiring: N producers feed one channel whose
    /// consumers are listed in `to`. Subsumes the
    /// `Union`-as-operator pattern — instead of inserting a Union
    /// node, attach N upstream operators to one downstream input
    /// port via a single multi-producer channel. The channel
    /// translates each pushed batch's span via its own output
    /// cursor so downstream sees monotonic batch spans regardless
    /// of arrival order.
    pub fn connect_multi(
        &mut self,
        from: Vec<OperatorId>,
        to: Vec<InputPortRef>,
        buffer: ChannelBuffer,
    ) {
        let label = format!(
            "channel_multi_{}",
            from.first().map(|o| o.index()).unwrap_or(0)
        );
        self.connect_multi_named(label, from, to, buffer);
    }

    pub fn connect_multi_named(
        &mut self,
        label: impl Into<String>,
        from: Vec<OperatorId>,
        to: Vec<InputPortRef>,
        buffer: ChannelBuffer,
    ) {
        assert!(
            !from.is_empty(),
            "connect_multi: at least one producer required"
        );
        self.channels
            .push(ChannelSpec::multi_producer(label, from, to, buffer));
    }

    /// Connect with an attached projection expression. The producer's
    /// batches have `array.apply(&expression)` run on push, before
    /// the batch lands in the channel. Subsumes a separate Project
    /// operator — projection is just data attached to the channel.
    /// Pass `root()` / `"$"` to skip (or use `connect` directly).
    pub fn connect_with_projection(
        &mut self,
        from: OperatorId,
        to: Vec<InputPortRef>,
        buffer: ChannelBuffer,
        projection: vortex_array::expr::Expression,
    ) {
        let label = format!("channel_proj_{}", from.index());
        self.connect_with_projection_named(label, from, to, buffer, projection);
    }

    pub fn connect_with_projection_named(
        &mut self,
        label: impl Into<String>,
        from: OperatorId,
        to: Vec<InputPortRef>,
        buffer: ChannelBuffer,
        projection: vortex_array::expr::Expression,
    ) {
        self.channels.push(
            ChannelSpec::single_producer(label, from, to, buffer)
                .with_projection(projection),
        );
    }

    /// Reference an operator's single output port. Equivalent to the
    /// operator id; kept as a helper so call sites read consistently.
    pub fn output(operator: OperatorId) -> OperatorId {
        operator
    }

    pub fn input(operator: OperatorId, port: usize) -> InputPortRef {
        InputPortRef::new(operator, InputPortId::from_index(port))
    }

    pub(crate) fn into_parts(self) -> (Vec<OperatorNode>, Vec<ChannelSpec>, Vec<ResourceSpec>) {
        (self.nodes, self.channels, self.resources)
    }

    /// Borrow the operator nodes in insertion order. Useful for
    /// debug dumps / inspection tools that want to walk the graph
    /// before consuming it via `prepare`.
    pub fn nodes(&self) -> &[OperatorNode] {
        &self.nodes
    }

    /// Borrow the channel specs. Each spec records `from` (a
    /// producer op) and `to` (a list of consumer input ports).
    pub fn channels(&self) -> &[ChannelSpec] {
        &self.channels
    }

    /// Print the operator graph in a human-readable form. Intended
    /// for debugging / docs — operators listed by id, channels
    /// listed by `producer → consumers`. Per-port input/output
    /// labels come from `OperatorSpec`.
    pub fn dump(&self) -> String {
        use std::fmt::Write;
        let mut out = String::new();
        writeln!(out, "operators ({}):", self.nodes.len()).ok();
        for (i, node) in self.nodes.iter().enumerate() {
            let spec = node.spec();
            let inputs: Vec<String> = spec
                .inputs
                .iter()
                .map(|p| {
                    format!(
                        "{}({:?},rows={:?})",
                        p.name,
                        p.domain.id().as_str(),
                        p.domain.cardinality()
                    )
                })
                .collect();
            let output_label = spec
                .output
                .as_ref()
                .map(|p| {
                    format!(
                        "{}({:?},rows={:?})",
                        p.name,
                        p.domain.id().as_str(),
                        p.domain.cardinality()
                    )
                })
                .unwrap_or_else(|| "—".to_string());
            writeln!(
                out,
                "  op#{i:<3} {label:<48}  in=[{ins}]  out={output_label}",
                label = spec.label,
                ins = inputs.join(", "),
            )
            .ok();
        }
        writeln!(out, "channels ({}):", self.channels.len()).ok();
        for (i, ch) in self.channels.iter().enumerate() {
            let to: Vec<String> = ch
                .to
                .iter()
                .map(|p| format!("op#{}.in{}", p.operator().index(), p.port().index()))
                .collect();
            let from: Vec<String> = ch
                .from
                .iter()
                .map(|o| format!("op#{}", o.index()))
                .collect();
            writeln!(
                out,
                "  ch#{i:<3} {label:<48}  [{from}] → [{to}]  ({topo:?})",
                label = ch.label,
                from = from.join(", "),
                to = to.join(", "),
                topo = ch.topology,
            )
            .ok();
        }
        out
    }
}
