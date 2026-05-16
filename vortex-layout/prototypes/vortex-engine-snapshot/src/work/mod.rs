use crate::InputPortId;

/// Operator-defined opaque work key.
///
/// The scheduler treats this as opaque bytes; `Operator::run` decodes
/// it back into the operator's private work enum. Small enough to be
/// cheap to clone in proposal queues.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct WorkKey(Vec<u8>);

impl WorkKey {
    pub fn from_byte(tag: u8) -> Self {
        Self(vec![tag])
    }

    pub fn from_bytes(bytes: impl Into<Vec<u8>>) -> Self {
        Self(bytes.into())
    }

    pub fn tag(&self) -> u8 {
        self.0.first().copied().unwrap_or_default()
    }

    pub fn bytes(&self) -> &[u8] {
        &self.0
    }
}

/// Coarse priority band for forward work admission.
///
/// Class priority is the deterministic tie-breaker; EV breaks ties
/// within a class. Operators must remain correct even if the EV
/// score is wrong.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum WorkClass {
    PublishResource,
    Seal,
    Release,
    Emit,
    Cpu,
    SubmitBroker,
}

impl WorkClass {
    pub const fn priority(self) -> i64 {
        match self {
            Self::PublishResource => 6,
            Self::Seal => 5,
            Self::Release => 4,
            Self::Emit => 3,
            Self::Cpu => 2,
            Self::SubmitBroker => 1,
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::PublishResource => "publish_resource",
            Self::Seal => "seal",
            Self::Release => "release",
            Self::Emit => "emit",
            Self::Cpu => "cpu",
            Self::SubmitBroker => "submit_broker",
        }
    }
}

/// Expected-value information used for within-class ranking.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct WorkValue {
    pub required_rows: u64,
    pub candidate_rows: u64,
    /// Probability that candidate rows will turn out to be needed,
    /// expressed as a fixed-point [0, 256] value.
    pub p_needed_x256: u32,
    pub memory_release_bytes: u64,
}

impl WorkValue {
    pub const fn empty() -> Self {
        Self {
            required_rows: 0,
            candidate_rows: 0,
            p_needed_x256: 0,
            memory_release_bytes: 0,
        }
    }

    pub const fn required(rows: u64) -> Self {
        Self {
            required_rows: rows,
            candidate_rows: 0,
            p_needed_x256: 256,
            memory_release_bytes: 0,
        }
    }

    pub const fn candidate(rows: u64, p_needed_x256: u32) -> Self {
        Self {
            required_rows: 0,
            candidate_rows: rows,
            p_needed_x256,
            memory_release_bytes: 0,
        }
    }

    pub const fn release(bytes: u64) -> Self {
        Self {
            required_rows: 0,
            candidate_rows: 0,
            p_needed_x256: 0,
            memory_release_bytes: bytes,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct WorkCost {
    pub cpu_micros: u32,
    pub memory_delta_bytes: i64,
}

impl WorkCost {
    pub const fn small_cpu() -> Self {
        Self {
            cpu_micros: 1,
            memory_delta_bytes: 0,
        }
    }

    pub const fn small_emit(bytes: i64) -> Self {
        Self {
            cpu_micros: 1,
            memory_delta_bytes: bytes,
        }
    }

    pub const fn release(bytes: i64) -> Self {
        Self {
            cpu_micros: 1,
            memory_delta_bytes: -bytes,
        }
    }
}

/// Hard constraints the scheduler checks before admitting a proposal.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct WorkConstraints {
    /// True if this work needs the operator's single output port to
    /// have capacity. Operators have at most one output, so this is
    /// just a boolean.
    pub needs_output_capacity: bool,
    pub needs_input_data: Option<InputPortId>,
}

impl WorkConstraints {
    pub const fn none() -> Self {
        Self {
            needs_output_capacity: false,
            needs_input_data: None,
        }
    }

    pub const fn output_capacity() -> Self {
        Self {
            needs_output_capacity: true,
            needs_input_data: None,
        }
    }

    pub const fn input_data(port: InputPortId) -> Self {
        Self {
            needs_output_capacity: false,
            needs_input_data: Some(port),
        }
    }

    pub const fn input_and_output(input: InputPortId) -> Self {
        Self {
            needs_output_capacity: true,
            needs_input_data: Some(input),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkProposal {
    pub key: WorkKey,
    pub class: WorkClass,
    pub value: WorkValue,
    pub cost: WorkCost,
    pub constraints: WorkConstraints,
}

impl WorkProposal {
    pub fn new(
        key: WorkKey,
        class: WorkClass,
        value: WorkValue,
        cost: WorkCost,
        constraints: WorkConstraints,
    ) -> Self {
        Self {
            key,
            class,
            value,
            cost,
            constraints,
        }
    }

    /// EV score for within-class ranking. Linear combination of value
    /// and cost terms. Operators must remain correct under any score.
    pub fn ev_score(&self) -> i64 {
        let value_score = (self.value.required_rows as i64).saturating_mul(256)
            + (self.value.candidate_rows as i64)
                .saturating_mul(self.value.p_needed_x256 as i64)
            + (self.value.memory_release_bytes as i64) / 64;
        let cost_score = self.cost.cpu_micros as i64
            + self.cost.memory_delta_bytes.max(0) / 64;
        value_score.saturating_sub(cost_score)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WorkStatus {
    /// Forward progress was made; the operator may have more work to
    /// propose on the next `update`.
    Made,
    /// The operator has nothing more to propose. Outputs sealed,
    /// inputs drained or released.
    Finished,
}
