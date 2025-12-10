//! This is just a stoway for some ideas about changing how we execute scans to more easily
//! be execute in a run loop, so we can execute them using something like iouring.

enum StepResult<T> {
    Needs { segments: Vec<SegmentId> },
    Done { result: VortexResult<T> },
}

trait ReaderPlan {
    fn step(
        &mut self,
        ctx: &ArrayContext,
        segments: &HashMap<SegmentId, ByteBuffer>,
    ) -> StepResult<ArrayRef>;
}

struct FlatReaderPlan(FlatLayout);

impl ReaderPlan for FlatReaderPlan {
    fn step(
        &mut self,
        ctx: &ArrayContext,
        segments: &HashMap<SegmentId, ByteBuffer>,
    ) -> StepResult<ArrayRef> {
        // Get access to the current set of buffers.
        let segment_id = self.0.segment_id();
        let Some(segment) = segments.get(&segment_id) else {
            return StepResult::Needs {
                segments: vec![segment_id],
            };
        };

        let segment = segment.clone();
        // Attempt to deserialize this array instead.
        let result = ArrayParts::try_from(segment).and_then(|array_parts| {
            array_parts.decode(ctx, self.0.dtype(), self.0.row_count() as usize)
        });

        StepResult::Done { result }
    }
}

struct ChunkedReaderPlan {
    layout: ChunkedLayout,
    children: Vec<Box<dyn ReaderPlan>>,
}

impl ReaderPlan for ChunkedReaderPlan {
    fn step(
        &mut self,
        ctx: &ArrayContext,
        segment_map: &HashMap<SegmentId, ByteBuffer>,
    ) -> StepResult<ArrayRef> {
        let mut requests = vec![];
        let mut chunks = vec![];

        for child in self.children.iter_mut() {
            match child.step(ctx, segment_map) {
                StepResult::Needs { segments } => {
                    requests.extend(segments);
                }
                StepResult::Done { result } => match result {
                    Ok(array) => {
                        chunks.push(array);
                    }
                    Err(err) => {
                        return StepResult::Done {
                            result: VortexResult::Err(err),
                        };
                    }
                },
            }
        }

        if requests.is_empty() {
            StepResult::Done {
                result: ChunkedArray::try_new(chunks, self.layout.dtype().clone())
                    .map(|chunked| chunked.into_array()),
            }
        } else {
            StepResult::Needs { segments: requests }
        }
    }
}

fn execute(mut reader: Box<dyn ReaderPlan>) -> VortexResult<ArrayRef> {
    // Attempt to execute a reader to completion.
    // This means dispatching it using something that looks a lot like a normal scan executor.
    // We can force the scan to run in parallel if we want it to.
    // The layouts don't actually make a lot of sense this way.
    // We create a new ReaderPlan, which resembles the current thing.
    let ctx = ArrayContext::empty();
    let segments = HashMap::new();

    // We can ask the plan if it has any prefetch that it knows about.
    // We might want to do this by asking for ourselves here.

    loop {
        match reader.step(&ctx, &segments) {
            StepResult::Needs { segments } => {
                // Dispatch reads for these segments.
                // Pre-fetching is a bit of a weird thing. It basically is just something
                // that we can support by asking for it.
            }
            StepResult::Done { result } => {
                return result;
            }
        }
    }
}

