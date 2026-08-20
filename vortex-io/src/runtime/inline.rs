// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hint::spin_loop;
use std::pin::pin;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::task::Context;
use std::task::Poll;
use std::task::Wake;
use std::task::Waker;

/// Drives a future synchronously until it completes.
///
/// This is only suitable for futures whose progress does not depend on yielding control to an
/// external event loop. It is used by the WebAssembly runtime when JavaScript interop is disabled.
pub(super) fn block_on<F: Future>(future: F) -> F::Output {
    let mut future = pin!(future);
    let wake_state = Arc::new(SpinWaker::default());
    let waker = Waker::from(Arc::clone(&wake_state));
    let mut context = Context::from_waker(&waker);

    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(output) => return output,
            Poll::Pending => {
                while !wake_state.woken.swap(false, Ordering::Acquire) {
                    spin_loop();
                }
            }
        }
    }
}

#[derive(Default)]
struct SpinWaker {
    woken: AtomicBool,
}

impl Wake for SpinWaker {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.woken.store(true, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use std::task::Poll;

    use futures::future::poll_fn;

    use crate::runtime::inline::block_on;

    #[test]
    fn test_block_on_repolls_pending_future() {
        let mut first_poll = true;
        let result = block_on(poll_fn(|context| {
            if first_poll {
                first_poll = false;
                context.waker().wake_by_ref();
                Poll::Pending
            } else {
                Poll::Ready(42)
            }
        }));

        assert_eq!(result, 42);
    }
}
