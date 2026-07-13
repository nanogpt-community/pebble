use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// A cheap, cloneable cancellation signal shared by every operation in a turn.
#[derive(Debug, Clone, Default)]
pub struct CancellationToken {
    pub(crate) cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    #[doc(hidden)]
    #[must_use]
    pub fn atomic_flag(&self) -> Arc<AtomicBool> {
        self.cancelled.clone()
    }

    pub(crate) fn from_atomic_flag(cancelled: Arc<AtomicBool>) -> Self {
        Self { cancelled }
    }
}

thread_local! {
    static ACTIVE_CANCELLATION: RefCell<Option<CancellationToken>> = const { RefCell::new(None) };
}

/// Makes the current turn's token available to deeply nested native tools.
pub struct ActiveCancellationGuard {
    previous: Option<CancellationToken>,
}

#[must_use]
pub fn set_active_cancellation(token: CancellationToken) -> ActiveCancellationGuard {
    let previous = ACTIVE_CANCELLATION.with(|active| active.replace(Some(token)));
    ActiveCancellationGuard { previous }
}

#[must_use]
pub fn active_cancellation() -> Option<CancellationToken> {
    ACTIVE_CANCELLATION.with(|active| active.borrow().clone())
}

impl Drop for ActiveCancellationGuard {
    fn drop(&mut self) {
        ACTIVE_CANCELLATION.with(|active| {
            active.replace(self.previous.take());
        });
    }
}
