use runtime::{CancellationToken, RuntimeError};
use signal_hook::consts::SIGINT;
use signal_hook::SigId;

/// Installs a scoped Ctrl+C handler for the whole active agent turn.
pub(crate) struct InterruptGuard {
    signal_id: SigId,
}

impl InterruptGuard {
    pub(crate) fn install(cancellation: &CancellationToken) -> Result<Self, RuntimeError> {
        let signal_id =
            signal_hook::flag::register(SIGINT, cancellation.atomic_flag()).map_err(|error| {
                RuntimeError::new(format!("could not install interrupt handler: {error}"))
            })?;
        Ok(Self { signal_id })
    }
}

impl Drop for InterruptGuard {
    fn drop(&mut self) {
        signal_hook::low_level::unregister(self.signal_id);
    }
}
