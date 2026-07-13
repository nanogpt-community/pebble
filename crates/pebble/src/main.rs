mod app;
mod eval;
mod eval_runner;
#[cfg(test)]
mod golden_tests;
mod grok_acp;
mod init;
mod input;
mod interrupt;
mod mcp;
mod model_catalog;
mod models;
mod provider_auth;
mod provider_diagnostics;
mod proxy;
mod render;
mod report;
mod runtime_client;
mod session_store;
mod tool_render;
mod trace_view;
mod ui;

fn main() {
    ui::configure_color_output();
    let default_panic_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        if !is_broken_pipe_panic(info.payload()) {
            default_panic_hook(info);
        }
    }));

    match std::panic::catch_unwind(app::run) {
        Ok(Ok(())) => {}
        Ok(Err(error)) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
        Err(payload) if is_broken_pipe_panic(payload.as_ref()) => {}
        Err(payload) => std::panic::resume_unwind(payload),
    }
}

fn is_broken_pipe_panic(payload: &(dyn std::any::Any + Send)) -> bool {
    payload
        .downcast_ref::<String>()
        .is_some_and(|message| message.contains("Broken pipe"))
        || payload
            .downcast_ref::<&str>()
            .is_some_and(|message| message.contains("Broken pipe"))
}

#[cfg(test)]
mod tests {
    use super::is_broken_pipe_panic;

    #[test]
    fn identifies_only_broken_pipe_panics() {
        let broken_pipe = String::from("failed printing to stdout: Broken pipe (os error 32)");
        let other = String::from("unexpected parser failure");

        assert!(is_broken_pipe_panic(&broken_pipe));
        assert!(!is_broken_pipe_panic(&other));
    }
}
