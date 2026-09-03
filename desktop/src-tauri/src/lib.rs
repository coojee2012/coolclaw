use std::sync::Mutex;
use tauri::{Emitter, Manager, Wry};
use tauri::menu::{MenuBuilder, MenuItemBuilder};
use tauri::tray::TrayIconBuilder;
use tauri_plugin_shell::process::{CommandChild, CommandEvent};
use tauri_plugin_shell::ShellExt;

struct BackendState(Mutex<Option<CommandChild>>);

fn is_port_in_use(port: u16) -> bool {
    std::net::TcpStream::connect(("127.0.0.1", port)).is_ok()
}

fn spawn_and_observe(
    shell: &tauri_plugin_shell::Shell<Wry>,
    exe: &str,
    args: &[&str],
    handle: &tauri::AppHandle,
) -> CommandChild {
    let (mut rx, child) = shell
        .command(exe)
        .args(args)
        .spawn()
        .unwrap_or_else(|e| panic!("Failed to spawn {}: {}", exe, e));

    let h = handle.clone();
    tauri::async_runtime::spawn(async move {
        while let Some(event) = rx.recv().await {
            match event {
                CommandEvent::Stdout(line) => {
                    println!("[Backend] {}", String::from_utf8_lossy(&line));
                }
                CommandEvent::Stderr(line) => {
                    eprintln!("[Backend] {}", String::from_utf8_lossy(&line));
                }
                CommandEvent::Terminated(status) => {
                    eprintln!("[Backend] Exited: {:?}", status);
                    let _ = h.emit("backend-stopped", ());
                }
                _ => {}
            }
        }
    });

    child
}

fn find_project_root() -> Option<std::path::PathBuf> {
    let exe_dir = std::env::current_exe().ok()?.parent()?.to_path_buf();
    let mut candidate = exe_dir;
    for _ in 0..10 {
        if candidate.join("main.py").exists() && candidate.join(".venv").exists() {
            return Some(candidate);
        }
        if !candidate.pop() {
            break;
        }
    }
    None
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(BackendState(Mutex::new(None)))
        .setup(|app| {
            let shell = app.shell();
            let handle = app.handle().clone();
            let backend_state = app.state::<BackendState>();

            let port: u16 = 8484;

            if is_port_in_use(port) {
                println!("[CoolClaw] Port {} already in use — assuming external backend is running", port);
            } else {
                match shell.sidecar("backend") {
                    Ok(cmd) => {
                        let (mut rx, child) = cmd
                            .args(["--port", &port.to_string()])
                            .spawn()
                            .expect("Failed to spawn backend sidecar");

                        let h = handle.clone();
                        tauri::async_runtime::spawn(async move {
                            while let Some(event) = rx.recv().await {
                                match event {
                                    CommandEvent::Stdout(line) => {
                                        println!("[Backend] {}", String::from_utf8_lossy(&line));
                                    }
                                    CommandEvent::Stderr(line) => {
                                        eprintln!("[Backend] {}", String::from_utf8_lossy(&line));
                                    }
                                    CommandEvent::Terminated(status) => {
                                        eprintln!("[Backend] Exited: {:?}", status);
                                        let _ = h.emit("backend-stopped", ());
                                    }
                                    _ => {}
                                }
                            }
                        });

                        *backend_state.0.lock().unwrap() = Some(child);
                        println!("[CoolClaw] Backend sidecar started on http://localhost:{}", port);
                    }
                    Err(e) => {
                        eprintln!("[CoolClaw] Sidecar failed: {}. Falling back to dev mode", e);
                        if let Some(root) = find_project_root() {
                            let python_bin = root.join(".venv/bin/python");
                            if python_bin.exists() {
                                let child = spawn_and_observe(
                                    &shell,
                                    python_bin.to_str().unwrap_or("python3"),
                                    &["main.py", "--port", &port.to_string()],
                                    &handle,
                                );
                                *backend_state.0.lock().unwrap() = Some(child);
                                println!("[CoolClaw] Backend (dev mode) started on http://localhost:{}", port);
                            }
                        }
                    }
                }
            }

            let show_item = MenuItemBuilder::with_id("show", "Show Window").build(app)?;
            let hide_item = MenuItemBuilder::with_id("hide", "Hide Window").build(app)?;
            let quit_item = MenuItemBuilder::with_id("quit", "Quit CoolClaw").build(app)?;
            let menu = MenuBuilder::new(app)
                .item(&show_item)
                .item(&hide_item)
                .separator()
                .item(&quit_item)
                .build()?;

            let _tray = TrayIconBuilder::new()
                .icon(app.default_window_icon().unwrap().clone())
                .menu(&menu)
                .tooltip("CoolClaw AI Coding Agent")
                .on_menu_event(move |app, event| match event.id.as_ref() {
                    "show" => {
                        if let Some(w) = app.get_webview_window("main") {
                            let _ = w.show();
                            let _ = w.set_focus();
                        }
                    }
                    "hide" => {
                        if let Some(w) = app.get_webview_window("main") {
                            let _ = w.hide();
                        }
                    }
                    "quit" => app.exit(0),
                    _ => {}
                })
                .on_tray_icon_event(|tray, event| {
                    if let tauri::tray::TrayIconEvent::Click {
                        button: tauri::tray::MouseButton::Left,
                        button_state: tauri::tray::MouseButtonState::Up,
                        ..
                    } = event
                    {
                        let app = tray.app_handle();
                        if let Some(w) = app.get_webview_window("main") {
                            let _ = w.show();
                            let _ = w.set_focus();
                        }
                    }
                })
                .build(app)?;

            #[cfg(debug_assertions)]
            {
                if let Some(w) = app.get_webview_window("main") {
                    w.open_devtools();
                }
            }

            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                let _ = window.hide();
                api.prevent_close();
            }
        })
        .build(tauri::generate_context!())
        .expect("error while running tauri application")
        .run(|app_handle, event| {
            if let tauri::RunEvent::Exit = event {
                let child = app_handle
                    .state::<BackendState>()
                    .0
                    .lock()
                    .ok()
                    .and_then(|mut g| g.take());
                if let Some(child) = child {
                    let _ = child.kill();
                    println!("[CoolClaw] Backend stopped");
                }
            }
        });
}
