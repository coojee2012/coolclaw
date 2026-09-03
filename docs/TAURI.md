# Tauri 桌面版打包指南

## 架构设计

```
┌─────────────────────────────────────────────┐
│           Tauri 桌面应用                      │
│  ┌─────────────────────────────────────┐   │
│  │  前端 (web/*)                       │   │
│  │  - HTMX + Alpine.js                │   │
│  │  - TailwindCSS                     │   │
│  │  - 通过 localhost 与后端通信        │   │
│  └─────────────────────────────────────┘   │
│                                             │
│  Rust 后端 (src-tauri/*)                   │
│  - 窗口管理                                │
│  - 系统集成                                │
│  - 自动启动 Python 后端服务               │
└─────────────────────────────────────────────┘
           ↕ localhost
┌─────────────────────────────────────────────┐
│     Python FastAPI 后台服务 (独立进程)       │
│  - AI 模型调度 (llama.cpp)                 │
│  - 任务管理 + 定时执行                     │
│  - API 接口                                │
└─────────────────────────────────────────────┘
```

## 快速开始

### 1. 安装依赖

```bash
# macOS
brew install rustup
rustup-init
rustup default stable

# 安装 Tauri CLI
cargo install tauri-cli

# 安装前端工具
npm install -g pnpm
```

### 2. 创建 Tauri 项目结构

```bash
# 在项目根目录创建 src-tauri
mkdir -p src-tauri/src
mkdir -p src-tauri/icons
```

### 3. Cargo.toml

```toml
[package]
name = "opencode-helper"
version = "0.1.0"
edition = "2021"

[build-dependencies]
tauri-build = "1.5"

[dependencies]
tauri = { version = "1.6", features = ["shell-open", "window-close", "window-minimize", "window-maximize"] }
tauri-plugin-shell = "1.0"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
```

### 4. main.rs

```rust
use tauri::{Manager, WindowEvent};

fn main() {
    tauri::Builder::default()
        .setup(|app| {
            let window = app.get_window("main").unwrap();
            
            // 启动 Python 后端服务
            std::process::Command::new("python3")
                .args(&["-m", "src.cli", "serve"])
                .spawn()
                .expect("Failed to start backend server");
            
            // 打开前端
            window.load_url("http://localhost:8484").unwrap();
            
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

### 5. tauri.conf.json

```json
{
  "build": {
    "distDir": "../web",
    "devPath": "http://localhost:8484",
    "beforeDevCommand": "",
    "beforeBuildCommand": ""
  },
  "package": {
    "productName": "OpenCode Helper",
    "version": "0.1.0"
  },
  "tauri": {
    "bundle": {
      "identifier": "com.opencode.helper",
      "icon": [
        "icons/32x32.png",
        "icons/128x128.png",
        "icons/128x128@2x.png"
      ],
      "targets": "all",
      "macOS": {
        "minimumSystemVersion": "10.15"
      }
    },
    "windows": [
      {
        "title": "OpenCode Helper",
        "width": 1200,
        "height": 800,
        "resizable": true,
        "fullscreen": false,
        "center": true
      }
    ]
  }
}
```

## 打包命令

```bash
# 开发模式
cargo tauri dev

# 构建发布版本
cargo tauri build
```

## 优化建议

### 1. 内嵌模型文件
```toml
[bundle]
resources = [
    "/Volumes/LynnData/myclaw_models/*"
]
```

### 2. 首次运行向导
- 模型下载提示
- API Key 配置引导
- 默认参数设置

### 3. 后台运行选项
```rust
// 后台静默启动 Python 服务
std::process::Command::new("python3")
    .args(&["-m", "src.cli", "serve"])
    .spawn()
    .ok();
```

## 后续功能扩展

### 可选：内嵌 Python 运行时
- 使用 PyOxidizer 将 Python 打包进 Tauri
- 无需用户安装 Python 环境

### 可选：WebView2 (Windows)
- 使用系统 WebView2
- 减少打包体积

## 文件清单

需要创建的文件：
```
opencode_helper/
├── src-tauri/
│   ├── Cargo.toml
│   ├── build.rs
│   ├── tauri.conf.json
│   ├── src/
│   │   └── main.rs
│   └── icons/
│       ├── 32x32.png
│       ├── 128x128.png
│       └── 128x128@2x.png
└── web/  (现有)
```

## 注意事项

1. **端口占用**：Tauri 默认使用 WebView，如需调试可配置 `devtools: true`
2. **模型路径**：打包时注意外部 SSD 模型路径的处理
3. **数据迁移**：用户数据在 `~/.opencode_helper/`，打包时需要考虑迁移

## 参考文档

- [Tauri 官方文档](https://tauri.app/)
- [Tauri + Python 集成](https://tauri.app/guides/building)
