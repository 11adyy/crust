# crust

crust is an experimental voxel sandbox and rendering engine written in Rust. It combines a GPU-driven renderer with background world generation and mesh building, so terrain can stream around the player while the main loop handles input, simulation and presentation. The sandbox layer provides block interaction, inventory and tools, dropped items, world persistence and an early TCP multiplayer implementation.

The renderer uses wgpu, packed quad descriptors, shared GPU arenas, compute visibility culling and indirect draws. The world layer generates deterministic terrain from a seed, then tracks player changes separately for saving. These systems are designed for continuous exploration rather than loading the entire world in advance.

The project is under active development. Rendering internals, APIs, save data and gameplay behavior can change between revisions. Multiplayer currently supplies transport and synchronization foundations; it does not provide a complete authoritative simulation.

## Demo

<https://github.com/user-attachments/assets/3f86d46e-7a33-4144-ae3d-f78887f2b1a7>

## Building and running

### Requirements

- **Current stable Rust with Rust 2024 edition support.** Edition support starts at Rust 1.85; current dependencies can require a newer compiler, so use the current stable toolchain rather than assuming 1.85 can build the locked dependency graph.
- A GPU and driver supported by **wgpu**.
- Windows is the primary development/release target at the moment.
- Linux/macOS may work through wgpu/winit, but are not guaranteed to be tested on every revision.

### Clone and run

```bash
git clone https://github.com/11adyy/crust.git
cd crust

cargo run --release
```

For development:

```bash
cargo run
```

Release mode is strongly recommended when evaluating rendering or chunk-generation
performance.

### Build only

```bash
cargo build --release
```

### Run tests

```bash
cargo test
```

### Formatting and linting

```bash
cargo fmt
cargo clippy --all-targets
```


## Controls and inventory input

| Input | Action |
|---|---|
| `W A S D` | Move |
| `Space` | Jump |
| `Left Shift` | Sprint |
| `Left Ctrl` | Sneak |
| Mouse | Look around |
| `LMB` | Mine / break targeted block |
| `RMB` | Place selected block |
| `E` | Open / close inventory |
| `1`–`9` | Select hotbar slot |
| Mouse wheel | Cycle hotbar |
| `Q` | Drop one selected item |
| `Ctrl + Q` | Drop selected stack |
| `Esc` | Close inventory / return to menu |
| `F1` | Toggle crosshair/HUD element |
| `C` | Toggle debug/performance overlay |
| `F4` | Toggle Hi-Z occlusion culling |
| `F5` | Save world |
| `F9` | Load world |
| `F11` | Toggle borderless fullscreen |

Inventory UI also supports left/right click behavior and Shift + left-click quick moves.


## Running a dedicated server

Start a headless TCP server on the default port (`25565`):

```bash
cargo run --release -- --server
```

Use a custom port:

```bash
cargo run --release -- --server --port 12345
```

The dedicated server does not create a game window.


## Engine data flow and architecture

### High-level data flow

1. Player and camera state drive world streaming. The streaming layer detects missing chunks, places requests in a distance-priority generation queue and dispatches background `ChunkGenerator` workers.
2. Generated world, chunk and subchunk data feed dirty-mesh tracking. Mesh versions travel with background mesh-worker jobs so stale results can be rejected.
3. Mesh workers produce `PackedQuad` streams. Terrain and water geometry enter separate shared GPU arenas.
4. Compute culling tests frustum visibility and Hi-Z occlusion, then writes indirect draw-command buffers.
5. The opaque terrain pass produces scene depth. Depth resolve and Hi-Z pyramid generation prepare visibility information for subsequent work.
6. The transparent water pass feeds the composite and post-processing stage. UI and text are drawn before the frame is presented.

### Chunk streaming

`ChunkLoader` uses background worker threads and a shared priority heap. Lower squared
distance to the player means higher urgency. Duplicate in-flight requests are prevented
with a pending set, and the queue is bounded to avoid unbounded memory growth.

### Mesh streaming

Mesh creation is decoupled from chunk generation.

A subchunk carries a mesh revision/version. When a worker finishes, the result is accepted
only if its revision still matches the current world state. This prevents a mesh generated
from stale neighbor/block data from overwriting a newer version.

### GPU geometry storage

Terrain and water use persistent packed-quad arenas instead of creating a GPU buffer for
every chunk.

The allocator supports:

- sub-allocation,
- free blocks,
- arena growth,
- compaction,
- culling-metadata refresh after relocation.

This keeps the renderer suitable for continuous load/unload cycles while the player moves
through the world.

### GPU-driven visibility

Per-subchunk metadata contains:

- world-space AABB,
- terrain draw range,
- water draw range.

The compute culling pass tests the AABB and emits indirect draw commands only for visible
subchunks. On GPUs that support `MULTI_DRAW_INDIRECT_COUNT`, the GPU also controls how many
draws are executed.


## Rendering

- **wgpu 30.0.1** renderer with Vulkan / Direct3D 12 / Metal backend support through wgpu.
- **GPU-driven terrain submission** using compute-generated visibility lists.
- **`multi_draw_indirect_count`** when supported by the active GPU, with a fallback path.
- **GPU frustum culling** for subchunks.
- **Hi-Z occlusion culling** using a hierarchical depth pyramid.
- **Packed quad vertex pulling** instead of a conventional per-subchunk vertex buffer layout.
- **Growable shared quad arenas** for terrain and water geometry.
- **Free-list allocation and arena compaction** for long-running chunk streaming.
- **Greedy meshing** on the CPU mesh path to merge compatible voxel faces.
- **Asynchronous mesh workers** with versioned mesh results so stale work can be rejected safely.
- Separate **opaque terrain** and **transparent water** rendering paths.
- Water shading with **screen-space reflection data, refraction, Fresnel blending, and foam/edge logic**.
- **4× MSAA** in the current renderer.
- Procedural sky/sun rendering.
- Full-screen composite stage with effects such as **underwater fog/color grading and vignette**.
- GPU timestamp profiling and an in-game performance/debug overlay.

## World generation and streaming

- Deterministic seed-based procedural generation.
- Chunk columns are **16×16 blocks** horizontally.
- World height is **256 blocks**.
- Each chunk column contains **16 subchunks**, each **16×16×16**.
- Multi-threaded chunk generation through a priority queue.
- Distance-prioritized generation requests.
- Separate generation, render, simulation, and unload distances.
- Caves, terrain variation, vegetation, water bodies, and procedural features.
- Current biome set:
    - Plains
    - Forest
    - Desert
    - Tundra
    - Mountains
    - Swamp
    - Ocean
    - Beach
    - River
    - Lake
    - Island
- Biome-dependent grass/leaf tinting and vegetation density.
- Automatic chunk unloading outside the configured streaming radius.
- Dirty-subchunk tracking and incremental remeshing after world changes.

## Blocks and world interaction

The current block model includes ordinary cubes as well as several special cases.

Examples include:

`Grass`, `Dirt`, `Stone`, `Sand`, `Water`, `Wood`, `Leaves`, `Bedrock`,
`Snow`, `Gravel`, `Clay`, `Ice`, `Cactus`, `DeadBush`, `WoodStairs`,
`WoodLogX`, and `WoodLogZ`.

Implemented interaction systems include:

- block breaking with per-block break times,
- block placement from the selected hotbar item,
- collision-aware placement,
- repeated straight-line placement while RMB is held,
- block loot/drop handling,
- dropped item entities with simple world physics and pickup behavior,
- block face visibility rules for transparent and partial blocks.

## Inventory and items

The gameplay inventory stores blocks and items, routes slot transactions and tracks the selected hotbar slot. Its interaction model includes cursor stacks and quick moves between containers.

- **27-slot main inventory**
- **9-slot hotbar**
- stack merging and maximum stack sizes,
- selected hotbar slot,
- left-click / right-click inventory interaction,
- shift-click quick move between main inventory and hotbar,
- dropping one item or a full stack,
- cursor-stack restoration when the inventory closes,
- hotbar-first insertion for freshly mined drops,
- stable item resource keys for save data,
- item registry with multiple item kinds:
    - blocks,
    - tools,
    - food,
    - materials/generic items,
- tool durability support,
- loot tables for block drops.

The codebase also contains a **furnace inventory/container scaffold** with typed slot
rules. Furnace simulation itself is still a work in progress.

## Saving world state

World saves use a compact binary format through `postcard`.

The save system stores:

- world seed,
- player position,
- player rotation,
- inventory and selected hotbar slot,
- item durability,
- player-modified chunks/subchunks.

Procedural terrain that was not modified by the player can be regenerated from the seed,
which keeps save files smaller than serializing the entire loaded world.

Default save file:

```text
world.crust
```

## Multiplayer protocol and limits

Multiplayer is currently an **early-stage TCP implementation**.

Implemented foundations include:

- headless dedicated server mode,
- TCP client/server transport,
- length-prefixed binary packet protocol,
- server-assigned player IDs,
- connection acknowledgement with the server world seed,
- position synchronization,
- rotation synchronization,
- block-change packets,
- chat packets,
- ping/pong packets,
- disconnect propagation,
- remote player rendering/name labels.

The current dedicated server primarily validates/stamps player identity and relays
packets between clients. It is **not yet a complete authoritative world-simulation server**.

## World and streaming constants

The active world-streaming values are currently compile-time constants in
`src/constants.rs`.

| Constant | Current value | Meaning |
|---|---:|---|
| `WORLD_HEIGHT` | `256` | World height in blocks |
| `CHUNK_SIZE` | `16` | Horizontal chunk size |
| `SUBCHUNK_HEIGHT` | `16` | Vertical subchunk size |
| `NUM_SUBCHUNKS` | `16` | Vertical subchunks per chunk column |
| `RENDER_DISTANCE` | `32` | Render radius in chunks |
| `SIMULATION_DISTANCE` | `16` | Simulation radius |
| `GENERATION_DISTANCE` | `34` | Generation/prefetch radius |
| `CHUNK_UNLOAD_DISTANCE` | `37` | Chunk eviction radius |
| `SEA_LEVEL` | `64` | Sea level |
| `MAX_CHUNKS_PER_FRAME` | `8` | Chunk generation request budget |
| `MAX_CHUNK_COMMITS_PER_FRAME` | `2` | Completed chunk commit budget |
| `MAX_MESH_BUILDS_PER_FRAME` | `8` | Mesh request budget |
| `MAX_MESH_COMMITS_PER_FRAME` | `2` | Finished mesh/GPU commit budget |

Worker counts are selected from the available CPU count and clamped to keep the main
thread responsive.


## Source layout

```text
crust/
├── assets/                     # Texture atlas, fonts, menu assets
├── assets_docs/                # Documentation assets
├── src/
│   ├── app/                    # Application state, game loop, rendering orchestration
│   │   ├── game.rs             # Event loop, CLI, controls, save/load hotkeys
│   │   ├── init.rs             # GPU/window/pipeline initialization
│   │   ├── input.rs            # Gameplay/menu/inventory input
│   │   ├── render.rs           # Frame render graph / passes
│   │   ├── update.rs           # Simulation, streaming and mesh commits
│   │   ├── state.rs            # Main runtime State
│   │   └── server.rs           # Dedicated server entry
│   ├── core/
│   │   ├── block.rs            # Block types and physical/render properties
│   │   ├── biome.rs            # Biome definitions
│   │   ├── chunk.rs            # Chunk/SubChunk storage and metadata
│   │   ├── item.rs             # Item registry, tools, food, loot
│   │   └── mobs/               # Early mob/AI scaffolding
│   ├── multiplayer/
│   │   ├── protocol.rs         # Binary packet protocol
│   │   ├── tcp.rs              # TCP transport implementation
│   │   ├── client.rs           # Multiplayer client
│   │   ├── server.rs           # Server-side network primitives
│   │   └── network.rs          # Game/network integration
│   ├── player/
│   │   ├── camera.rs           # Camera, movement and collision
│   │   ├── inventory/          # Inventory/container transaction model
│   │   └── player_stats.rs     # Player statistics/state
│   ├── render/
│   │   ├── indirect.rs         # GPU arenas, metadata, culling and indirect draws
│   │   ├── mesh_loader.rs      # Background mesh workers
│   │   ├── mesh.rs             # Mesh helpers and models
│   │   ├── quad.rs             # PackedQuad representation
│   │   ├── frustum.rs          # AABB/frustum math
│   │   ├── texture.rs          # Texture handling
│   │   └── atlas_map.rs        # Texture atlas mapping
│   ├── shaders/
│   │   ├── terrain.wgsl
│   │   ├── water.wgsl
│   │   ├── cull.wgsl
│   │   ├── hiz.wgsl
│   │   ├── depth_resolve.wgsl
│   │   ├── sky.wgsl
│   │   ├── sun.wgsl
│   │   ├── composite.wgsl
│   │   ├── outline.wgsl
│   │   └── ui.wgsl
│   ├── ui/                     # Menu, HUD, inventory and text UI
│   ├── utils/                  # Settings and GPU/system helpers
│   ├── world/
│   │   ├── generator.rs        # Procedural terrain generator
│   │   ├── loader.rs           # Async priority chunk loader
│   │   ├── terrain.rs          # World data + meshing snapshots/building
│   │   ├── spline.rs           # Terrain interpolation/splines
│   │   ├── item_entity.rs      # Dropped item entities
│   │   └── structures/         # Structure framework/basic structures
│   ├── save.rs                 # Postcard world/inventory persistence
│   ├── constants.rs
│   ├── lib.rs
│   └── main.rs
├── Cargo.toml
├── docs/
│   ├── DEVELOPMENT.md
│   ├── DOCUMENTATION_MAP.md
│   └── FOLDER_STRUCTURE.md
└── README.md
```


## Dependency reference

| Crate | Version | Purpose |
|---|---:|---|
| `wgpu` | `30.0.1` | Cross-platform GPU API |
| `winit` | `0.30.13` | Windowing and input |
| `glam` | `0.33.6` | Vector/matrix math |
| `glyphon` | `0.12.0` | GPU text rendering |
| `tokio` | `1.50` | Async networking/runtime |
| `fastnoise-lite` | `1.1` | Procedural noise |
| `postcard` | `1.1` | Compact save serialization |
| `crossbeam-channel` | `0.5` | Worker communication |
| `parking_lot` | `0.12` | Synchronization |
| `rustc-hash` | `2` | Fast hash collections |
| `clap` | `4.4` | CLI argument parsing |

See [`Cargo.toml`](Cargo.toml) for the complete dependency list.


## Profiling and performance work

crust is designed around profiling rather than fixed performance claims.

The engine contains instrumentation for:

- overall frame time,
- CPU-side update/render sections,
- process CPU time on Windows,
- GPU timestamps,
- chunk-generation backlog,
- mesh streaming work,
- visibility/culling statistics.

Use a **release build** for meaningful performance measurements:

```bash
cargo run --release
```

The current architecture is specifically intended to reduce:

- per-chunk CPU draw submission,
- unnecessary hidden geometry rendering,
- synchronous terrain generation stalls,
- per-chunk GPU buffer allocation overhead.

Actual FPS, RAM use, and VRAM use depend heavily on render distance, resolution,
GPU, world complexity, and current development state.


## Incomplete and experimental systems

Several parts of the repository are intentionally incomplete or experimental:

- `gpu_mesher.rs` contains an experimental compute-based face extraction implementation,
  but it is not currently wired into the public `render` module/main rendering path.
- Mob classes and passive AI scaffolding exist, but there is no complete mob gameplay loop yet.
- Furnace/container slot rules exist, but furnace processing is not implemented yet.
- Multiplayer is functional as a transport/synchronization foundation, but server-authoritative
  gameplay/world simulation is not complete.
- Streaming, meshing prioritization, and Hi-Z behavior are actively being optimized.
- Some older module-level documentation may lag behind the current code; the source code is the
  authoritative reference.


## Planned work

Near-term areas that fit the current architecture:

- [ ] Priority-based mesh streaming to favor missing/near-camera subchunks
- [ ] More robust temporal Hi-Z occlusion handling
- [ ] Temporal anti-aliasing (TAA)
- [ ] Dynamic/local light sources
- [ ] Expanded global illumination / ambient lighting
- [ ] Crafting and furnace simulation
- [ ] Complete tool/food gameplay behavior
- [ ] Mob spawning and AI
- [ ] More structures and world-generation features
- [ ] More authoritative multiplayer world state
- [ ] Better network delta/state synchronization
- [ ] Continued RAM/VRAM reduction and streaming optimization


## Documentation

Additional documentation is available in:

- [`DEVELOPMENT.md`](docs/DEVELOPMENT.md)
- [`DOCUMENTATION_MAP.md`](docs/DOCUMENTATION_MAP.md)
- [`FOLDER_STRUCTURE.md`](docs/FOLDER_STRUCTURE.md)
- [`src/app/README.md`](src/app/README.md)
- [`src/core/README.md`](src/core/README.md)
- [`src/render/README.md`](src/render/README.md)
- [`src/world/README.md`](src/world/README.md)
- [`src/multiplayer/README.md`](src/multiplayer/README.md)
- [`src/player/README.md`](src/player/README.md)
- [`src/ui/README.md`](src/ui/README.md)

Because the project changes quickly, some detailed module documentation can become stale.
When documentation and implementation disagree, prefer the current source code.


## Contributing

Contributions are welcome.

Recommended workflow:

```bash
git checkout -b feature/my-change

cargo fmt
cargo clippy --all-targets
cargo test

git commit -m "feat: describe the change"
```

Please keep changes focused and update relevant documentation when changing public behavior
or architecture.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for additional project guidelines.


## Build automation and archives

The repository contains two GitHub Actions workflows. The CI workflow runs on pushes and pull requests, installs stable Rust, checks the locked dependency graph and runs the test suite on Linux and Windows. Its purpose is to check source changes on both supported build environments; it does not measure GPU performance or validate interactive gameplay.

The Windows packaging workflow can be started manually from the Actions tab and also runs when a version tag matching `v*` is pushed. It builds `crust.exe` for `x86_64-pc-windows-msvc`, then packages the executable, the complete `assets/` directory, the README and the MIT license into `crust-windows.zip`. The archive is retained as an Actions artifact. A version-tag run also attaches it to a GitHub Release.

Extract the archive into a directory before starting the executable. Keep `assets/` beside `crust.exe` and launch from that directory, because the texture loader uses relative asset paths. The embedded shaders, menu background and font do not remove the runtime texture requirement.

The source ZIP available from GitHub contains the source files and assets but does not include a compiled executable. Use Cargo to build that source archive. Version tags identify source snapshots; an executable release exists only after its packaging job succeeds.

## License

crust is licensed under the **MIT License**.

See [`LICENSE`](LICENSE) for details.

## Operating notes

For development, run the debug build while editing behavior, then switch to `cargo run --release` when comparing rendering or streaming performance. Compiler optimization changes the CPU cost of procedural generation and meshing, so debug and release frame times are not directly comparable. Record the render distance, resolution, GPU, seed and scene when comparing two revisions.

The server command is separate from the graphical client entry path. `--server` selects headless TCP operation and `--port` changes the listener port. Running a dedicated server does not turn the relay into an authoritative world simulator; keep that distinction in mind when extending validation, persistence or gameplay rules.

Streaming distances and per-frame budgets currently live in `src/constants.rs`. Changing a radius affects how many chunk columns can remain active, while commit budgets control how much completed background work reaches the main thread per frame. Review generation, simulation, rendering and eviction distances together when tuning memory pressure or exploration responsiveness.

Keep a backup of important save files before changing versions. The default filename is `world.crust`; seed-based regeneration reduces the stored data, but inventory resource keys and player-modified subchunks still depend on the current serialization and registry definitions. The project does not promise compatibility with save files from every historical revision.

If a packaged build cannot load textures, confirm that the ZIP was extracted and that `assets/textures.png` is available relative to the working directory. For shader validation, inventory behavior or save-code changes, run the tests as well as building the executable. An automated build checks compilation and unit behavior; visual rendering, window input and multiplayer sessions need an actual runtime environment.
