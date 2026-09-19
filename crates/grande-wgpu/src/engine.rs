//! The forward pass on wgpu. Weights and a token-capacity-sized workspace live
//! on the device for the life of the engine; a request writes its token ids
//! and (position, sequence) metadata, records one command buffer for the
//! whole network, and reads back the requested rows.
//!
//! Weights arrive one tensor at a time through `EngineBuilder::push` (a
//! multi-GB checkpoint never has to sit in host memory at once), then
//! `finish` allocates the workspace and wires the bind groups.

use std::collections::HashMap;

use anyhow::{anyhow, bail, Context, Result};
use bytemuck::{Pod, Zeroable};
use grande_core::{BranchOutput, BranchTokens, Want};
use half::f16;
use wgpu::util::DeviceExt;

use crate::model::{Config, Dtype, QTensor, Weights};

const PARAM_SLOT: u64 = 256;

/// Workgroup memory the attention kernel declares at a head_dim (see
/// attention.wgsl): Q tile, K tile, scores, positions.
fn attn_workgroup_bytes(hd: usize) -> u32 {
    let (rows, kb) = attn_tile(hd);
    (rows * hd / 2 * 4 + kb * hd / 2 * 4 + 128 * 4 + kb * 8 + rows * 8 + 4) as u32
}

/// Attention workgroup shape (query rows, keys per tile) per head_dim; the
/// product is the 128-invocation workgroup.
fn attn_tile(hd: usize) -> (usize, usize) {
    if hd >= 512 {
        (16, 8)
    } else {
        (8, 16)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct EmbedParams {
    t: u32,
    d: u32,
    scale: f32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct NormParams {
    t: u32,
    d: u32,
    eps: f32,
    residual: u32,
    offset: f32,
    scale: f32,
    _p0: u32,
    _p1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MatmulParams {
    m: u32,
    n: u32,
    k: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct QkParams {
    t: u32,
    heads: u32,
    theta: f32,
    scale: f32,
    eps: f32,
    rope_dims: u32,
    has_kv: u32,
    v_norm: u32,
    offset: f32,
    q_stride: u32,
    kv_heads: u32,
    _p1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AttnParams {
    t: u32,
    heads: u32,
    window: u32,
    q_stride: u32,
    kv_heads: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PlCombineParams {
    t: u32,
    p: u32,
    layers: u32,
    eps: f32,
    offset: f32,
    proj_scale: f32,
    emb_scale: f32,
    out_scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PlGateParams {
    t: u32,
    p: u32,
    layers: u32,
    layer: u32,
}

struct Kernel {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

/// Compiled kernel variants, keyed by (kernel, head_dim, quant). Built on
/// demand while wiring the layers.
struct Kernels {
    device: wgpu::Device,
    map: HashMap<(&'static str, usize, u32), Kernel>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum K {
    Embed,
    Norm,
    Matmul,
    MatmulGated,
    Logits,
    Qk,
    Attention,
    PlCombine,
    PlGate,
}

impl K {
    fn name(self) -> &'static str {
        match self {
            K::Embed => "embed",
            K::Norm => "rmsnorm",
            K::Matmul => "matmul",
            K::MatmulGated => "matmul_gated",
            K::Logits => "logits",
            K::Qk => "qk_prep",
            K::Attention => "attention",
            K::PlCombine => "pl_combine",
            K::PlGate => "pl_gate",
        }
    }

    fn source(self) -> &'static str {
        match self {
            K::Embed => include_str!("shaders/embed.wgsl"),
            K::Norm => include_str!("shaders/rmsnorm.wgsl"),
            K::Matmul => include_str!("shaders/matmul.wgsl"),
            K::MatmulGated => include_str!("shaders/matmul_gated.wgsl"),
            K::Logits => include_str!("shaders/logits.wgsl"),
            K::Qk => include_str!("shaders/qk_prep.wgsl"),
            K::Attention => include_str!("shaders/attention.wgsl"),
            K::PlCombine => include_str!("shaders/pl_combine.wgsl"),
            K::PlGate => include_str!("shaders/pl_gate.wgsl"),
        }
    }

    /// Storage bindings after the uniform at binding 0.
    fn bindings(self) -> Vec<wgpu::BindingType> {
        let ro = storage(true);
        let rw = storage(false);
        match self {
            K::Embed => vec![ro, ro, ro, rw],
            K::Norm => vec![ro, ro, rw],
            K::Matmul => vec![ro, ro, ro, rw],
            K::MatmulGated => vec![ro, ro, ro, ro, ro, rw],
            K::Logits => vec![ro, ro, ro, rw],
            K::Qk => vec![rw, ro, ro, ro, rw],
            K::Attention => vec![ro, ro, ro, rw],
            K::PlCombine => vec![ro, ro, ro, rw],
            K::PlGate => vec![ro, rw],
        }
    }

    fn reads_weights(self) -> bool {
        matches!(self, K::Embed | K::Matmul | K::MatmulGated | K::Logits)
    }

    fn uses_head_dim(self) -> bool {
        matches!(self, K::Qk | K::Attention)
    }
}

impl Kernels {
    /// `hd` is substituted into qk_prep / attention, `quant` is the pipeline
    /// constant of the weight-reading kernels; both are normalized to 0 for
    /// kernels that do not use them so variants are shared.
    fn get(&mut self, k: K, hd: usize, quant: u32) -> &Kernel {
        let hd = if k.uses_head_dim() { hd } else { 0 };
        let quant = if k.reads_weights() { quant } else { 0 };
        let key = (k.name(), hd, quant);
        if !self.map.contains_key(&key) {
            let mut src = String::new();
            if k.reads_weights() {
                src.push_str(include_str!("shaders/weight.wgsl"));
                src.push('\n');
            }
            // Attention tile shape per head_dim: 8 rows x 16 keys fits HD 256
            // in ~12.5 KB of workgroup memory; HD 512 needs 16 x 8 to stay
            // under 32 KB.
            let (rows, kb) = attn_tile(hd);
            src.push_str(
                &k.source()
                    .replace(
                        "const HD: u32 = 256u;",
                        &format!("const HD: u32 = {}u;", hd.max(256)),
                    )
                    .replace(
                        "const ROWS: u32 = 8u;",
                        &format!("const ROWS: u32 = {rows}u;"),
                    )
                    .replace("const KB: u32 = 16u;", &format!("const KB: u32 = {kb}u;")),
            );
            let label = format!("{}/{hd}/{quant}", k.name());
            let mut entries = vec![wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: None,
                },
                count: None,
            }];
            for (i, ty) in k.bindings().iter().enumerate() {
                entries.push(wgpu::BindGroupLayoutEntry {
                    binding: (i + 1) as u32,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: *ty,
                    count: None,
                });
            }
            let layout = self
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&label),
                    entries: &entries,
                });
            let desc = wgpu::ShaderModuleDescriptor {
                label: Some(&label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            };
            #[cfg(not(target_arch = "wasm32"))]
            let unchecked = std::env::var("GRANDE_WGPU_CHECKED").is_err();
            #[cfg(target_arch = "wasm32")]
            let unchecked = false;
            let module = if unchecked {
                // Native default: naga clamps every array index in the
                // generated MSL / SPIR-V (`min(i, len - 1)`), which costs ~20%
                // on the E2B pass (M4: 1.12 -> 0.88 s). The kernels guard
                // m / n / k themselves and the shapes come from the engine, so
                // an out-of-range access would be an engine bug; set
                // GRANDE_WGPU_CHECKED=1 to get the clamps back when chasing
                // one. wasm always runs Tint's checks.
                unsafe {
                    self.device
                        .create_shader_module_trusted(desc, wgpu::ShaderRuntimeChecks::unchecked())
                }
            } else {
                self.device.create_shader_module(desc)
            };
            let pl = self
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(&label),
                    bind_group_layouts: &[Some(&layout)],
                    immediate_size: 0,
                });
            let constants: Vec<(&str, f64)> = if k.reads_weights() {
                vec![("QUANT", quant as f64)]
            } else {
                Vec::new()
            };
            let pipeline = self
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(&label),
                    layout: Some(&pl),
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &constants,
                        ..Default::default()
                    },
                    cache: None,
                });
            self.map.insert(key, Kernel { pipeline, layout });
        }
        &self.map[&key]
    }
}

/// An uploaded weight: payload plus (for quantized types) block scales.
struct GpuTensor {
    dtype: Dtype,
    data: wgpu::Buffer,
    scales: Option<wgpu::Buffer>,
}

/// A kernel variant bound to its buffers.
struct Step {
    kernel: (&'static str, usize, u32),
    bind: wgpu::BindGroup,
}

struct LayerBinds {
    norm_in: Step,
    mm_qkv: Step,
    qk: Step,
    attn: Step,
    mm_o: Step,
    norm_post_attn: Step,
    norm_pre_ff: Step,
    mm_gate_up: Step,
    mm_down: Step,
    norm_post_ff: Step,
    /// Per-layer embedding chain (Gemma 4).
    pl: Option<PlBinds>,
    out_scale: f32,
    sliding: bool,
    has_kv: bool,
    hd: usize,
    ff: usize,
    qkv_width: usize,
}

struct PlBinds {
    mm_gate: Step,
    gate: Step,
    mm_proj: Step,
    norm: Step,
}

pub struct Engine {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pub config: Config,
    /// Token capacity of the workspace.
    pub capacity: usize,
    /// Rows that can be read back per request.
    pub max_rows: usize,
    kernels: Kernels,
    params: wgpu::Buffer,
    param_slots: usize,
    ids: wgpu::Buffer,
    meta: wgpu::Buffer,
    h: wgpu::Buffer,
    rows: wgpu::Buffer,
    logits: wgpu::Buffer,
    staging: wgpu::Buffer,
    /// Gathered per-layer token embeddings for the request (f16 pairs).
    plg: Option<wgpu::Buffer>,
    embed: Step,
    embed_dtype: Dtype,
    /// Per-layer pre-pass: projection matmul and the combine kernel.
    pl_pre: Option<(Step, Step)>,
    layers: Vec<LayerBinds>,
    final_norm: Step,
    mm_logits: Step,
    /// Host copy of the per-layer token table, gathered per request when
    /// the caller does not supply rows itself.
    pub per_layer_table: Option<QTensor>,
    _keep: Vec<wgpu::Buffer>,
    _weights: Vec<GpuTensor>,
    /// Per-dispatch GPU timings (timestamp queries), when the adapter has
    /// them and profiling is on. Native only; wasm never sets it.
    profile: Option<Profiler>,
}

struct Profiler {
    queries: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    capacity: u32,
    period_ns: f32,
}

/// One kernel launch as recorded for a request.
struct Dispatch<'a> {
    name: &'static str,
    step: &'a Step,
    offset: u32,
    wg: (u32, u32),
}

/// Collects per-dispatch parameter blocks into one uniform buffer image.
struct Params {
    bytes: Vec<u8>,
}

impl Params {
    fn push<T: Pod>(&mut self, v: T) -> u32 {
        let off = self.bytes.len();
        self.bytes.extend_from_slice(bytemuck::bytes_of(&v));
        self.bytes.resize(off + PARAM_SLOT as usize, 0);
        off as u32
    }
}

fn storage(read_only: bool) -> wgpu::BindingType {
    wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only },
        has_dynamic_offset: false,
        min_binding_size: None,
    }
}

fn div_ceil(a: usize, b: usize) -> u32 {
    a.div_ceil(b) as u32
}

/// Opens the device and receives the weights one tensor at a time.
pub struct EngineBuilder {
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: Config,
    expected: HashMap<String, Vec<usize>>,
    weights: HashMap<String, GpuTensor>,
    scalars: HashMap<String, f32>,
    per_layer_table: Option<QTensor>,
    profile: bool,
}

impl EngineBuilder {
    /// Open the default adapter with limits sized for this checkpoint.
    pub async fn new(config: Config) -> Result<Self> {
        config.validate()?;
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY | wgpu::Backends::BROWSER_WEBGPU,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .map_err(|e| anyhow!("no GPU adapter: {e}"))?;
        let info = adapter.get_info();
        eprintln!(
            "grande-wgpu: {} ({:?}, {:?})",
            info.name, info.backend, info.device_type
        );
        let mut limits = wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits());
        // The embedding table is the largest single binding (f16: vocab x d x 2).
        let need = (config.vocab * config.d * 2) as u64;
        limits.max_storage_buffer_binding_size = adapter
            .limits()
            .max_storage_buffer_binding_size
            .max(limits.max_storage_buffer_binding_size);
        limits.max_buffer_size = adapter.limits().max_buffer_size.max(limits.max_buffer_size);
        // matmul_gated binds six storage buffers (x, two weights with scales,
        // y); WebGPU guarantees eight, the downlevel default is four.
        let sb = adapter.limits().max_storage_buffers_per_shader_stage;
        if sb < 6 {
            bail!("adapter allows {sb} storage buffers per stage; the kernels need 6");
        }
        limits.max_storage_buffers_per_shader_stage = sb.min(8);
        let wg_mem = adapter.limits().max_compute_workgroup_storage_size;
        let attn_mem = attn_workgroup_bytes(config.max_head_dim());
        if wg_mem < attn_mem {
            bail!(
                "attention at head_dim {} needs {attn_mem} bytes of workgroup memory; adapter allows {wg_mem}",
                config.max_head_dim()
            );
        }
        limits.max_compute_workgroup_storage_size = wg_mem;
        if limits.max_storage_buffer_binding_size < need {
            bail!(
                "embedding table needs a {need}-byte storage binding; adapter allows {}",
                limits.max_storage_buffer_binding_size
            );
        }
        #[cfg(not(target_arch = "wasm32"))]
        let profile = std::env::var("GRANDE_WGPU_PROFILE").is_ok()
            && adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        #[cfg(target_arch = "wasm32")]
        let profile = false;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("grande"),
                required_features: if profile {
                    wgpu::Features::TIMESTAMP_QUERY
                } else {
                    wgpu::Features::empty()
                },
                required_limits: limits,
                ..Default::default()
            })
            .await
            .map_err(|e| anyhow!("request_device: {e}"))?;
        let expected = config
            .tensors()
            .into_iter()
            .map(|s| (s.name, s.shape))
            .collect();
        Ok(EngineBuilder {
            device,
            queue,
            config,
            expected,
            weights: HashMap::new(),
            scalars: HashMap::new(),
            per_layer_table: None,
            profile,
        })
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Catalogued names not pushed yet.
    pub fn missing(&self) -> Vec<String> {
        let mut v: Vec<String> = self
            .expected
            .keys()
            .filter(|n| !self.weights.contains_key(*n) && !self.scalars.contains_key(*n))
            .cloned()
            .collect();
        v.sort();
        v
    }

    /// Upload one catalogued tensor (see `Config::tensors`).
    pub fn push(&mut self, name: &str, t: &QTensor) -> Result<()> {
        let shape = self
            .expected
            .get(name)
            .ok_or_else(|| anyhow!("unexpected tensor {name}"))?;
        if &t.shape != shape {
            bail!("{name}: shape {:?}, expected {:?}", t.shape, shape);
        }
        if name.ends_with("out_scale") {
            self.scalars.insert(name.to_string(), t.get(0));
            return Ok(());
        }
        // Norm weights are read as f16 pairs.
        if t.shape.len() == 1 && t.dtype != Dtype::F16 {
            bail!("{name}: 1-D tensors must be f16");
        }
        if t.shape.len() == 2 && !t.shape[1].is_multiple_of(32) {
            bail!(
                "{name}: inner dimension {} is not a multiple of 32",
                t.shape[1]
            );
        }
        let upload = |label: &str, bytes: &[u8]| {
            let mut padded = bytes.to_vec();
            while !padded.len().is_multiple_of(4) {
                padded.push(0);
            }
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: &padded,
                    usage: wgpu::BufferUsages::STORAGE,
                })
        };
        let data = upload(name, &t.data);
        let scales = (t.dtype != Dtype::F16).then(|| upload(&format!("{name}.scales"), &t.scales));
        self.weights.insert(
            name.to_string(),
            GpuTensor {
                dtype: t.dtype,
                data,
                scales,
            },
        );
        Ok(())
    }

    /// Host copy of the per-layer token table (`[vocab, P * layers]`); rows
    /// are gathered per request. Optional: the caller may gather itself and
    /// pass rows to `Engine::evaluate_rows`.
    pub fn set_per_layer_table(&mut self, t: QTensor) -> Result<()> {
        let want = [
            self.config.vocab,
            self.config.per_layer_dim * self.config.layers,
        ];
        if t.shape != want {
            bail!("per-layer table: shape {:?}, expected {:?}", t.shape, want);
        }
        self.per_layer_table = Some(t);
        Ok(())
    }

    /// Allocate the workspace for `capacity` packed tokens and `max_rows`
    /// readback rows, and wire every dispatch.
    pub fn finish(self, capacity: usize, max_rows: usize) -> Result<Engine> {
        let missing = self.missing();
        if !missing.is_empty() {
            bail!(
                "{} tensors missing, e.g. {}",
                missing.len(),
                missing
                    .iter()
                    .take(4)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        let EngineBuilder {
            device,
            queue,
            config: cfg,
            weights,
            scalars,
            per_layer_table,
            profile,
            ..
        } = self;
        let d = cfg.d;
        let pl = cfg.per_layer_dim;
        let l_count = cfg.layers;
        let mut kernels = Kernels {
            device: device.clone(),
            map: HashMap::new(),
        };

        let f32_buf = |name: &str, n: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(name),
                size: (n.max(1) * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        // Bound where a kernel has no scales / k_norm to read.
        let dummy = f32_buf("dummy", 4);

        // Workspace.
        let ids = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ids"),
            size: (capacity * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let meta = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("meta"),
            size: (capacity * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let max_qkv = (0..l_count).map(|l| cfg.qkv_width(l)).max().unwrap_or(0);
        let max_attn = cfg.heads * cfg.max_head_dim();
        let max_act = cfg.ff.iter().copied().max().unwrap_or(0).max(pl * l_count);
        let x = f32_buf("x", capacity * d);
        let h = f32_buf("h", capacity * d);
        let qkv = f32_buf("qkv", capacity * max_qkv);
        let attn = f32_buf("attn", capacity * max_attn);
        let a = f32_buf("a", capacity * d);
        let act = f32_buf("act", capacity * max_act);
        let rows = f32_buf("rows", max_rows * d);
        let logits = f32_buf("logits", max_rows * cfg.vocab);
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (max_rows * cfg.vocab.max(d) * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // K/V per layer that owns one: [capacity][2 x kv_heads x hd].
        let kv_bufs: Vec<Option<wgpu::Buffer>> = (0..l_count)
            .map(|l| {
                cfg.has_kv(l).then(|| {
                    f32_buf(
                        &format!("kv.{l}"),
                        capacity * 2 * cfg.kv_heads * cfg.head_dim[l],
                    )
                })
            })
            .collect();
        // Per-layer inputs: gathered table rows (f16 pairs) and the combined
        // inputs (f32), both [capacity][layers x P].
        let (plg, pli) = if pl > 0 {
            let g = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("plg"),
                size: (capacity * pl * l_count * 2) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            (Some(g), Some(f32_buf("pli", capacity * pl * l_count)))
        } else {
            (None, None)
        };
        // Enough parameter slots for every dispatch of one request.
        let param_slots = 4 + l_count * 14 + 2;
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: PARAM_SLOT * param_slots as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let w = |name: &str| -> Result<&GpuTensor> {
            weights
                .get(name)
                .ok_or_else(|| anyhow!("missing tensor {name}"))
        };
        fn sc<'b>(t: &'b GpuTensor, dummy: &'b wgpu::Buffer) -> &'b wgpu::Buffer {
            t.scales.as_ref().unwrap_or(dummy)
        }

        // Bind a kernel variant to buffers.
        let mut step = |k: K, hd: usize, quant: u32, name: &str, bufs: &[&wgpu::Buffer]| -> Step {
            let kernel = kernels.get(k, hd, quant);
            let mut entries = vec![wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &params,
                    offset: 0,
                    size: wgpu::BufferSize::new(PARAM_SLOT),
                }),
            }];
            for (i, b) in bufs.iter().enumerate() {
                entries.push(wgpu::BindGroupEntry {
                    binding: (i + 1) as u32,
                    resource: b.as_entire_binding(),
                });
            }
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(name),
                layout: &kernel.layout,
                entries: &entries,
            });
            Step {
                kernel: (
                    k.name(),
                    if k.uses_head_dim() { hd } else { 0 },
                    if k.reads_weights() { quant } else { 0 },
                ),
                bind,
            }
        };

        let embed_t = w("embed")?;
        let embed = step(
            K::Embed,
            0,
            embed_t.dtype.code(),
            "embed",
            &[&ids, &embed_t.data, sc(embed_t, &dummy), &x],
        );
        let final_norm = step(
            K::Norm,
            0,
            0,
            "final_norm",
            &[&x, &w("final_norm")?.data, &h],
        );
        let mm_logits = step(
            K::Logits,
            0,
            embed_t.dtype.code(),
            "logits",
            &[&rows, &embed_t.data, sc(embed_t, &dummy), &logits],
        );
        let pl_pre = if pl > 0 {
            let proj = w("pl_model_proj")?;
            let mm = step(
                K::Matmul,
                0,
                proj.dtype.code(),
                "pl_model_proj",
                &[&x, &proj.data, sc(proj, &dummy), &act],
            );
            let combine = step(
                K::PlCombine,
                0,
                0,
                "pl_combine",
                &[
                    &act,
                    &w("pl_proj_norm")?.data,
                    plg.as_ref().unwrap(),
                    pli.as_ref().unwrap(),
                ],
            );
            Some((mm, combine))
        } else {
            None
        };

        let mut layers = Vec::with_capacity(l_count);
        for l in 0..l_count {
            let n = |s: &str| format!("blk.{l}.{s}");
            let hd = cfg.head_dim[l];
            let has_kv = cfg.has_kv(l);
            let kv = kv_bufs[cfg.kv_source[l]]
                .as_ref()
                .ok_or_else(|| anyhow!("layer {l}: K/V source has no buffer"))?;
            let t_qkv = w(&n("qkv"))?;
            let t_o = w(&n("o"))?;
            let t_gate = w(&n("gate"))?;
            let t_up = w(&n("up"))?;
            let t_down = w(&n("down"))?;
            if t_gate.dtype != t_up.dtype {
                bail!("layer {l}: gate and up projections must share a storage type");
            }
            let k_norm: &wgpu::Buffer = if has_kv {
                &w(&n("k_norm"))?.data
            } else {
                &dummy
            };
            let pl_binds = if pl > 0 {
                let t_g = w(&n("pl_gate"))?;
                let t_p = w(&n("pl_proj"))?;
                Some(PlBinds {
                    // The gate projection lands in `attn` (>= P wide per token).
                    mm_gate: step(
                        K::Matmul,
                        0,
                        t_g.dtype.code(),
                        "pl_mm_gate",
                        &[&x, &t_g.data, sc(t_g, &dummy), &attn],
                    ),
                    gate: step(K::PlGate, 0, 0, "pl_gate", &[pli.as_ref().unwrap(), &attn]),
                    mm_proj: step(
                        K::Matmul,
                        0,
                        t_p.dtype.code(),
                        "pl_mm_proj",
                        &[&attn, &t_p.data, sc(t_p, &dummy), &a],
                    ),
                    norm: step(K::Norm, 0, 0, "pl_norm", &[&a, &w(&n("pl_norm"))?.data, &x]),
                })
            } else {
                None
            };
            layers.push(LayerBinds {
                norm_in: step(
                    K::Norm,
                    0,
                    0,
                    "norm_in",
                    &[&x, &w(&n("attn_norm"))?.data, &h],
                ),
                mm_qkv: step(
                    K::Matmul,
                    0,
                    t_qkv.dtype.code(),
                    "mm_qkv",
                    &[&h, &t_qkv.data, sc(t_qkv, &dummy), &qkv],
                ),
                qk: step(
                    K::Qk,
                    hd,
                    0,
                    "qk_prep",
                    &[&qkv, &w(&n("q_norm"))?.data, k_norm, &meta, kv],
                ),
                attn: step(K::Attention, hd, 0, "attention", &[&qkv, kv, &meta, &attn]),
                mm_o: step(
                    K::Matmul,
                    0,
                    t_o.dtype.code(),
                    "mm_o",
                    &[&attn, &t_o.data, sc(t_o, &dummy), &a],
                ),
                norm_post_attn: step(
                    K::Norm,
                    0,
                    0,
                    "norm_post_attn",
                    &[&a, &w(&n("post_attn_norm"))?.data, &x],
                ),
                norm_pre_ff: step(
                    K::Norm,
                    0,
                    0,
                    "norm_pre_ff",
                    &[&x, &w(&n("ffn_norm"))?.data, &h],
                ),
                mm_gate_up: step(
                    K::MatmulGated,
                    0,
                    t_gate.dtype.code(),
                    "mm_gate_up",
                    &[
                        &h,
                        &t_gate.data,
                        sc(t_gate, &dummy),
                        &t_up.data,
                        sc(t_up, &dummy),
                        &act,
                    ],
                ),
                mm_down: step(
                    K::Matmul,
                    0,
                    t_down.dtype.code(),
                    "mm_down",
                    &[&act, &t_down.data, sc(t_down, &dummy), &a],
                ),
                norm_post_ff: step(
                    K::Norm,
                    0,
                    0,
                    "norm_post_ff",
                    &[&a, &w(&n("post_ffn_norm"))?.data, &x],
                ),
                pl: pl_binds,
                out_scale: scalars.get(&n("out_scale")).copied().unwrap_or(1.0),
                sliding: cfg.sliding[l],
                has_kv,
                hd,
                ff: cfg.ff[l],
                qkv_width: cfg.qkv_width(l),
            });
        }

        let profile = profile.then(|| {
            let capacity = 2 * param_slots as u32;
            Profiler {
                queries: device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("timestamps"),
                    ty: wgpu::QueryType::Timestamp,
                    count: capacity,
                }),
                resolve: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("ts-resolve"),
                    size: capacity as u64 * 8,
                    usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }),
                readback: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("ts-readback"),
                    size: capacity as u64 * 8,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                capacity,
                period_ns: queue.get_timestamp_period(),
            }
        });
        let embed_dtype = embed_t.dtype;
        let mut keep = vec![x, qkv, attn, a, act, dummy];
        keep.extend(kv_bufs.into_iter().flatten());
        keep.extend(pli);
        Ok(Engine {
            device,
            queue,
            config: cfg,
            capacity,
            max_rows,
            kernels,
            params,
            param_slots,
            ids,
            meta,
            h,
            rows,
            logits,
            staging,
            plg,
            embed,
            embed_dtype,
            pl_pre,
            layers,
            final_norm,
            mm_logits,
            per_layer_table,
            _keep: keep,
            _weights: weights.into_values().collect(),
            profile,
        })
    }
}

impl Engine {
    /// Open the default adapter and upload a whole host checkpoint.
    /// `capacity` is the largest packed request (prefix + all branches) in
    /// tokens.
    pub async fn new(weights: &Weights, capacity: usize, max_rows: usize) -> Result<Self> {
        weights.check()?;
        let mut b = EngineBuilder::new(weights.config.clone()).await?;
        for spec in weights.config.tensors() {
            b.push(&spec.name, weights.get(&spec.name)?)?;
        }
        if let Some(t) = &weights.per_layer_table {
            b.set_per_layer_table(t.clone())?;
        }
        b.finish(capacity, max_rows)
    }

    /// Run a two-token request once so the first real request does not pay
    /// for the workspace's first touch (wgpu zero-initializes buffers on
    /// first use; on Metal that is ~150 ms for a Gemma 4 workspace).
    pub async fn warmup(&self) -> Result<()> {
        let branches = [BranchTokens {
            tokens: vec![grande_core::Token(0)],
            want: vec![0],
        }];
        let rows = (self.config.per_layer_dim > 0)
            .then(|| vec![0u8; 2 * self.config.per_layer_dim * self.config.layers * 2]);
        self.evaluate_rows(&[self.config.bos], &branches, Want::Logits, rows.as_deref())
            .await?;
        Ok(())
    }

    /// Gather the per-layer token table rows for `ids` (f16 little-endian,
    /// as `evaluate_rows` expects).
    pub fn gather_per_layer(table: &QTensor, ids: &[u32]) -> Vec<u8> {
        let width = table.shape[1];
        let mut row = vec![f16::ZERO; width];
        let mut out = Vec::with_capacity(ids.len() * width * 2);
        for &id in ids {
            table.row_f16(id as usize, &mut row);
            for v in &row {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        out
    }

    /// Evaluate the prefix and every branch in one pass. `prefix` and branch
    /// tokens are ids in this checkpoint's vocabulary. Models with per-layer
    /// embeddings gather them from `per_layer_table`.
    pub async fn evaluate(
        &self,
        prefix: &[u32],
        branches: &[BranchTokens],
        want: Want,
    ) -> Result<Vec<BranchOutput>> {
        let rows = if self.config.per_layer_dim > 0 {
            let table = self
                .per_layer_table
                .as_ref()
                .ok_or_else(|| anyhow!("this model needs per-layer embedding rows; none loaded"))?;
            let mut ids: Vec<u32> = prefix.to_vec();
            for b in branches {
                ids.extend(b.tokens.iter().map(|t| t.0 as u32));
            }
            #[cfg(not(target_arch = "wasm32"))]
            let t0 = std::time::Instant::now();
            let rows = Self::gather_per_layer(table, &ids);
            #[cfg(not(target_arch = "wasm32"))]
            if self.profile.is_some() {
                eprintln!(
                    "host gather: {:.1} ms for {} tokens",
                    t0.elapsed().as_secs_f64() * 1e3,
                    ids.len()
                );
            }
            Some(rows)
        } else {
            None
        };
        self.evaluate_rows(prefix, branches, want, rows.as_deref())
            .await
    }

    /// `evaluate` with the per-layer embedding rows supplied by the caller:
    /// f16 little-endian, `[prefix + branch tokens][layers x P]`.
    pub async fn evaluate_rows<'s>(
        &'s self,
        prefix: &[u32],
        branches: &[BranchTokens],
        want: Want,
        per_layer_rows: Option<&[u8]>,
    ) -> Result<Vec<BranchOutput>> {
        let cfg = &self.config;
        let d = cfg.d;
        // Pack: prefix at positions 0..P in sequence 0, each branch restarting
        // at P in its own sequence.
        let p = prefix.len();
        let total = p + branches.iter().map(|b| b.tokens.len()).sum::<usize>();
        if total > self.capacity {
            bail!(
                "{total} tokens exceed the engine capacity {}",
                self.capacity
            );
        }
        if total == 0 {
            bail!("empty request");
        }
        let mut ids: Vec<u32> = Vec::with_capacity(total);
        let mut meta: Vec<i32> = Vec::with_capacity(total * 2);
        // (branch, slot, packed index)
        let mut wanted: Vec<(usize, usize, usize)> = Vec::new();
        ids.extend_from_slice(prefix);
        for i in 0..p {
            meta.push(i as i32);
            meta.push(0);
        }
        for (bi, b) in branches.iter().enumerate() {
            let start = ids.len();
            for (j, t) in b.tokens.iter().enumerate() {
                ids.push(t.0 as u32);
                meta.push((p + j) as i32);
                meta.push((bi + 1) as i32);
            }
            for (slot, &w) in b.want.iter().enumerate() {
                if w >= b.tokens.len() {
                    bail!(
                        "branch {bi}: wanted position {w} past its {} tokens",
                        b.tokens.len()
                    );
                }
                wanted.push((bi, slot, start + w));
            }
        }
        if wanted.len() > self.max_rows {
            bail!(
                "{} rows requested, engine reads back at most {}",
                wanted.len(),
                self.max_rows
            );
        }
        if let Some(&id) = ids.iter().find(|&&id| id as usize >= cfg.vocab) {
            bail!("token id {id} outside the vocabulary of {}", cfg.vocab);
        }
        let t = total;
        self.queue
            .write_buffer(&self.ids, 0, bytemuck::cast_slice(&ids));
        self.queue
            .write_buffer(&self.meta, 0, bytemuck::cast_slice(&meta));
        if let Some(plg) = &self.plg {
            let rows =
                per_layer_rows.ok_or_else(|| anyhow!("per-layer embedding rows required"))?;
            let need = t * cfg.per_layer_dim * cfg.layers * 2;
            if rows.len() != need {
                bail!("per-layer rows: {} bytes, expected {need}", rows.len());
            }
            self.queue.write_buffer(plg, 0, rows);
        }

        let mut params = Params {
            bytes: Vec::with_capacity(self.param_slots * PARAM_SLOT as usize),
        };
        let mut plan: Vec<Dispatch<'s>> = Vec::with_capacity(self.param_slots);
        {
            let mut run = |name: &'static str, step: &'s Step, offset: u32, wg: (u32, u32)| {
                plan.push(Dispatch {
                    name,
                    step,
                    offset,
                    wg,
                });
            };
            let off = params.push(EmbedParams {
                t: t as u32,
                d: d as u32,
                scale: (d as f32).sqrt(),
                _pad: 0,
            });
            run("embed", &self.embed, off, (div_ceil(t * d / 4, 256), 1));

            let norm_p = |residual: u32, scale: f32| NormParams {
                t: t as u32,
                d: d as u32,
                eps: cfg.eps,
                residual,
                offset: cfg.norm_offset,
                scale,
                _p0: 0,
                _p1: 0,
            };
            let mm = |n: usize, k: usize| MatmulParams {
                m: t as u32,
                n: n as u32,
                k: k as u32,
                _pad: 0,
            };
            // matmul.wgsl tiles 64 rows x 128 cols, matmul_gated.wgsl 64 x 64.
            let mm_wg = |n: usize| (div_ceil(n, 128), div_ceil(t, 64));
            let gated_wg = |n: usize| (div_ceil(n, 64), div_ceil(t, 64));
            let pl = cfg.per_layer_dim;
            if let Some((mm_proj, combine)) = &self.pl_pre {
                let n = pl * cfg.layers;
                let off = params.push(mm(n, d));
                run("pl_model_proj", mm_proj, off, mm_wg(n));
                let off = params.push(PlCombineParams {
                    t: t as u32,
                    p: pl as u32,
                    layers: cfg.layers as u32,
                    eps: cfg.eps,
                    offset: cfg.norm_offset,
                    proj_scale: 1.0 / (d as f32).sqrt(),
                    emb_scale: (pl as f32).sqrt(),
                    out_scale: 1.0 / 2f32.sqrt(),
                });
                run("pl_combine", combine, off, ((t * cfg.layers) as u32, 1));
            }
            #[cfg(not(target_arch = "wasm32"))]
            let n_layers = std::env::var("GRANDE_WGPU_LAYERS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(self.layers.len());
            #[cfg(target_arch = "wasm32")]
            let n_layers = self.layers.len();
            for (li, l) in self.layers.iter().enumerate().take(n_layers) {
                let off = params.push(norm_p(0, 1.0));
                run("norm", &l.norm_in, off, (t as u32, 1));
                let off = params.push(mm(l.qkv_width, d));
                run("mm_qkv", &l.mm_qkv, off, mm_wg(l.qkv_width));
                let off = params.push(QkParams {
                    t: t as u32,
                    heads: cfg.heads as u32,
                    theta: if l.sliding {
                        cfg.theta_local
                    } else {
                        cfg.theta_global
                    },
                    scale: cfg.query_scale,
                    eps: cfg.eps,
                    rope_dims: cfg.rope_dims[li] as u32,
                    has_kv: l.has_kv as u32,
                    v_norm: cfg.v_norm as u32,
                    offset: cfg.norm_offset,
                    q_stride: l.qkv_width as u32,
                    kv_heads: cfg.kv_heads as u32,
                    _p1: 0,
                });
                run("qk_prep", &l.qk, off, (t as u32, 1));
                let off = params.push(AttnParams {
                    t: t as u32,
                    heads: cfg.heads as u32,
                    window: if l.sliding { cfg.window as u32 } else { 0 },
                    q_stride: l.qkv_width as u32,
                    kv_heads: cfg.kv_heads as u32,
                    _p0: 0,
                    _p1: 0,
                    _p2: 0,
                });
                // One workgroup per ROWS query rows of one KV head (wg.y).
                let (rows, _) = attn_tile(l.hd);
                let hpg = cfg.heads / cfg.kv_heads;
                run(
                    "attention",
                    &l.attn,
                    off,
                    (div_ceil(t * hpg, rows), cfg.kv_heads as u32),
                );
                let attn_w = cfg.heads * l.hd;
                let off = params.push(mm(d, attn_w));
                run("mm_o", &l.mm_o, off, mm_wg(d));
                let off = params.push(norm_p(1, 1.0));
                run("norm", &l.norm_post_attn, off, (t as u32, 1));
                let off = params.push(norm_p(0, 1.0));
                run("norm", &l.norm_pre_ff, off, (t as u32, 1));
                let off = params.push(mm(l.ff, d));
                run("mm_gate_up", &l.mm_gate_up, off, gated_wg(l.ff));
                let off = params.push(mm(d, l.ff));
                run("mm_down", &l.mm_down, off, mm_wg(d));
                match &l.pl {
                    None => {
                        let off = params.push(norm_p(1, l.out_scale));
                        run("norm", &l.norm_post_ff, off, (t as u32, 1));
                    }
                    Some(pb) => {
                        let off = params.push(norm_p(1, 1.0));
                        run("norm", &l.norm_post_ff, off, (t as u32, 1));
                        let off = params.push(mm(pl, d));
                        run("pl_mm_gate", &pb.mm_gate, off, mm_wg(pl));
                        let off = params.push(PlGateParams {
                            t: t as u32,
                            p: pl as u32,
                            layers: cfg.layers as u32,
                            layer: li as u32,
                        });
                        run("pl_gate", &pb.gate, off, (div_ceil(t * pl, 256), 1));
                        let off = params.push(mm(d, pl));
                        run("pl_mm_proj", &pb.mm_proj, off, mm_wg(d));
                        let off = params.push(norm_p(1, l.out_scale));
                        run("norm", &pb.norm, off, (t as u32, 1));
                    }
                }
            }
            let off = params.push(norm_p(0, 1.0));
            run("norm", &self.final_norm, off, (t as u32, 1));
        }

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("grande"),
            });
        let pipeline_of =
            |step: &Step| -> &wgpu::ComputePipeline { &self.kernels.map[&step.kernel].pipeline };
        match &self.profile {
            None => {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("forward"),
                    timestamp_writes: None,
                });
                for dsp in &plan {
                    pass.set_pipeline(pipeline_of(dsp.step));
                    pass.set_bind_group(0, &dsp.step.bind, &[dsp.offset]);
                    pass.dispatch_workgroups(dsp.wg.0, dsp.wg.1, 1);
                }
            }
            Some(prof) => {
                if (plan.len() as u32 + 1) * 2 > prof.capacity {
                    bail!("too many dispatches to profile");
                }
                for (i, dsp) in plan.iter().enumerate() {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some(dsp.name),
                        timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                            query_set: &prof.queries,
                            beginning_of_pass_write_index: Some(2 * i as u32),
                            end_of_pass_write_index: Some(2 * i as u32 + 1),
                        }),
                    });
                    pass.set_pipeline(pipeline_of(dsp.step));
                    pass.set_bind_group(0, &dsp.step.bind, &[dsp.offset]);
                    pass.dispatch_workgroups(dsp.wg.0, dsp.wg.1, 1);
                }
            }
        }
        // Gather the wanted rows of the final hidden state.
        let row_bytes = (d * 4) as u64;
        for (r, &(_, _, idx)) in wanted.iter().enumerate() {
            enc.copy_buffer_to_buffer(
                &self.h,
                idx as u64 * row_bytes,
                &self.rows,
                r as u64 * row_bytes,
                row_bytes,
            );
        }
        let width = match want {
            Want::Hidden => d,
            Want::Logits => cfg.vocab,
        };
        let mut n_timed = plan.len() as u32;
        if want == Want::Logits {
            // Profiled as one more timestamped pass after the plan.
            let timestamp_writes =
                self.profile
                    .as_ref()
                    .map(|prof| wgpu::ComputePassTimestampWrites {
                        query_set: &prof.queries,
                        beginning_of_pass_write_index: Some(2 * n_timed),
                        end_of_pass_write_index: Some(2 * n_timed + 1),
                    });
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("logits"),
                timestamp_writes,
            });
            let off = params.push(MatmulParams {
                m: wanted.len() as u32,
                n: cfg.vocab as u32,
                k: d as u32,
                _pad: 0,
            });
            pass.set_pipeline(pipeline_of(&self.mm_logits));
            pass.set_bind_group(0, &self.mm_logits.bind, &[off]);
            pass.dispatch_workgroups(div_ceil(cfg.vocab, 256), 1, 1);
            drop(pass);
            n_timed += 1;
        }
        if let Some(prof) = &self.profile {
            let n = 2 * n_timed;
            enc.resolve_query_set(&prof.queries, 0..n, &prof.resolve, 0);
            enc.copy_buffer_to_buffer(&prof.resolve, 0, &prof.readback, 0, n as u64 * 8);
        }
        let out_bytes = (wanted.len() * width * 4) as u64;
        let src = if want == Want::Logits {
            &self.logits
        } else {
            &self.rows
        };
        enc.copy_buffer_to_buffer(src, 0, &self.staging, 0, out_bytes);

        if params.bytes.len() > self.param_slots * PARAM_SLOT as usize {
            bail!("parameter slots exhausted");
        }
        self.queue.write_buffer(&self.params, 0, &params.bytes);
        #[cfg(not(target_arch = "wasm32"))]
        let t_enc = std::time::Instant::now();
        let cb = enc.finish();
        #[cfg(not(target_arch = "wasm32"))]
        let t_fin = t_enc.elapsed();
        self.queue.submit(Some(cb));
        #[cfg(not(target_arch = "wasm32"))]
        if self.profile.is_some() {
            eprintln!(
                "encode finish {:.1} ms, submit {:.1} ms",
                t_fin.as_secs_f64() * 1e3,
                t_enc.elapsed().as_secs_f64() * 1e3
            );
        }

        #[cfg(not(target_arch = "wasm32"))]
        let t_submit = std::time::Instant::now();
        let slice = self.staging.slice(..out_bytes);
        let (tx, rx) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).ok();
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| anyhow!("device poll: {e:?}"))?;
        #[cfg(not(target_arch = "wasm32"))]
        if self.profile.is_some() {
            eprintln!(
                "poll done {:.1} ms after submit",
                t_submit.elapsed().as_secs_f64() * 1e3
            );
        }
        rx.await
            .context("map callback dropped")?
            .map_err(|e| anyhow!("map_async: {e:?}"))?;
        let mut data: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
        self.staging.unmap();
        if let Some(prof) = &self.profile {
            #[cfg(not(target_arch = "wasm32"))]
            eprintln!(
                "submit to mapped: {:.1} ms",
                t_submit.elapsed().as_secs_f64() * 1e3
            );
            self.report_profile(prof, &plan, n_timed as usize).await?;
        }
        if want == Want::Logits && cfg.softcap > 0.0 {
            let cap = cfg.softcap;
            for v in &mut data {
                *v = cap * (*v / cap).tanh();
            }
        }

        let mut outputs: Vec<BranchOutput> = branches
            .iter()
            .map(|b| BranchOutput {
                rows: Vec::with_capacity(b.want.len()),
            })
            .collect();
        for (r, &(bi, slot, _)) in wanted.iter().enumerate() {
            if outputs[bi].rows.len() != slot {
                bail!("row order mismatch for branch {bi}");
            }
            outputs[bi]
                .rows
                .push(data[r * width..(r + 1) * width].to_vec());
        }
        Ok(outputs)
    }

    /// Storage type of the embedding table (also the logits projection).
    pub fn embed_dtype(&self) -> Dtype {
        self.embed_dtype
    }

    /// Print GPU time per kernel for the dispatches just run (stderr).
    async fn report_profile(
        &self,
        prof: &Profiler,
        plan: &[Dispatch<'_>],
        timed: usize,
    ) -> Result<()> {
        let n = timed as u64 * 2;
        let slice = prof.readback.slice(..n * 8);
        let (tx, rx) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).ok();
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| anyhow!("device poll: {e:?}"))?;
        rx.await
            .context("map callback dropped")?
            .map_err(|e| anyhow!("{e:?}"))?;
        let ts: Vec<u64> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
        prof.readback.unmap();
        let mut by_name: Vec<(&str, f64, usize)> = Vec::new();
        let mut total = 0.0;
        let names = plan
            .iter()
            .map(|d| d.name)
            .chain(std::iter::repeat("logits"));
        for (i, name) in names.take(timed).enumerate() {
            let ms = (ts[2 * i + 1].saturating_sub(ts[2 * i])) as f64 * prof.period_ns as f64 / 1e6;
            total += ms;
            match by_name.iter_mut().find(|(n, _, _)| *n == name) {
                Some(e) => {
                    e.1 += ms;
                    e.2 += 1;
                }
                None => by_name.push((name, ms, 1)),
            }
        }
        by_name.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        eprintln!("gpu profile: {total:.2} ms over {timed} dispatches");
        for (name, ms, count) in by_name {
            eprintln!(
                "  {name:<14} {ms:8.2} ms  ({count} x {:.3} ms)",
                ms / count as f64
            );
        }
        Ok(())
    }
}
