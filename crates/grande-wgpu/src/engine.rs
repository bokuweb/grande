//! The forward pass on wgpu. Weights and a token-capacity-sized workspace live
//! on the device for the life of the engine; a request writes its token ids
//! and (position, sequence) metadata, records one command buffer for the
//! whole network, and reads back the requested rows.

use anyhow::{anyhow, bail, Context, Result};
use bytemuck::{Pod, Zeroable};
use grande_core::{BranchOutput, BranchTokens, Want};
use wgpu::util::DeviceExt;

use crate::model::{Config, Tensor16, Weights};

const PARAM_SLOT: u64 = 256;

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
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AttnParams {
    t: u32,
    heads: u32,
    window: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GegluParams {
    t: u32,
    f: u32,
    _p0: u32,
    _p1: u32,
}

struct Kernel {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

struct LayerBinds {
    norm_in: wgpu::BindGroup,
    mm_qkv: wgpu::BindGroup,
    qk: wgpu::BindGroup,
    attn: wgpu::BindGroup,
    mm_o: wgpu::BindGroup,
    norm_post_attn: wgpu::BindGroup,
    norm_pre_ff: wgpu::BindGroup,
    mm_gate_up: wgpu::BindGroup,
    geglu: wgpu::BindGroup,
    mm_down: wgpu::BindGroup,
    norm_post_ff: wgpu::BindGroup,
    sliding: bool,
    // Weight buffers stay alive with the bind groups that reference them.
    _weights: Vec<wgpu::Buffer>,
}

pub struct Engine {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pub config: Config,
    /// Token capacity of the workspace.
    pub capacity: usize,
    /// Rows that can be read back per request.
    pub max_rows: usize,
    embed: Kernel,
    norm: Kernel,
    matmul: Kernel,
    qk: Kernel,
    attention: Kernel,
    geglu: Kernel,
    params: wgpu::Buffer,
    param_slots: usize,
    ids: wgpu::Buffer,
    meta: wgpu::Buffer,
    h: wgpu::Buffer,
    rows: wgpu::Buffer,
    logits: wgpu::Buffer,
    staging: wgpu::Buffer,
    embed_bg: wgpu::BindGroup,
    layers: Vec<LayerBinds>,
    final_norm_bg: wgpu::BindGroup,
    mm_logits_bg: wgpu::BindGroup,
    _keep: Vec<wgpu::Buffer>,
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

/// One kernel launch, recorded before encoding so the same list can go into
/// a single compute pass or, when profiling, one timestamped pass each.
struct Dispatch<'a> {
    name: &'static str,
    kernel: &'a Kernel,
    bind: &'a wgpu::BindGroup,
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

impl Engine {
    /// Open the default adapter and upload the weights. `capacity` is the
    /// largest packed request (prefix + all branches) in tokens.
    pub async fn new(weights: &Weights, capacity: usize, max_rows: usize) -> Result<Self> {
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
        let need = (weights.embed.data.len() * 2) as u64;
        limits.max_storage_buffer_binding_size = adapter
            .limits()
            .max_storage_buffer_binding_size
            .max(limits.max_storage_buffer_binding_size);
        limits.max_buffer_size = adapter.limits().max_buffer_size.max(limits.max_buffer_size);
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
        Self::with_device(device, queue, weights, capacity, max_rows, profile)
    }

    fn with_device(
        device: wgpu::Device,
        queue: wgpu::Queue,
        weights: &Weights,
        capacity: usize,
        max_rows: usize,
        profile: bool,
    ) -> Result<Self> {
        let cfg = weights.config.clone();
        let d = cfg.d;
        let qkv_w = (cfg.heads + 2) * cfg.head_dim;

        let kernel = |name: &str, src: &str, bindings: &[wgpu::BindingType]| -> Kernel {
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
            for (i, ty) in bindings.iter().enumerate() {
                entries.push(wgpu::BindGroupLayoutEntry {
                    binding: (i + 1) as u32,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: *ty,
                    count: None,
                });
            }
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(name),
                entries: &entries,
            });
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(name),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(name),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&pl),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
            Kernel { pipeline, layout }
        };
        let ro = storage(true);
        let rw = storage(false);
        let embed = kernel("embed", include_str!("shaders/embed.wgsl"), &[ro, ro, rw]);
        let norm = kernel(
            "rmsnorm",
            include_str!("shaders/rmsnorm.wgsl"),
            &[ro, ro, rw],
        );
        let matmul = kernel("matmul", include_str!("shaders/matmul.wgsl"), &[ro, ro, rw]);
        let qk = kernel(
            "qk_prep",
            include_str!("shaders/qk_prep.wgsl"),
            &[rw, ro, ro, ro],
        );
        let attention = kernel(
            "attention",
            include_str!("shaders/attention.wgsl"),
            &[ro, ro, rw],
        );
        let geglu = kernel("geglu", include_str!("shaders/geglu.wgsl"), &[ro, rw]);

        let f32_buf = |name: &str, n: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(name),
                size: (n * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let weight_buf = |name: &str, t: &Tensor16| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(name),
                contents: t.bytes(),
                usage: wgpu::BufferUsages::STORAGE,
            })
        };

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
        let x = f32_buf("x", capacity * d);
        let h = f32_buf("h", capacity * d);
        let qkv = f32_buf("qkv", capacity * qkv_w);
        let attn = f32_buf("attn", capacity * cfg.heads * cfg.head_dim);
        let a = f32_buf("a", capacity * d);
        let gu = f32_buf("gu", capacity * 2 * cfg.ff);
        let act = f32_buf("act", capacity * cfg.ff);
        let rows = f32_buf("rows", max_rows * d);
        let logits = f32_buf("logits", max_rows * cfg.vocab);
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (max_rows * cfg.vocab.max(d) * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Enough parameter slots for every dispatch of one request.
        let param_slots = 2 + cfg.layers * 11 + 2;
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: PARAM_SLOT * param_slots as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind = |k: &Kernel, name: &str, bufs: &[&wgpu::Buffer]| {
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
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(name),
                layout: &k.layout,
                entries: &entries,
            })
        };

        let embed_w = weight_buf("embed", &weights.embed);
        let embed_bg = bind(&embed, "embed", &[&ids, &embed_w, &x]);
        let final_w = weight_buf("final_norm", &weights.final_norm);
        let final_norm_bg = bind(&norm, "final_norm", &[&x, &final_w, &h]);
        let mm_logits_bg = bind(&matmul, "logits", &[&rows, &embed_w, &logits]);

        let mut layers = Vec::with_capacity(cfg.layers);
        for (i, l) in weights.layers.iter().enumerate() {
            let w_in = weight_buf("input_norm", &l.input_norm);
            let w_qkv = weight_buf("qkv", &l.qkv);
            let w_qn = weight_buf("q_norm", &l.q_norm);
            let w_kn = weight_buf("k_norm", &l.k_norm);
            let w_o = weight_buf("o", &l.o);
            let w_pa = weight_buf("post_attn_norm", &l.post_attn_norm);
            let w_pf = weight_buf("pre_ff_norm", &l.pre_ff_norm);
            let w_gu = weight_buf("gate_up", &l.gate_up);
            let w_down = weight_buf("down", &l.down);
            let w_pff = weight_buf("post_ff_norm", &l.post_ff_norm);
            layers.push(LayerBinds {
                norm_in: bind(&norm, "norm_in", &[&x, &w_in, &h]),
                mm_qkv: bind(&matmul, "mm_qkv", &[&h, &w_qkv, &qkv]),
                qk: bind(&qk, "qk", &[&qkv, &w_qn, &w_kn, &meta]),
                attn: bind(&attention, "attn", &[&qkv, &meta, &attn]),
                mm_o: bind(&matmul, "mm_o", &[&attn, &w_o, &a]),
                norm_post_attn: bind(&norm, "norm_post_attn", &[&a, &w_pa, &x]),
                norm_pre_ff: bind(&norm, "norm_pre_ff", &[&x, &w_pf, &h]),
                mm_gate_up: bind(&matmul, "mm_gate_up", &[&h, &w_gu, &gu]),
                geglu: bind(&geglu, "geglu", &[&gu, &act]),
                mm_down: bind(&matmul, "mm_down", &[&act, &w_down, &a]),
                norm_post_ff: bind(&norm, "norm_post_ff", &[&a, &w_pff, &x]),
                sliding: cfg.sliding[i],
                _weights: vec![
                    w_in, w_qkv, w_qn, w_kn, w_o, w_pa, w_pf, w_gu, w_down, w_pff,
                ],
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
        Ok(Engine {
            device,
            queue,
            config: cfg,
            capacity,
            max_rows,
            embed,
            norm,
            matmul,
            qk,
            attention,
            geglu,
            params,
            param_slots,
            ids,
            meta,
            h,
            rows,
            logits,
            staging,
            embed_bg,
            layers,
            final_norm_bg,
            mm_logits_bg,
            _keep: vec![x, qkv, attn, a, gu, act, embed_w, final_w],
            profile,
        })
    }

    /// Evaluate the prefix and every branch in one pass. `prefix` and branch
    /// tokens are ids in this checkpoint's vocabulary.
    pub async fn evaluate<'a>(
        &'a self,
        prefix: &[u32],
        branches: &[BranchTokens],
        want: Want,
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
        let t = total;
        self.queue
            .write_buffer(&self.ids, 0, bytemuck::cast_slice(&ids));
        self.queue
            .write_buffer(&self.meta, 0, bytemuck::cast_slice(&meta));

        let mut params = Params {
            bytes: Vec::with_capacity(self.param_slots * PARAM_SLOT as usize),
        };
        let mut plan: Vec<Dispatch> = Vec::with_capacity(self.param_slots);
        {
            let mut run = |name: &'static str,
                           k: &'a Kernel,
                           bind: &'a wgpu::BindGroup,
                           offset: u32,
                           wg: (u32, u32)| {
                plan.push(Dispatch {
                    name,
                    kernel: k,
                    bind,
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
            run(
                "embed",
                &self.embed,
                &self.embed_bg,
                off,
                (div_ceil(t * d / 2, 256), 1),
            );

            let norm_p = |residual: u32| NormParams {
                t: t as u32,
                d: d as u32,
                eps: cfg.eps,
                residual,
            };
            let mm = |n: usize, k: usize| MatmulParams {
                m: t as u32,
                n: n as u32,
                k: k as u32,
                _pad: 0,
            };
            let mm_wg = |n: usize| (div_ceil(n, 64), div_ceil(t, 64));
            let qkv_w = (cfg.heads + 2) * cfg.head_dim;
            let attn_w = cfg.heads * cfg.head_dim;
            #[cfg(not(target_arch = "wasm32"))]
            let n_layers = std::env::var("GRANDE_WGPU_LAYERS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(self.layers.len());
            #[cfg(target_arch = "wasm32")]
            let n_layers = self.layers.len();
            for l in &self.layers[..n_layers] {
                let off = params.push(norm_p(0));
                run("norm", &self.norm, &l.norm_in, off, (t as u32, 1));
                let off = params.push(mm(qkv_w, d));
                run("mm_qkv", &self.matmul, &l.mm_qkv, off, mm_wg(qkv_w));
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
                    _p0: 0,
                    _p1: 0,
                    _p2: 0,
                });
                run("qk_prep", &self.qk, &l.qk, off, (t as u32, 1));
                let off = params.push(AttnParams {
                    t: t as u32,
                    heads: cfg.heads as u32,
                    window: if l.sliding { cfg.window as u32 } else { 0 },
                    _pad: 0,
                });
                run(
                    "attention",
                    &self.attention,
                    &l.attn,
                    off,
                    (div_ceil(t, 16 / cfg.heads), 1),
                );
                let off = params.push(mm(d, attn_w));
                run("mm_o", &self.matmul, &l.mm_o, off, mm_wg(d));
                let off = params.push(norm_p(1));
                run("norm", &self.norm, &l.norm_post_attn, off, (t as u32, 1));
                let off = params.push(norm_p(0));
                run("norm", &self.norm, &l.norm_pre_ff, off, (t as u32, 1));
                let off = params.push(mm(2 * cfg.ff, d));
                run(
                    "mm_gate_up",
                    &self.matmul,
                    &l.mm_gate_up,
                    off,
                    mm_wg(2 * cfg.ff),
                );
                let off = params.push(GegluParams {
                    t: t as u32,
                    f: cfg.ff as u32,
                    _p0: 0,
                    _p1: 0,
                });
                run(
                    "geglu",
                    &self.geglu,
                    &l.geglu,
                    off,
                    (div_ceil(t * cfg.ff, 256), 1),
                );
                let off = params.push(mm(d, cfg.ff));
                run("mm_down", &self.matmul, &l.mm_down, off, mm_wg(d));
                let off = params.push(norm_p(1));
                run("norm", &self.norm, &l.norm_post_ff, off, (t as u32, 1));
            }
            let off = params.push(norm_p(0));
            run("norm", &self.norm, &self.final_norm_bg, off, (t as u32, 1));
        }

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("grande"),
            });
        match &self.profile {
            None => {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("forward"),
                    timestamp_writes: None,
                });
                for dsp in &plan {
                    pass.set_pipeline(&dsp.kernel.pipeline);
                    pass.set_bind_group(0, dsp.bind, &[dsp.offset]);
                    pass.dispatch_workgroups(dsp.wg.0, dsp.wg.1, 1);
                }
            }
            Some(prof) => {
                if plan.len() as u32 * 2 > prof.capacity {
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
                    pass.set_pipeline(&dsp.kernel.pipeline);
                    pass.set_bind_group(0, dsp.bind, &[dsp.offset]);
                    pass.dispatch_workgroups(dsp.wg.0, dsp.wg.1, 1);
                }
                let n = plan.len() as u32 * 2;
                enc.resolve_query_set(&prof.queries, 0..n, &prof.resolve, 0);
                enc.copy_buffer_to_buffer(&prof.resolve, 0, &prof.readback, 0, n as u64 * 8);
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
        if want == Want::Logits {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("logits"),
                timestamp_writes: None,
            });
            let off = params.push(MatmulParams {
                m: wanted.len() as u32,
                n: cfg.vocab as u32,
                k: d as u32,
                _pad: 0,
            });
            pass.set_pipeline(&self.matmul.pipeline);
            pass.set_bind_group(0, &self.mm_logits_bg, &[off]);
            pass.dispatch_workgroups(div_ceil(cfg.vocab, 64), div_ceil(wanted.len(), 64), 1);
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
        self.queue.submit(Some(enc.finish()));

        let slice = self.staging.slice(..out_bytes);
        let (tx, rx) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).ok();
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| anyhow!("device poll: {e:?}"))?;
        rx.await
            .context("map callback dropped")?
            .map_err(|e| anyhow!("map_async: {e:?}"))?;
        let data: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
        self.staging.unmap();
        if let Some(prof) = &self.profile {
            self.report_profile(prof, &plan).await?;
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

    /// Print GPU time per kernel for the dispatches just run (stderr).
    async fn report_profile(&self, prof: &Profiler, plan: &[Dispatch<'_>]) -> Result<()> {
        let n = plan.len() as u64 * 2;
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
        for (i, dsp) in plan.iter().enumerate() {
            let ms = (ts[2 * i + 1].saturating_sub(ts[2 * i])) as f64 * prof.period_ns as f64 / 1e6;
            total += ms;
            match by_name.iter_mut().find(|(n, _, _)| *n == dsp.name) {
                Some(e) => {
                    e.1 += ms;
                    e.2 += 1;
                }
                None => by_name.push((dsp.name, ms, 1)),
            }
        }
        by_name.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        eprintln!("gpu profile: {total:.2} ms over {} dispatches", plan.len());
        for (name, ms, count) in by_name {
            eprintln!(
                "  {name:<12} {ms:8.2} ms  ({count} x {:.3} ms)",
                ms / count as f64
            );
        }
        Ok(())
    }
}
