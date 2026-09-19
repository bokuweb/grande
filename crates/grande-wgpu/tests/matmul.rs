//! The matmul kernels against a CPU reference on odd shapes (partial row and
//! column tiles) for every weight storage type.

use std::num::NonZeroU64;

use grande_wgpu::{Dtype, QTensor};
use half::f16;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

const WEIGHT: &str = include_str!("../src/shaders/weight.wgsl");
const MATMUL: &str = include_str!("../src/shaders/matmul.wgsl");
const GATED: &str = include_str!("../src/shaders/matmul_gated.wgsl");

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044715 * x * x * x)).tanh())
}

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

fn gpu() -> Option<Gpu> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).ok()?;
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()?;
    Some(Gpu { device, queue })
}

impl Gpu {
    fn storage(&self, bytes: &[u8], rw: bool) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | if rw {
                        wgpu::BufferUsages::COPY_SRC
                    } else {
                        wgpu::BufferUsages::empty()
                    },
            })
    }

    /// Run `src` (weight.wgsl prepended) with the given storage buffers
    /// after a uniform [m, n, k, 0] at binding 0; returns buffer `out`.
    fn run(
        &self,
        src: &str,
        quant: u32,
        dims: [u32; 3],
        buffers: &[&wgpu::Buffer],
        out: usize,
        wg: (u32, u32),
    ) -> Vec<f32> {
        use wgpu::util::DeviceExt;
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(format!("{WEIGHT}\n{src}").into()),
            });
        let mut entries = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }];
        for (i, _) in buffers.iter().enumerate() {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: (i + 1) as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: i != out,
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }
        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &entries,
            });
        let pl = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
        let constants = [("QUANT", quant as f64)];
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&pl),
                module: &module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &constants,
                    ..Default::default()
                },
                cache: None,
            });
        let params = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&[dims[0], dims[1], dims[2], 0]),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let mut bg = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &params,
                offset: 0,
                size: NonZeroU64::new(16),
            }),
        }];
        for (i, b) in buffers.iter().enumerate() {
            bg.push(wgpu::BindGroupEntry {
                binding: (i + 1) as u32,
                resource: b.as_entire_binding(),
            });
        }
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &bg,
        });
        let size = buffers[out].size();
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(wg.0, wg.1, 1);
        }
        enc.copy_buffer_to_buffer(buffers[out], 0, &staging, 0, size);
        self.queue.submit(Some(enc.finish()));
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        let v: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
        staging.unmap();
        v
    }
}

/// Weight [n, k] in `dtype`, as (payload bytes, scale bytes, dequantized
/// row-major f32).
fn weight(r: &mut Rng, n: usize, k: usize, dtype: Dtype) -> (Vec<u8>, Vec<u8>, Vec<f32>) {
    let data: Vec<f32> = (0..n * k).map(|_| r.next()).collect();
    let q = match dtype {
        Dtype::F16 => QTensor::from_f32(vec![n, k], &data),
        Dtype::Q8 => QTensor::quantize_q8(vec![n, k], &data).unwrap(),
        Dtype::Q4 => QTensor::quantize_q4(vec![n, k], &data).unwrap(),
    };
    // The kernels stage weights as f16, so round the reference the same way.
    let deq: Vec<f32> = q
        .to_f32()
        .iter()
        .map(|&v| f16::from_f32(v).to_f32())
        .collect();
    let scales = if q.scales.is_empty() {
        vec![0u8; 4]
    } else {
        q.scales.clone()
    };
    (q.data.clone(), scales, deq)
}

fn check_matmul(g: &Gpu, m: usize, n: usize, k: usize, dtype: Dtype, tol: f32) {
    let mut r = Rng(0x9e3779b97f4a7c15 ^ (m * n * k) as u64);
    let x: Vec<f32> = (0..m * k).map(|_| r.next() * 4.0).collect();
    let (wb, sb, wd) = weight(&mut r, n, k, dtype);
    let bx = g.storage(bytemuck::cast_slice(&x), false);
    let bw = g.storage(&wb, false);
    let bs = g.storage(&sb, false);
    let by = g.storage(&vec![0u8; m * n * 4], true);
    let y = g.run(
        MATMUL,
        dtype.code(),
        [m as u32, n as u32, k as u32],
        &[&bx, &bw, &bs, &by],
        3,
        (n.div_ceil(128) as u32, m.div_ceil(64) as u32),
    );
    let mut worst = 0.0f32;
    for i in 0..m {
        for j in 0..n {
            let exp: f32 = (0..k)
                .map(|t| f16::from_f32(x[i * k + t]).to_f32() * wd[j * k + t])
                .sum();
            let d = (y[i * n + j] - exp).abs() / exp.abs().max(1.0);
            if d > worst {
                worst = d;
            }
        }
    }
    assert!(
        worst < tol,
        "matmul {dtype:?} {m}x{n}x{k}: max rel diff {worst}"
    );
}

fn check_gated(g: &Gpu, m: usize, n: usize, k: usize, dtype: Dtype, tol: f32) {
    let mut r = Rng(0xdeadbeef ^ (m * n * k) as u64);
    let x: Vec<f32> = (0..m * k).map(|_| r.next()).collect();
    let (gb, gsb, gd) = weight(&mut r, n, k, dtype);
    let (ub, usb, ud) = weight(&mut r, n, k, dtype);
    let bx = g.storage(bytemuck::cast_slice(&x), false);
    let bg = g.storage(&gb, false);
    let bgs = g.storage(&gsb, false);
    let bu = g.storage(&ub, false);
    let bus = g.storage(&usb, false);
    let by = g.storage(&vec![0u8; m * n * 4], true);
    let y = g.run(
        GATED,
        dtype.code(),
        [m as u32, n as u32, k as u32],
        &[&bx, &bg, &bgs, &bu, &bus, &by],
        5,
        (n.div_ceil(64) as u32, m.div_ceil(64) as u32),
    );
    let mut worst = 0.0f32;
    for i in 0..m {
        for j in 0..n {
            let xr = |t: usize| f16::from_f32(x[i * k + t]).to_f32();
            let gate: f32 = (0..k).map(|t| xr(t) * gd[j * k + t]).sum();
            let up: f32 = (0..k).map(|t| xr(t) * ud[j * k + t]).sum();
            let exp = gelu(gate) * up;
            let d = (y[i * n + j] - exp).abs() / exp.abs().max(1.0);
            if d > worst {
                worst = d;
            }
        }
    }
    assert!(
        worst < tol,
        "gated {dtype:?} {m}x{n}x{k}: max rel diff {worst}"
    );
}

#[test]
fn matmul_matches_cpu() {
    let Some(g) = gpu() else { return };
    for dtype in [Dtype::F16, Dtype::Q8, Dtype::Q4] {
        check_matmul(&g, 23, 96, 64, dtype, 2e-3);
        check_matmul(&g, 130, 200, 96, dtype, 2e-3);
        check_matmul(&g, 64, 128, 32, dtype, 2e-3);
    }
}

#[test]
fn gated_matches_cpu() {
    let Some(g) = gpu() else { return };
    for dtype in [Dtype::F16, Dtype::Q8, Dtype::Q4] {
        check_gated(&g, 23, 96, 64, dtype, 2e-3);
        check_gated(&g, 70, 100, 160, dtype, 2e-3);
    }
}
