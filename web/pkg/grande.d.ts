/* tslint:disable */
/* eslint-disable */

/**
 * Assemble the TypeSafe-shaped response. `rows` is a JSON array with one
 * entry per branch (request order): the option logits the backend read at
 * the branch's answer position, plus an optional candidate-mass diagnostic.
 */
export function answer(request: string, rows: string, temperature: number, model: string, input_tokens: number): string;

/**
 * Option labels for the label readout, in order (`A`..`Z`, `a`..`z`).
 */
export function labels(): string;

/**
 * Render a request. `layout` is a JSON `Renderer`, e.g.
 * `{"layout":"label","turn_start":"<|turn>","turn_end":"<turn|>","user":"user","model":"model"}`
 * or `{"layout":"pointer", ...delimiters}`. Returns the `Rendered` JSON:
 * prefix segments and, per branch, segments / marks / option keys.
 */
export function render(request: string, layout: string): string;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly answer: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => [number, number, number, number];
    readonly labels: () => [number, number];
    readonly render: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
