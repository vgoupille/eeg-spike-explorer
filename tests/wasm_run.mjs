// Run the WebAssembly detector from Node (used by tests/test_wasm.py).
// usage: node wasm_run.mjs core.wasm signal.f64 n_ch n_t k smooth_n refractory_n
import { readFileSync } from "node:fs";
const [wasmPath, sigPath, nCh, nT, k, sm, rf] = process.argv.slice(2);
const { instance } = await WebAssembly.instantiate(readFileSync(wasmPath));
const w = instance.exports, n = +nCh * +nT;
const src = new Float64Array(readFileSync(sigPath).buffer.slice(0));
const ptr = w.alloc_f64(n);
new Float64Array(w.memory.buffer, ptr, n).set(src);
const t0 = performance.now();
const m = w.detect(ptr, +nCh, +nT, +k, +sm, +rf);
const ms = performance.now() - t0;
const ch = Array.from(new Uint32Array(w.memory.buffer, w.out_channels(), m));
const s = Array.from(new Uint32Array(w.memory.buffer, w.out_samples(), m));
const sc = Array.from(new Float64Array(w.memory.buffer, w.out_scores(), m));
console.log(JSON.stringify({ ch, s, sc, ms }));
