// Model file cache in IndexedDB, plugged into transformers.js as `env.customCache`.
//
// transformers.js defaults to the Cache API, but Chromium's `Cache.put` fails
// on the large `.onnx_data` shards ("Unexpected internal error"), so only the
// small files were cached and the weights were re-downloaded on every visit.
// IndexedDB stores Blobs on disk by reference and takes multi-GB values.
//
// Keys are the request URLs transformers.js uses; values are { blob, headers }.

const DB = "grande-models";
const STORE = "files";

let dbPromise = null;
function db() {
  return (dbPromise ??= new Promise((resolve, reject) => {
    const req = indexedDB.open(DB, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  }));
}

function tx(mode, fn) {
  return db().then((d) => new Promise((resolve, reject) => {
    const t = d.transaction(STORE, mode);
    const req = fn(t.objectStore(STORE));
    t.oncomplete = () => resolve(req?.result);
    t.onerror = () => reject(t.error);
    t.onabort = () => reject(t.error);
  }));
}

const keyOf = (request) => (typeof request === "string" ? request : request.url);

export const idbCache = {
  async match(request) {
    const rec = await tx("readonly", (s) => s.get(keyOf(request)));
    if (!rec) return undefined;
    return new Response(rec.blob, { status: 200, headers: rec.headers });
  },
  async put(request, response) {
    const blob = await response.blob();
    const headers = {};
    response.headers.forEach((v, k) => (headers[k] = v));
    headers["content-length"] = String(blob.size);
    await tx("readwrite", (s) => s.put({ blob, headers, size: blob.size, at: Date.now() }, keyOf(request)));
  },
  async delete(request) {
    await tx("readwrite", (s) => s.delete(keyOf(request)));
    return true;
  },
  async keys() {
    return tx("readonly", (s) => s.getAllKeys());
  },
};

// Model ids (e.g. "onnx-community/gemma-3-1b-it-ONNX") whose weights are in the cache.
export async function cachedModelIds() {
  const ids = new Set();
  try {
    for (const k of await idbCache.keys()) {
      const m = /^https:\/\/huggingface\.co\/([^/]+\/[^/]+)\/resolve\/[^/]+\/onnx\/.+\.onnx(_data(_\d+)?)?$/.exec(k);
      if (m) ids.add(m[1]);
    }
  } catch {}
  return ids;
}

// Bytes held by the cache, and the browser's quota.
export async function cacheUsage() {
  let bytes = 0;
  try {
    const recs = await tx("readonly", (s) => s.getAll());
    for (const r of recs) bytes += r.size ?? 0;
  } catch {}
  const est = await navigator.storage?.estimate?.().catch(() => null);
  return { bytes, quota: est?.quota ?? null };
}

export async function clearCache() {
  await tx("readwrite", (s) => s.clear());
}
