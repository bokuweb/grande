# omg-train

The Python side: LoRA + pointer head on a Gemma 4 base checkpoint with the
same packed branch layout the Rust runtime serves. Nothing here runs at
inference time; the outputs are `head.safetensors` (read by `omg-core`)
and a merged GGUF.

```
render.py   layout, byte-for-byte the same as omg-core::render; parity check against `omg render`
model.py    block-causal branch mask (full and sliding-window variants), PointerHead, DecisionModel
data.py     JGLUE train → TypeSafe-shaped records + gold
train.py    LoRA + head training, exports head.safetensors and the adapter
```

Status: written, **not yet run on Gemma 4**. Two places to expect adjustment:

- reaching the text decoder without `lm_head` on the multimodal E2B / E4B
  checkpoints (`text_backbone()` walks `.model.language_model`);
- the per-layer-type attention mask. Gemma 4 interleaves 512-token
  sliding-window layers with global ones; llama.cpp applies the window at
  inference, so training passes both masks (`{"full_attention", "sliding_attention"}`).
  If the installed transformers does not accept the mapping, states under 512
  tokens are unaffected; longer ones are not, and that is a real mismatch to fix.

Parity check (run after `omg render` dumps a request):

```python
from transformers import AutoTokenizer
from omg_train.render import Renderer, check_parity
import json
tok = AutoTokenizer.from_pretrained("google/gemma-4-E2B")
enc = Renderer(tok).encode(json.load(open("examples/isolation-ja.json")))
print(check_parity(enc, json.load(open("render.json"))))   # [] when identical
```

After training:

```bash
# merge the adapter into the base weights, then GGUF
python -c "from peft import PeftModel; ..."      # merge_and_unload → save_pretrained
python llama.cpp/convert_hf_to_gguf.py merged/ --outfile grande-e2b.gguf --outtype q8_0
omg probe --model grande-e2b.gguf --head runs/grande-e2b/head.safetensors --request examples/ticket-ja.json
```
