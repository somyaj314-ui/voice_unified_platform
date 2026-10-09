# Training notes

## Selected adapter

- Name: `indic-mio-spicor-fullshort-stage2`
- Base: `SPRINGLab/Indic-Mio`
- Method: PEFT LoRA, causal language modelling over experimental MioCodec audio tokens
- LoRA rank: 16
- LoRA alpha: 32
- LoRA dropout: 0.05
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
  `up_proj`, and `down_proj`
- Stage-two learning rate: `1e-5`
- Scheduler: constant with warm-up
- Epochs: 2
- Per-device batch size: 2
- Gradient accumulation: 8
- Effective batch size: 16 examples
- Warm-up ratio: 0.05
- Seed: 42

## Stage-two short-sentence dataset

- Source: IISc SPICOR English TTS Corpus, female speaker 001
- Selected short clips: 1,607
- Total selected duration: 1.987 hours
- Training: 1,451 clips, 1.791 hours (90.29%)
- Validation: 78 clips, 0.098 hours (4.93%)
- Held-out test: 78 clips, 0.098 hours (4.93%)
- Clip duration: mainly 3–5 seconds
- Split: stratified by domain and duration bucket

The stage-two adapter was initialized from the earlier
`indic-mio-spicor-short-lr2e5-v2` adapter and then trained on the larger short
sentence subset.

## Observed limitation

The adapter improved some Indian-English short sentences but did not eliminate
accent drift. Loss alone does not measure accent consistency. Evaluate with
human listening tests and an accent-focused test set. Fixed production lines
should use approved WAV caching.

