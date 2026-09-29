# Practice-notebook rebuild: progress (paused 2026-09-29)

Task: the "three notebook styles" plan (guided / drills / exercises for TraditionalML, DL, CV, NLP & Audio).
Research deliverable: `notes/ioai-task-patterns.md` (done). Nothing is committed yet.

## How the notebooks are made

Every notebook is generated from one source file in `src/<Area>/<NN_topic>.<guided|drill|exercise>.src`.
**Edit the source, not the .ipynb**, then rebuild.

- `nbsrc.py`: the source format. `@@md` / `@@code` start cells. In guided sources,
  `@@gap <hint>` … `@@as` … `@@end` marks a gap. In drill sources, `@@code drill=ID` marks a cell that is
  emptied for the student. In exercise sources, `only=task` / `only=sol` marks cells for one notebook only.
- `build.py [filter…]`: writes `<Area>/{guided,drills,exercises}/`, `<Area>/solutions/{guided/*.md,drills/*.py,exercises/*.ipynb}`,
  and gap-filled / reference-filled test copies to `_out/test/`.
- `run.py [--env IOAI_SHOW_FINAL=1] nb…`: executes with nbclient (cwd `_out/run/<Area>`) and appends to `_out/results.jsonl`.
  Executed copies go to `_out/executed/`.
- `gen/*.py`: hidden data generators for the exercises. `gen/*.blob` = base64(zlib(source)), pasted into the
  exercise so the formula is not visible.

Smoke test per topic (from the repo root, with the project venv):
```
.venv/Scripts/python.exe tools/notebook_build/build.py DL/02
.venv/Scripts/python.exe tools/notebook_build/run.py --env IOAI_SHOW_FINAL=1 \
  tools/notebook_build/_out/test/DL/guided/02_sequence_models.ipynb tools/notebook_build/_out/test/DL/drills/02_sequence_models.ipynb \
  DL/exercises/02_sequence_models.ipynb DL/solutions/exercises/02_sequence_models.ipynb
```

## Status

| Area / topic | guided | drill | exercise | calibration (LB-B: baseline → reference) |
|---|---|---|---|---|
| TraditionalML 01 tabular_pipeline | ✅ | ✅ | ✅ | 0.69 → 0.81 (macro F1) |
| TraditionalML 02 feature_engineering | ✅ | ✅ | ✅ | 1.00 → 0.12 (nRMSE, lower better) |
| TraditionalML 03 unsupervised | ✅ | ✅ | ✅ | 0.48 → 0.97 (accuracy) |
| DL 01 training_loop | ✅ | ✅ | ✅ | 0.74 → 0.905 (accuracy, ≤20,260 params) |
| DL 02 sequence_models | ✅ | ✅ | ✅ | 0.05 → 0.89 (segment F1) |
| DL 03 finetune_extend | ✅ | ✅ | ✅ | 0.46 → 0.85 (½ old + ½ new) |
| DL 04 embeddings_prototypes | ✅ | ✅ | ✅ | 0.49 → 0.73 (macro F1) |
| CV 01 transfer_classification | ✅ | ✅ | ✅ | 0.53 → 0.93 (accuracy) |
| CV 02 segmentation_density | ✅ | ✅ | ⚠️ too easy | 0.93 → 0.995. Needs a harder generator (see below) |
| CV 03 clip_zero_shot | written, not run | written, not run | not written | not calibrated |
| CV 04 adversarial_saliency | – | – | – | T26-DOUBLE mirror (resnet18 vs timm vit_tiny, PF = 1.5 − σ(1e5·L2/px)) |
| CV 05 diffusion_pipeline | – | – | – | T24-O-COW mirror (InternationalOlympiadAI/miniSD-diffusers + DETR; models prefetched in HF cache) |
| NLPAudio 01–07 | – | – | – | see plan below |

✅ = smoke-tested: guided with the gaps filled, drill with the reference code, exercise baseline and reference
solution. Both leaderboards were checked for exercises.

### Next steps
1. **CV 02 exercise**: rewrite `gen/radar.py` so the official 3-conv baseline scores about 0.65–0.75 and a U-Net about 0.9.
   Plan: classes differ by **shape** as well as channel ratio (suitcase small, chair wide, human tall; suitcase
   and chair have near-equal signatures), amplitudes 0.2–0.6 over gamma(2, 0.1) noise, per-sample channel jitter
   0.7–1.3, dashed walls, low-frequency clutter, and unlabelled "ghost" blobs with their own signature. Prototype it and
   calibrate before regenerating the blob. Then update the statement numbers (they currently say 0.70 / 0.95,
   which are placeholders).
2. CV 03: write the exercise (T25-PIXEL mirror: ≤ 6.25 % rectangle, CLIP ViT-B/32, classes + "other",
   `submission.jsonl`, 8 min for 500 images), then build and run all three.
3. CV 04, CV 05, then NLPAudio (new folder `NLPAudio/`), 7 topics: 01 text_classification (emotion; Trainer +
   manual loop), 02 text_encoders (bge-small; exercise = T26-GHOST-style changepoint on concatenated IMDB reviews),
   03 language_modeling (char-GPT drill from scratch; exercise = TinyStories sentence ordering with an LM,
   T26-FIND text version), 04 encoder_decoder (flan-t5-small headline generation on ag_news, ROUGE-L; MarianMT;
   BLIP captioning), 05 pretrained_llms (Qwen2.5-0.5B-Instruct batched chat generation; OpenAI-compatible API with
   `chat.completions.parse`, async Semaphore + tenacity, tested only against a local stub unless a key is
   available; exercise = T26-H-WILKINS 20-questions mirror), 06 audio_encoders (HuBERT; LibriSpeech validation.clean
   speaker ID / minds14), 07 audio_models (Whisper + chunking; Voxtral / Qwen2-Audio processors, large models
   optional; exercise = T26-FIND audio mirror on LibriSpeech chapters).
4. Step 5: README.md (structure + ladder), `notes/nlp-audio.md` (modelled on `notes/computer-vision.md`), and the final
   smoke-test report (what did not run and why).

## Findings to keep in mind
- transformers 5.17: `Trainer(processing_class=…)`, `eval_strategy`, `dtype=`; CLIP `get_*_features()` returns an
  object, and the projected embedding is `.pooler_output`.
- datasets 5.0: `Audio` decodes via torchcodec (not installed) → use `Audio(decode=False)` + soundfile/librosa.
  torchaudio ≥ 2.9 dropped `load`/`save`/`datasets`.
- On the RTX 3060, cuDNN TF32 makes results differ by ~1e-3 between batch sizes. Checks that compare across
  batch compositions use atol ≈ 2e-3 (RNN) or rtol 1e-2 (ResNet features).
- The local `CompuerVision/implementations/data/cifar-10-python.tar.gz` is a partial download (93.6 MB of
  170.5 MB). The notebooks use `uoft-cs/cifar10` from the HF Hub instead.
- Drill thresholds sit well below what the reference implementation reaches, to absorb GPU run-to-run variance.
