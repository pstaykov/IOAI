# IOAI task patterns (research notes)

Source of truth for every practice notebook in this repo. Each notebook header cites the
pattern IDs (P1–P16) and the real tasks (T-codes) it is modelled on.

Researched 2026-09-29 from the official repos, reading the task notebooks, statements, baselines,
reference solutions, graders and hints directly:

- `IOAI-official/IOAI-2024`: only `On-Site-Round/` is in the repo (3 tasks + solutions). The
  take-home statements are **not** in the repo. I fetched them from the zip linked by
  `open-cu/awesome-ioai-tasks` (`ioai-official.org/wp-content/uploads/2025/06/At-home-problems.zip`).
  I did not open the "best solutions" zips.
- `IOAI-official/IOAI-2025`: `At-Home-Round/` (Chameleon, Radar, Weather), `Individual-Contest/`
  (6 tasks, each with a `Solution` notebook and `metrics.py`) and `GAITE-Contest/` (Word
  Segmentation, Synthetic Speech Detector). GAITE task 3 "Resonance Elf" is credited in the README
  but **not present** in the repo.
- `IOAI-official/IOAI-2026`: `At-Home-Round/` (3 notebooks), `Individual-Contest/` (6 tasks, each
  with `statement.md`, the original baseline and grader, and an unofficial Colab baseline) and
  `GAITE-Contest/` (hints only; same tasks). Task data is on Hugging Face
  (`IOAI-official/ioai-2026-*`), not in the repo. I did not download the datasets.
- `IOAI-official/IOAI-AI-Models-Track` (README + Kaggle requirements only).

Not found anywhere: the 2025 "score unification" formula. Statements say "the highest score by the
Scientific Committee … and the baseline score … are used for score unification". They point to an
"Appendix Platform Mechanisms and Restrictions" that is not published. I have not guessed it.

---

## 1. Task catalogue

`Given` = what the starter notebook provides. `Fill` = what the contestant writes.

### 2024 (Burgas). Colab + L4, submissions were files and notebooks sent to the organisers

| ID | Task | Type | Data | Given / Fill | Submission | Metric | Limits | Models / APIs |
|---|---|---|---|---|---|---|---|---|
| T24-H-ML | Save the Factory (take-home) | Feature engineering, binary classification | Each sample is a `187×8` array. Train/val pickle, test released 48 h before the deadline | Given: loading, plots, **fixed** `DecisionTreeClassifier(max_depth=20)` and `(max_depth=4)` eval functions. Fill: the feature function | `efficient_test_predictions.txt`, `super_duper_efficient_test_predictions.txt` + notebook | ROC AUC on hard predictions | 5 min CPU (Colab) incl. feature generation | none (sklearn only) |
| T24-H-NLP | Help BOBAI: classify an unknown language (take-home) | Text classification, low-resource | Encrypted text, a small labelled set plus a larger raw corpus (HF, gated) | Given: full HF `Trainer` baseline on `bert-base-multilingual-uncased` (`DataCollatorWithPadding`, `evaluate` f1 macro, `push_to_hub`). Fill: improve it (the raw corpus invites MLM continued pre-training) | `test_predictions.txt` (one int per line) + HF Hub model + notebook | Macro F1 | train < 8 h L4, 500 samples inferred < 5 min | mBERT only |
| T24-H-CV | Lost in Translation (take-home) | Diffusion fine-tuning (swap zebra/giraffe) | Own/extra public data allowed | Given: `diffusers` walkthrough of tokenizer → text encoder → UNet → VAE. Fill: update UNet/VAE weights only | HF Hub model + Colab | (judged generation) | ≤ 3 h L4 | `lambdalabs/miniSD-diffusers` |
| T24-O-BOBAI | Help BOBAI 2: 5 → 7 classes | Class-incremental, **no new learned params** | Cached mBERT pooled embeddings `(…, 768)` + labels. Frozen `Linear(768,5)` in `base_classifier.pth` | Given: `SevenWayClassifier` wrapper, `compute_f1`, `inference`, a DO-NOT-CHANGE test cell. Fill: `__call__` (means and distances allowed) | `{team}_predictions.txt`, `submission.csv` (ID,class) | Macro F1 | reproduce < 1 h L4; 500 samples < 2 min | frozen linear head. Reference: kNN on embeddings routes to base head or new class |
| T24-O-HYPER | Lost in Hyperspace | Feature engineering, regression ×3 | `(N,5,5,5,6)` arrays with symmetries; pickle with train/val/live_test | Given: DO-NOT-CHANGE `test_solution` using `LinearRegression()` and asserting ≤ 300 features, `SCALING_WEIGHTS`. Fill: `feature_extractor(X)` | `predictions.csv` (ID,y1,y2,y3) | Mean of 3 scaled RMSEs | 5 min CPU each; no supervised feature extractors, no pretrained models, no AutoML | none. Reference: 6× axis-swap augmentation + PCA(299) + one domain feature |
| T24-O-COW | Madarian Cow | Steer text-to-image without retraining | HF dataset of captions + images (cows with hydrants included) | Given: `Magic(nn.Module).forward(latents, text_embeddings_mean)` stub, a fixed `custom_inference` CFG denoising loop, a DETR-based scorer. Fill: `Magic.forward` | `predictions.json` of modified latents / means for test embeddings | Detector (DETR r101, thr 0.6): cow ⇒ cow+hydrant, others ⇒ class & no hydrant | Colab GPU | `miniSD-diffusers`, `facebook/detr-resnet-101`. Reference: MLP "is it a cow?" on mean text embedding + swap in VAE-encoded cow+hydrant latents + learned shift of the text mean |

### 2025 (Beijing). Bohrium platform, Python 3.12.7, pinned `requirements.txt` (`pip` not allowed on-site)

Common frame on every 2025 task: statement markdown, then a **seed cell** (random / numpy / torch /
cuda + `cudnn.deterministic`), data loading, a deliberately weak baseline, a local eval, and a
submission cell that zips predictions for **test A (public leaderboard) and test B (private,
final)**. Hidden sets are reached through `os.environ["DATA_PATH"]`. Each statement gives a
**baseline score** and a **Scientific Committee (SC) score** that are used for "score unification",
so the target band is always visible.

| ID | Task | Type | Data | Given / Fill | Submission | Metric | Limits | Models / APIs |
|---|---|---|---|---|---|---|---|---|
| T25-H-CHAM | Chameleon (home) | Ranking: icon hints → word | 118 icon descriptions, 20-example validation, 100 options per game | Given: SentenceTransformer baseline and a `model.fit` CosineSimilarityLoss example. Fill: `guess_words(hints, choices)` → 10 ranked | function | 0.9·Hits@10 + 0.1·NDCG@10 | < 1B params, no API at inference | `all-MiniLM-L6-v2` (any < 1B) |
| T25-H-RADAR | Radar (home) | Binary semantic segmentation | `.pt` tensors `7×50×181` (6 heatmaps + label) | as T25-RADAR | | | | |
| T25-H-WEATHER | Satellite Weather | Binary segmentation + metadata | GOES-16 16-channel patches 128² / 256², MRMS masks, metadata CSV (lat/lon/time → sun elevation). Train biased to rainy scenes, val realistic with corrupted channels | Given: pretrained U-Net `model_weights.pth`, utilities. Fill: fine-tune / post-process | `submission.zip` ← `pred_a.npz`, `pred_b.npz` with `Y_pred_128 (51,128,128)`, `Y_pred_256 (183,256,256)` bool | (mean Dice + image-level rain accuracy) / 2 | | provided U-Net only |
| T25-RADAR | Radar (D1T1) | 5-class segmentation, heavy imbalance | 1800 train / 500 val / 500 test `.mat.pt`, `7×50×181`, labels −1..3 | Given: `CustomDataset` (labels +1 → 0..4), 3-conv baseline, train loop, inference → per-pixel CSV. Fill: model/loss | `submission.zip` ← `submission_val.csv`, `submission_test.csv` (`filename, pixel_0..pixel_9049`) | Weighted pixel accuracy: bg ×1, object ×50, normalised | notebook run on grader | none. SC 0.90, baseline 0.67. Reference: 6-branch encoder–decoder + focal loss |
| T25-CHICKEN | Chicken Counting (D1T2) | Density-map regression (counting) | 100 train images 720×1280 + density 180×320; 100 val, 100 test hidden | Given: `FeatureExtraction` (4 dilated convs) + `load_pretrained_weights_partial(base.pth)`, 1-conv `DensityDecoder`, L1 training, eval. Fill: `DensityDecoder` / training | `submission.npz` with `pred_a`, `pred_b` `(100,1,180,320)`, all ≥ 0 | exp(−mean relative count error) | "UNET easily fills GPU memory" | provided partial encoder. SC 0.89, baseline 0.71 |
| T25-CONCEPTS | Concepts (D1T3) | **Clue giving** for a black-box LLM guesser | 30 train labels × 100 options, 118 markers; test A/B 150 each | Given: vLLM `LLM("facebook/opt-125m")` with `GuidedDecodingParams(json=schema)` from a pydantic `Hints` model with `Literal[...]` ids, a `GameClient`, a judge API with 12,500 calls. Fill: `ClueGiver.construct_clues` | `clues_a.jsonl`, `clues_b.jsonl` (list[list[int]], ≤ 4 seqs × ≤ 8 ids) | 0.9·Hits@10 + 0.1·NDCG@10 | 10 min offline; ≤ 2 GB attached dataset; 15 submissions; judge API and LLM proxy **not** available at inference | offline allow-list: MiniLM/mpnet/e5/gte/arctic/bge/UAE/mxbai embedders, Qwen3-0.6B, Qwen2.5-0.5B(-Instruct), opt-125m/350m; vllm, sglang, unsloth. Dev-time **LLM proxy** (OpenAI SDK `AsyncClient`, `base_url`; gpt-4.1*, gpt-4o*, gemini-2.5-pro/flash, kimi-k2, qwen3-235b, claude-sonnet-4; $10 credit). Reference: distil API-LLM clues into Qwen3-0.6B via unsloth LoRA, serve with vLLM guided decoding |
| T25-RESTROOM | Restroom Icon Matching (D2T4) | Cross-domain image retrieval | 82 restrooms × 4 images; test: query crops vs gallery originals (10 / 30 queries) | Given: OpenAI `clip` ViT-B/16 zero-shot embeddings, cosine top-k matching. Fill: training a matcher | `submission_a.npy`, `submission_b.npy` (1-based gallery ids) | Top-1 match accuracy | time limit (0 if exceeded) | CLIP. SC 0.90, baseline 0.77. Reference: CLIP-ReID (unfreeze last blocks) |
| T25-ANTIQUE | Antique Authentication (D2T5) | Semi-supervised binary tabular | 500 train rows (5 features, label 1/−1/0 = unknown); val 500, test 500 | Given: SVC after randomly filling unknown labels. Fill: model | `submissionA.csv`, `submissionB.csv` (one ±1 per line, **no header**) in zip | Accuracy | | sklearn. SC 0.98, baseline 0.46. Reference: SpectralClustering → majority label per cluster → SVC |
| T25-PIXEL | Pixel Efficiency (D2T6) | Keep ≤ 6.25 % pixels (one rectangle) so frozen CLIP still classifies | 700 train 224² images + species; 698 test (30/70 A/B) | Given: HF `CLIPModel`/`CLIPProcessor` (`openai/clip-vit-large-patch14`), label list + `"other"`, centre-crop baseline, mask helpers. Fill: `((top,left),(bottom,right))` per image | `submission.jsonl` | Zero-shot accuracy on masked images (invalid mask = wrong) | < 8 min for 698 images | frozen CLIP. SC 0.83, baseline 0.19. Reference: extra mask tokens with a custom attention mask score every rectangle in one pass |
| T25-G-WORDSEG | Combinatorial Word Segmentation (GAITE) | Char-level sequence labelling | 94k German compounds → 0/1 end-of-word arrays; val/test ≈ 11.8k | Given: char vocab (0 = pad), padding `collate_fn` returning lengths, per-char one-hot MLP (≈ 0 F1), **padding mask applied before BCE**. Fill: model ("Embedding + LSTM recommended") | zip ← `submissionval.json`, `submissiontest.json` | Mean per-word F1 of boundaries | 20 min, notebook must include training; 50 submissions | none |
| T25-G-SPEECH | Synthetic Speech Detector (GAITE) | Binary audio classification on mel spectrograms | `.pt` mel tensors, `bonafide/` vs `spoof/` (~11k train) | Given: DO-NOT-MODIFY `SpectrogramDataset` returning `{'spectrogram','label'}`, `resnet18` with `conv1` swapped to 1 input channel and `fc` → 2, 1-epoch train (lr=0.1!). Fill: model/training | zip ← `submissionA.csv`, `submissionB.csv` (0/1, no header) | Macro F1 | 20 min | any offline pretrained (ResNet18 weights provided). SC 0.95 |

### 2026 (Astana). Yandex Contest + JupyterLab, submit `solution.ipynb` by git push

Every Individual task: **one GPU (≈ 16 GB), no internet, 5 GB storage, `solution.ipynb` ≤ 1 MB**
(20 MB for Ghost), **time limit 5–12 minutes covering grade-time training plus inference**. The
notebook is **re-run** at grading with `dataset/test_public/` silently replaced by hidden
`test_leaderboard_a` (public) / `test_leaderboard_b` (final). Models must be loaded from **local
paths** (`models/...`); a hub id triggers a download and fails. Baselines are deliberately
near-zero ("they show the input/output contract"). GAITE = same tasks plus hints.

| ID | Task | Type | Data | Given / Fill | Output | Metric | Limit | Allowed models |
|---|---|---|---|---|---|---|---|---|
| T26-H-NIGHT | Operation Night Watch (home) | Audio class-incremental learning (16 → 29 classes) | 5 s 16 kHz clips; 491 retained old clips (3–62 per class), 792 new (24–60 per class) | Given: `ASTFeatureExtractor` + `ASTForAudioClassification` checkpoint, `AudioManifest` Dataset, `Collator`, embeddings = mean of CLS + distill tokens, confusion matrices, `competition_score`. Fill: grow the head, fight forgetting (replay, KD, freezing/LoRA) | model | ½·acc_old + ½·acc_new | ~10 min GPU training | the provided AST only |
| T26-H-ROBOT | Robot Delivery Academy (home) | Behaviour cloning on an 8×8 grid | pickled demos with small budget | Given: simulator, observation = 6-channel grid + vector, MLP baseline, episode rollouts. Fill: model ("preserve 8×8 spatial structure") | action lists per scenario | episode success | | none |
| T26-H-WILKINS | Analytical Language of John Wilkins (home) | Interactive 20 questions against an LLM oracle | ~1400 animals, ~500 yes/no questions | Given: `Interactor.ask/guess`, `evaluate`, fixed-question reference. Fill: `MySolution.solve` (precompute the animal × question table with **your own copy of the oracle LLM**, batched; greedy information-gain questions; robust to noisy bits) | screenshot + class | per row max(0, 1[correct] − 0.02·queries) | 15 queries/row | oracle = `Qwen2.5-3B-Instruct` at temperature 0 |
| T26-FIND | Find the Order (D1T1) | Reorder shuffled dialogue turns (audio) | 1,288 train dialogues, 7–20 `.wav` turns each, 44.1 kHz; `prefix.json` gives the first two turns | Given: identity-permutation baseline + snippets: wav2vec2 mean-pooled embeddings, `Qwen2.5-0.5B` `generate`, Whisper-small `processor(...).input_features` → `generate(language="en", task="transcribe")`. Fill: everything | `answers.json` {dialogue: rank permutation} | Pairwise ordering accuracy 1 − I/M | 10 min | wav2vec 2.0, Whisper (any size, encoder usable as features), Qwen2.5-0.5B. Hint: ASR + MFCC speaker clusters + alternating speakers + search for the most natural order |
| T26-ROBOT | Robot Chasing (D1T2) | Predict next action from grid + NL mission | 60k train snapshots (10k/robot), 3,600 test; `8×8×2` object/colour grid, direction, mission text, carrying | Given: "always up" baseline. Fill: all | `predictions.json` list of ints | Mean per-robot accuracy | 5 min | none. Hint: regex mission parser + relative offsets + per-robot classifiers |
| T26-POTATO | Potato (D1T3) | Interactive semantic search | 1,602-word vocab + public embeddings `(1602, 2560)`; judge uses private embeddings | Given: stdin/stdout JSON protocol cell (do not edit), `PublicEmbeddingPlayer.respond`. Fill: player strategy | protocol (debug prints to **stderr**) | per game 1 − 0.02·max(0, t − 10), 0 if > 30 turns | 10 min for 120 games | `Qwen3-Embedding-0.6B`, `bge-m3` (local paths); numpy, torch, sentence-transformers |
| T26-DOUBLE | Double Agent Dilemma (D2T4) | Differential adversarial perturbations | 100 train / 100 public images (ImageNet classes, varied sizes) + labels | Given: models from local files, exact `preprocess` (Resize 256 → CenterCrop 224 → Normalize), `check()`, official `compute_score`, zip builder. Fill: `Solution._attack` | `submission.zip` of `{i}_a.pt`, `{i}_b.pt` (`3×H×W` at original resolution) | (A + B)/(2M) × PF, PF = 1.5 − σ(1e5 · mean L2/pixel) | 12 min | `torchvision resnet18`, `timm vit_tiny_patch16_224`. Hint path: mask search → Adam on δ with CE(keep) − CE(fool) → early stop → L∞ clamp / L2 penalty |
| T26-GHOST | Ghost of the Machine (D2T5) | Human → LLM changepoint in text | 1,221 train passages (500–800 words) + answers; 380 per test split | Given: mean-fraction baseline (28.6). Fill: all | `answers.jsonl` {id, boundary_char_index} | mean exp(−\|p − t\|/100) | 10 min incl. fine-tuning | `bge-base-en-v1.5` only (frozen or fine-tuned); sklearn free. SC 93.4: fine-tuned encoder + sentence-level changepoint |
| T26-FIELD | IOAI Field (D2T6) | Fit an implicit 2-D field from a generator, from scratch | No dataset: `make_batch(cfg, include_regions, stratified)` generator + JSON config (hidden test config) | Given: `custom_model.py` MLP with Dropout, `evaluate_model` per-region explainer, trivial constant models. Fill: `CustomModel` + training → `model.pt` | `model.pt` + `custom_model.py` (torch only) | Mean of 5 region scores (normalised MAE; last region = dropout-induced std within [−2026, 2026]) × 100; **halved if > 20,260 params** | 5 min | none (no pretrained, no `torch.rand*` at inference) |

---

## 2. Recurring patterns (cite these IDs)

**P1: Notebook skeleton.** Statement (story → dataset → task → submission → scoring with
baseline and SC scores → requirements) → seed cell → data loading → **deliberately weak baseline** →
local evaluation that mirrors the official metric → submission cell that writes exact filenames
(often zipped) for **both** hidden splits. You edit a marked region; "DO NOT CHANGE" cells hold
loaders, protocol code and scorers. (all 2025 and 2026 tasks)

**P2: Hidden-test contract.** Hidden data is swapped in at grade time (`DATA_PATH` env var in 2025;
folder overlay of `dataset/test_public/` in 2026). Code must never read test labels, must not
hard-code sizes, and must regenerate everything from the notebook (2026 even re-trains at grade
time). Answers go into a strict format, and a malformed file scores 0: header / no header, 1-based
ids (Restroom), rank-vs-order permutations (Find the Order), `≥ 0` densities (Chicken), shapes at the
**original** image resolution (Double Agent). (T25-*, T26-*)

**P3: Tight compute, offline, allow-listed models.** 5–20 min total runtime on one GPU; no internet;
only the listed checkpoints, loaded from local paths (`HF_HUB_OFFLINE=1`). Budget for grade-time
training, batch everything, and time your notebook. (T25-CONCEPTS, T25-G-*, all T26)

**P4: Adapt a frozen / deployed model under constraints.** Add classes with no new parameters
(T24-O-BOBAI); change only text-mean + initial latents (T24-O-COW); load a partial pretrained
encoder and train a decoder (T25-CHICKEN); grow a head without forgetting (T26-H-NIGHT); a fixed
provided U-Net (T25-H-WEATHER). Skills: `load_state_dict(strict=False)` / partial dicts, freezing,
param groups, replay, distillation, prototypes.

**P5: Feature engineering for a fixed model.** The model is locked (`LinearRegression`, depth-limited
`DecisionTreeClassifier`, ≤ 300 features, 5 min CPU) and you engineer features: symmetry augmentation,
PCA, statistics, domain features. (T24-H-ML, T24-O-HYPER)

**P6: Embedding similarity, retrieval and ranking.** Encode with a sentence or CLIP encoder, normalise,
cosine / dot product, top-k, Hits@k / NDCG, optionally fine-tune with a contrastive loss.
(T25-H-CHAM, T25-CONCEPTS, T25-RESTROOM, T26-POTATO, T26-GHOST, T24-O-COW's cow detector)

**P7: Few labels, semi- or self-supervised.** Unknown labels mixed in (T25-ANTIQUE: spectral clustering
+ pseudo-labels), tiny train sets (T25-CHICKEN 100 images, T26-H-NIGHT 3–60 clips per class), large raw
corpus + small labelled set (T24-H-NLP).

**P8: Dense prediction with imbalance-aware metrics.** Per-pixel CE on `[B,C,H,W]` logits vs
`[B,H,W]` long targets, label shifts (−1..3 → 0..4), weighted metrics (bg ×1 vs object ×50), Dice +
image-level accuracy, density maps whose sum is a count. (T25-RADAR, T25-H-WEATHER, T25-CHICKEN)

**P9: Sequences with padding and masks.** Build a vocab (0 = pad), pad in `collate_fn`, keep lengths,
mask padded positions out of the loss and the metric, Embedding → (Bi)LSTM / Transformer. (T25-G-WORDSEG,
plus every HF tokenizer call with `attention_mask`)

**P10: Audio pipelines.** Load with `librosa.load(path, sr=16000)`, then either a feature extractor /
processor (`Wav2Vec2Processor`, `WhisperProcessor(...).input_features`, `ASTFeatureExtractor`) or a
mel spectrogram treated as an image (1-channel `conv1` ResNet). Pool frame embeddings (mean, or CLS +
distill). ASR via `generate(language=..., task="transcribe")`. (T25-G-SPEECH, T26-FIND, T26-H-NIGHT)

**P11: LLMs, local and API.** Local: small instruct / base models (Qwen2.5-0.5B/3B, Qwen3-0.6B) via
`transformers` `generate` or vLLM, chat templates, **batched** prompts, constrained / JSON output
(vLLM `GuidedDecodingParams(json=pydantic_schema)`), yes/no logits as an oracle. API (development only,
never at inference): OpenAI-compatible SDK with a custom `base_url`, pydantic structured outputs
(`beta.chat.completions.parse`), asyncio `Semaphore` concurrency + `tenacity` exponential-backoff
retries, credit budgeting. Typical use: generate labels / synthetic data, then **distil** into an
allowed small model (T25-CONCEPTS reference). (T25-CONCEPTS, T26-H-WILKINS, T26-FIND)

**P12: Interactive / search protocols.** The judge answers queries under a budget: pairwise "closer"
comparisons (T26-POTATO), yes/no questions (T26-H-WILKINS), black-box guesser API (T25-CONCEPTS).
Winning moves: precompute a table offline, maintain a candidate set, choose queries by information
gain, be robust to noisy answers, keep stdout clean for the protocol.

**P13: Behaviour cloning on grids.** Observation (grid channels, direction, carrying, mission text) →
action classifier; per-agent patterns; hand-built relative features beat raw inputs. (T26-ROBOT,
T26-H-ROBOT)

**P14: Optimise inputs against frozen models.** Gradients w.r.t. the input or search over masks /
crops: differential adversarial examples with norm penalties and early stopping (T26-DOUBLE); keep a
crop that preserves a CLIP prediction (T25-PIXEL); edit latents / text embeddings (T24-O-COW).
Requires `requires_grad` on the input, frozen params, a differentiable copy of the exact preprocess,
and `clamp` to the valid pixel range.

**P15: From scratch under a parameter budget.** Count `sum(p.numel())`, route by region, normalise
huge-range targets, stratified sampling, dropout as a controlled noise source. (T26-FIELD; T25-G-WORDSEG
"must include training")

**P16: Heuristics first, then learn.** Official hints climb a ladder: look at the data → simple rule
(snap to sentence starts, marker words) → embeddings + sklearn classifier → fine-tune the allowed
encoder → combine (T26-GHOST, T26-DOUBLE, T26-FIELD hints). Validate locally after every change; the
local public split is a faithful preview.

---

## 3. Environment facts that shape the notebooks

- 2025 pinned stack (`IOAI-2025/requirements.txt`): Python 3.12.7, `accelerate 1.8.1`, `anthropic 0.60`,
  `bitsandbytes 0.46.1`, catboost, vllm, sglang, unsloth (per the Concepts statement).
- 2026 AI-models-track Kaggle stack (`kaggle-requirements.txt`): torch 2.13, torchvision 0.28,
  torchaudio 2.11, transformers 5.14, datasets 5.0, sentence-transformers 5.6, peft 0.19, trl 1.9,
  timm **absent**, librosa 0.11, soundfile, scikit-learn 1.9, xgboost / lightgbm / catboost,
  albumentations, torchmetrics, spacy, gensim, nltk. The human contest env list is not published, but the
  baselines import `librosa`, `transformers`, `timm`, `sentence-transformers`, `scipy`.
- API changes verified locally (transformers 5.17, datasets 5.0.1, openai 3.22, sentence-transformers
  6.1, librosa 1.0, torch 2.7.1) and against current docs:
  - `from_pretrained(..., dtype=torch.bfloat16)`; the 2026 baseline already uses `dtype=`.
  - `Trainer(processing_class=tokenizer)` (the `tokenizer=` argument is gone); `TrainingArguments(eval_strategy=...)`
    (`evaluation_strategy` removed).
  - `openai`: `client.chat.completions.parse(response_format=PydanticModel)` is GA; the
    `client.beta.chat.completions.parse` alias used in the 2025 proxy tutorial still exists.
  - `datasets.Audio` decodes through **torchcodec** (`AudioDecoder` objects), which needs FFmpeg. Notebooks
    use `Audio(decode=False)` + `soundfile` / `librosa` so they work without torchcodec.
  - torchaudio 2.9 removed `load` / `save` / `io` / `datasets` (decoding moved to TorchCodec); `transforms`,
    `functional`, `models` remain. Notebooks use librosa for I/O, as the 2026 baselines do.
  - torchvision: `weights=ResNet18_Weights.DEFAULT` (the `pretrained=` flag is deprecated; the 2025 GAITE
    baseline shows the warning).

## 4. What this means for practice

1. Know the skeleton cold: seed cell, `DATA_PATH` / overlay handling, and writers for CSV (header or
   not), npz, npy, json, jsonl and zip, plus a validator that checks shape, dtype, range and
   permutation-ness.
2. Most points come from **using a given frozen model well** and from **validation discipline**, not
   from novel architectures.
3. Library fluency that recurs: HF tokenizers / processors (padding, truncation, `return_tensors`,
   `attention_mask`, `sampling_rate`), pooling, `torch.no_grad` / `eval`, dtype of labels, partial
   `state_dict` loading, CLIP text/image features, Whisper `generate`, chat templates + batched
   generation, pydantic-constrained outputs.
4. Time yourself: every 2026 task re-runs training inside a 5–12 minute window.
