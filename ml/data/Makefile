# Dataset factory (PLAN_V2 stage C, CONTEXT_V2). Run from anywhere: paths are relative to the repo root.
REPO := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../..)
# the real screen path (ml.screens + ml.ocr) needs the OCR venv; the generator itself is stdlib-only
PY    ?= $(REPO)/.cache/ml/ocr/venv/bin/python
PYPUB ?= $(REPO)/.cache/ml/data/venv/bin/python
OUT   ?= $(REPO)/.cache/ml/data/synth_v3
# render + OCR results per render spec: a rebuild that changes only the channel or the filters skips them
CTX_CACHE ?= $(REPO)/.cache/ml/data/ctx_cache
N     ?= 3000
DEV   ?= 300
TEST  ?= 300
WORKERS ?= 5
MINED ?= $(REPO)/ml/roundtrip/samples/mined/full
# shared server: heavy jobs in a memory-capped scope at low priority
MEM   ?= 24G
HEAVY := env MURMUR_MEM=$(MEM) $(REPO)/tools/heavy nice -n 15
STAGE ?= all
# extra scenarios for cells that fall short (measured yields, docs/ml/DATA_CARD.md), in ";"-separated layers (a later
# layer never changes an answered batch); keep the same value for every
# command of one run, or the reserve scenarios drop out of the build
RESERVE ?= train/context_resolution=150,train/near_miss=58,train/injection=18,train/list=5,dev/context_resolution=15,dev/near_miss=6,dev/injection=3,dev/list=1,test/context_resolution=15,test/near_miss=6,test/injection=3,test/list=1;train/address=6,train/injection=6,dev/context_names=4,dev/address=2;train/near_miss=16,train/meaningful_filler=5,train/address=5,train/injection=3,dev/near_miss=5,test/meaningful_filler=3;train/injection=4;train/near_miss=4,train/injection=6,dev/context_irrelevant=3
MINED_ARG := $(if $(MINED),--mined $(MINED),)

.PHONY: venv test stub proof synth rebuild recheck sample public validate contam disfl-probe realism-measure realism-calibrate audit girgur-hebrew girgur-merge

venv:   # secondary venv for the public converters (pyarrow) and tests
	uv venv -q -p 3.13 $(REPO)/.cache/ml/data/venv && uv pip install -q -p $(PYPUB) -r $(REPO)/ml/data/requirements.txt

test:
	cd $(REPO) && $(PY) -m pytest -q -p no:cacheprovider ml/data/tests

# offline plumbing check (no network, no OCR): stub teacher, fixture screens, toy scale
stub:
	cd $(REPO) && $(PY) -m ml.data.generate --teacher stub --screens fixture --out $(REPO)/.cache/ml/data/synth_runs/stub \
	    --n 300 --dev 30 --test 30 --workers 2 $(MINED_ARG)

# 216 teacher scenarios, real teacher + real render/OCR (38 requests, measured ~7 min teacher + ~6 min build)
proof:
	cd $(REPO) && tools/with-teacher $(HEAVY) $(PY) -m ml.data.generate --out $(REPO)/.cache/ml/data/synth_runs/proof \
	    --limit 216 --max-requests 45 --workers 3 --screens real $(MINED_ARG)

# THE full set: ~633 teacher requests (see docs/ml/DATA_CARD.md for the measured wall time)
synth:
	cd $(REPO) && tools/with-teacher $(HEAVY) $(PY) -m ml.data.generate --out $(OUT) --n $(N) --dev $(DEV) --test $(TEST) \
	    --workers $(WORKERS) --screens real --stage $(STAGE) $(if $(RESERVE),--reserve '$(RESERVE)',) $(MINED_ARG) \
	    --ctx-cache $(CTX_CACHE)

# rebuild OUT from its answered teacher shards (never sends a request; ~2 min with a warm CTX_CACHE, ~22 min cold)
rebuild:
	cd $(REPO) && $(HEAVY) $(PY) -m ml.data.generate --out $(OUT) --n $(N) --dev $(DEV) --test $(TEST) \
	    --workers $(WORKERS) --screens real --offline $(if $(RESERVE),--reserve '$(RESERVE)',) $(MINED_ARG) \
	    --ctx-cache $(CTX_CACHE)

# real round trip -> per-model style table (final marks, first-word drops, word miss): ml/data/probe/asr_style.v1.json
realism-measure:
	cd $(REPO) && $(PY) -m ml.data.realism measure

# refit the per-(model, language) word-noise factors on a built set: ml/data/probe/word_noise_scale.v1.json
realism-calibrate:
	cd $(REPO) && $(PY) -m ml.data.realism calibrate --data $(OUT)

# the data-review numbers (terms without screen support, guessed words, garble scripts, noise, punctuation, numbers,
# artifact strings) of a built set
audit:
	cd $(REPO) && $(PY) -m ml.data.realism audit --data $(OUT)

# after any prompts/render.py change: re-apply the CONTEXT_V2 rules to the stored set (no OCR, no teacher)
recheck:
	cd $(REPO) && $(PY) -m ml.data.recheck --run $(OUT)

# committed sample for reviewers (<= 200 lines) from a finished run directory RUN=...
sample:
	cd $(REPO) && $(PY) -m ml.data.make_sample --run $(RUN) --out ml/data/samples/synth_v2.sample.jsonl --n 200

public:
	cd $(REPO) && $(HEAVY) $(PYPUB) -m ml.data.public.convert --out $(REPO)/.cache/ml/data/public_out

validate:
	cd $(REPO) && for f in $(OUT)/train.jsonl $(OUT)/dev.jsonl $(OUT)/test.jsonl; do $(PY) -m ml.common.validate --strict $$f || exit 1; done

# held-out eval sets (prompts/eval) vs a run and vs the teacher prompt's quoted examples
contam:
	cd $(REPO) && $(PY) -m ml.data.contam --prompt --data $(OUT)

# Girgur (ru + en + he): generate Hebrew+mixed shard (1200/150/150) and merge with .cache/ml/data/final -> .cache/ml/data/girgur
HEBREW_OUT ?= $(REPO)/.cache/ml/data/hebrew_v1
GIRGUR_BASE ?= $(REPO)/.cache/ml/data/final
GIRGUR_OUT ?= $(REPO)/.cache/ml/data/girgur
girgur-hebrew:
	cd $(REPO) && tools/with-teacher $(HEAVY) $(PY) -m ml.data.generate_hebrew \
	    --hebrew-out $(HEBREW_OUT) --base-dir $(GIRGUR_BASE) --out $(GIRGUR_OUT) \
	    --n 1200 --dev 150 --test 150 --workers $(WORKERS) --screens real --ctx-cache $(CTX_CACHE)

girgur-merge:
	cd $(REPO) && $(PY) -m ml.data.generate_hebrew --merge-only \
	    --hebrew-out $(HEBREW_OUT) --base-dir $(GIRGUR_BASE) --out $(GIRGUR_OUT)

# per-ASR-model rendering of hesitations / repeats: CPU TTS -> every ASR model (~35 min, ml/roundtrip runner,
# read-only use), then ml/data/probe/disfluency_render.v1.json
disfl-probe:
	cd $(REPO) && tools/heavy $(REPO)/.cache/ml/roundtrip/venv/bin/python -m ml.roundtrip.run_roundtrip \
	    --probe ml/data/probe/disfluency_probe.v1.jsonl --out $(REPO)/.cache/ml/data/disfl_probe --tts cpu \
	    --voices-per-sentence 1 --channels clean --asr all --workers 3 --threads 2 --mem-gb 14
	cd $(REPO) && $(PY) -m ml.data.probe.measure_disfluency .cache/ml/data/disfl_probe/roundtrip/disfluency_probe.v1.jsonl
