"""Unit tests for Hebrew + mixed (he, he_en, he_ru) data generation, ASR noise, and Girgur merging."""
from __future__ import annotations

import argparse
import random
from collections import Counter

import pytest

from ml.common import schema, text as T
from ml.data import generate as G, generate_hebrew as GH
from ml.data.synth import asr_noise as AN, build as B, channel as CH, disfluency as D, filters as F, sampler as S
from ml.data.synth.teacher_prompts import GEN_SYSTEM, gen_request


def test_common_text_hebrew_tokenization_numbers_and_phonetics():
    # Niqqud stripped and geresh/gershayim preserved inside words
    assert T.words("שָׁלוֹם צה״ל ג׳ ל-GitHub") == ["שלום", "צה״ל", "ג׳", "ל-GitHub"]
    assert T.is_heb("שלום") and not T.is_lat("ל-GitHub")
    assert T.guess_lang("שלום עולם מה שלומך היום") == "he"
    assert T.guess_lang("עשיתי push ל-GitHub בבוקר") == "he_en"
    assert T.guess_lang("יאללה בוא נדבר завтра בבוקר") == "he_ru"

    # Hebrew numbers including teens, compound vav-units, hundreds
    assert T.numbers_in("עשרים וחמישה בנובמבר", "he") == [25]
    assert T.numbers_in("חמישה עשר אחוז", "he") == [15]
    assert T.numbers_in("מאתיים וחמישים שקלים", "he") == [250]
    assert T.numbers_in("שלוש מאות", "he") == [300]

    # Cross-script phonetic similarity (Hebrew transliteration <-> Latin)
    assert T.phon_sim("גיטהאב", "GitHub") >= 0.75
    assert T.phon_sim("סלאק", "Slack") >= 0.90
    assert T.phon_sim("דוקר", "Docker") >= 0.70
    assert T.phon_sim("קליקהאוס", "ClickHouse") >= 0.75
    assert T.has_preamble("בטח, הנה הטקסט המתוקן:")
    assert not T.has_preamble("שלום, נתראה מחר.")


def test_hebrew_asr_noise_zipformer2_habits():
    # Transliterates Latin tech terms (including Hebrew-prefixed ל-GitHub), strips punctuation,
    # introduces homophone confusions and/or split prefixes over multiple seeds.
    spoken = "עשיתי push ל-GitHub, ובדקתי את Docker בבית עכשיו עם כל כך הרבה דברים."
    target = "עשיתי push ל-GitHub, ובדקתי את Docker בבית עכשיו עם כל כך הרבה דברים."
    ents = [{"text": "GitHub", "kind": "tech"}, {"text": "Docker", "kind": "tech"}]

    hyps = []
    all_ops = Counter()
    for s in range(40):
        res = AN.noise(spoken, target, "he_en", "zipformer2_he", 1.0, random.Random(s), entities=ents)
        hyps.append(res.hyp)
        for o in res.ops:
            all_ops[o["op"]] += 1

    # Punctuation dropped on zipformer2_he
    assert all("," not in h and "." not in h for h in hyps)
    # Latin tech terms transliterated into Hebrew
    assert any("גיטהאב" in h or "גיט האב" in h for h in hyps)
    assert any("דוקר" in h for h in hyps)
    assert all("GitHub" not in h and "Docker" not in h for h in hyps)
    # Homophone confusion and split prefix operators fire
    assert all_ops["conf"] > 0
    assert all_ops["split_prefix"] > 0

    # Hebrew surname homophone respelling preserves final forms
    sp_out, old, new = B.misspell_surname("דיברתי עם נועה כהן היום", "נועה כהן", random.Random(3))
    assert old == "כהן" and new and new != old and new in sp_out


def test_hebrew_disfluency_and_channel_routing():
    sp, ops = D.inject("אני בדרך למשרד, נדבר עוד מעט.", "he", 0.8, random.Random(7))
    assert ops and any(o["op"] in ("start_filler", "filler", "hesitation", "repeat", "restart") for o in ops)
    ch = CH.Channel(None)
    assert set(ch.models("he")) == {"zipformer2_he", "whisper_he"}
    picks = {ch.pick_model("he_en", random.Random(i)) for i in range(50)}
    assert "zipformer2_he" in picks and "whisper_he" in picks


def test_hebrew_plan_split_disjoint_and_ru_en_untouched():
    # RU+EN default plan remains identical and produces only ru/en/mixed
    ru_plan = S.plan(120, 30, 30, seed=1)
    assert {sc["lang"] for sc in ru_plan} <= {"ru", "en", "mixed"}

    # Hebrew profile produces he/he_en/he_ru with split-disjoint people and terms
    he_plan = S.plan(180, 45, 45, seed=101, lang_profile="hebrew")
    assert {sc["lang"] for sc in he_plan} == {"he", "he_en", "he_ru"}
    train_people = {p for sc in he_plan if sc["split"] == "train" for p in sc["people"]}
    held_people = {p for sc in he_plan if sc["split"] != "train" for p in sc["people"]}
    assert not (train_people & held_people)

    # Teacher prompt appends Hebrew rules ONLY for Hebrew batches (RU/EN prompt untouched)
    ru_ins, _ = gen_request([ru_plan[0]], S.TONES)
    he_ins, _ = gen_request([he_plan[0]], S.TONES)
    assert ru_ins == GEN_SYSTEM
    assert he_ins != GEN_SYSTEM and "Additional Hebrew rules" in he_ins


def test_hebrew_end_to_end_stub_and_girgur_merge(tmp_path):
    # 1. Generate small RU+EN base shard (offline stub + fixture)
    base_dir = tmp_path / "base_ru_en"
    G.run(argparse.Namespace(
        out=str(base_dir), n=60, dev=15, test=15, seed=1, lang_profile="ru_en",
        teacher="stub", mined=None, batch=6, overgen=1.3, concurrency=2, max_requests=0,
        max_retries=2, shard_size=25, limit=0, topup=False, screens="fixture",
        allow_fixture=True, workers=1, reserve="", stage="all", offline=False, sample=0, ctx_cache=None,
    ))

    # 2. Generate small Hebrew+mixed shard (offline stub + fixture)
    he_dir = tmp_path / "hebrew_shard"
    G.run(argparse.Namespace(
        out=str(he_dir), n=60, dev=15, test=15, seed=101, lang_profile="hebrew",
        teacher="stub", mined=None, batch=6, overgen=1.3, concurrency=2, max_requests=0,
        max_retries=2, shard_size=25, limit=0, topup=False, screens="fixture",
        allow_fixture=True, workers=1, reserve="", stage="all", offline=False, sample=0, ctx_cache=None,
    ))

    he_train = schema.read_jsonl(he_dir / "train.jsonl")
    assert len(he_train) >= 25
    assert {"he", "he_en", "he_ru"} & {r["lang"] for r in he_train}
    # Every Hebrew record passes strict contract validation and sets language.asr == "he"
    for r in he_train:
        assert schema.validate_record(r, check_render=True, strict=True) == []
        assert r["ctx"]["language"]["asr"] == "he"

    # 3. Merge RU+EN + Hebrew into Girgur output dir with recheck + leak check
    girgur_dir = tmp_path / "girgur"
    stats = GH.merge_datasets(base_dir, he_dir, girgur_dir, check_render=True, run_recheck=True)
    assert stats["final_by_split"]["train"] > len(he_train)
    assert stats["final_by_split"]["dev"] > 0
    assert stats["final_by_split"]["test"] > 0
    assert stats["leakage"] == {"entity_overlap_train_dev": [], "entity_overlap_train_test": [], "group_overlap": []}
    assert {"ru", "en", "he"} <= set(stats["final_by_lang"])

    # 4. Duplicate IDs across base and hebrew must fail loudly
    with pytest.raises(SystemExit, match="duplicate record id"):
        GH.merge_datasets(he_dir, he_dir, tmp_path / "dup_out", check_render=False, run_recheck=False)
