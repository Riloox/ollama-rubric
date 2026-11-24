"""Plain asserts, no runner. `python3 tests.py` — exits non-zero on the first failure."""

import json
import tempfile
from pathlib import Path

import scoring
import ollama_http
import rubric

FAIL = []


def eq(name, got, want):
    if got != want:
        FAIL.append(f"{name}: got {got!r}, want {want!r}")


# contains_any: accents matter, case does not, whitespace is collapsed
eq("contains/accents", scoring.check_contains_any(
    "**고친 문장:** Ella compró un libro nuevo.", {"values": ["compró un libro nuevo"]})[0], True)
eq("contains/case", scoring.check_contains_any(
    "ELLA COMPRÓ UN LIBRO NUEVO", {"values": ["compró un libro nuevo"]})[0], True)
eq("contains/missing-accent", scoring.check_contains_any(
    "Ella compro un libro nuevo", {"values": ["compró un libro nuevo"]})[0], False)
eq("contains/spaces", scoring.check_contains_any(
    "Ella  compró   un libro", {"values": ["compró un libro"]})[0], True)
eq("contains/empty", scoring.check_contains_any("", {"values": ["x"]})[0], False)

# require_korean
eq("korean/yes", scoring.check_require_korean("맞게 쓰셨어요.")[0], True)
eq("korean/no", scoring.check_require_korean("That is correct.")[0], False)

# says_correct: a correct sentence that gets "fixed" is a failure
eq("says/confirms", scoring.check_says_correct(
    "**맞게 썼어요.** 그대로 두시면 됩니다.",
    {"values": ["Mañana voy a visitar el banco."]})[0], True)
eq("says/invents-fix", scoring.check_says_correct(
    "**고친 문장:** Meñana voy a visitar el banco.",
    {"values": ["Mañana voy a visitar el banco."]})[0], False)
eq("says/echoes-unchanged", scoring.check_says_correct(
    "맞아요!\n고친 문장: Mañana voy a visitar el banco.",
    {"values": ["Mañana voy a visitar el banco."]})[0], True)
# regressions: the first version of these checks was far too lenient on correct sentences
eq("says/echo-without-verdict", scoring.check_says_correct(
    "Mañana voy a visitar el banco. 맞는 것 같아요.",
    {"values": ["Mañana voy a visitar el banco."]})[0], False)
eq("says/verdict-then-change", scoring.check_says_correct(
    "맞아요!\n고친 문장: Mañana voy a visitar al banco.",
    {"values": ["Mañana voy a visitar el banco."]})[0], False)
eq("says/own-check-flags-it", scoring.check_says_correct(
    "쓴 형태: 'a un libro' ✗\n맞아요.",
    {"values": ["Ella compró a un libro."]})[0], False)
eq("contains/whole-word", scoring.check_contains_any("tú eras amable", {"values": ["era"]})[0], False)
eq("contains/markdown", scoring.check_contains_any(
    "Ella compró **un** libro nuevo", {"values": ["compró un libro nuevo"]})[0], True)
corr = {"id": "c", "category": "correction", "prompt": '"Ella compró a un libro nuevo." 자연스러워요?',
        "check": {"type": "contains_any", "values": ["compró un libro nuevo"]}}
eq("score/says-fine-but-fix-in-exercise", scoring.score(
    corr, "맞아요! 틀린 곳이 없어요.\n연습: Ella compró un libro nuevo.")["passed"], False)
eq("score/real-fix", scoring.score(
    corr, "고친 문장: Ella compró un libro nuevo.\n'a un' → 'un'")["passed"], True)
# rewrite-first answers that confirm in the honorific form (t4-ok-028, -068, -081 in the 20 Nov run)
eq("says/rewrite-first-honorific", scoring.check_says_correct(
    "**다시 써 보면:** Mañana voy a visitar el banco.\n**바뀐 곳:** 없음\n\n네, 아주 잘 쓰셨어요! 👍",
    {"values": ["Mañana voy a visitar el banco."]})[0], True)
eq("says/honorific-but-rewritten", scoring.check_says_correct(
    "다시 써 보면: Mañana voy al banco.\n바뀐 곳: 'visitar el' → 'al'\n\n잘 쓰셨어요!",
    {"values": ["Mañana voy a visitar el banco."]})[0], False)
eq("says/honorific-after-no", scoring.check_says_correct(
    "아니요, 자연스럽지 않아요.\n\n다시 써 보면: Mañana voy a visitar el banco.\n바뀐 곳: 없음",
    {"values": ["Mañana voy a visitar el banco."]})[0], False)
eq("says/silent", scoring.check_says_correct(
    "Interesting question.", {"values": ["Mañana voy a visitar el banco."]})[0], False)

# score(): all declared checks must pass; rubric is ungraded by default, fatal in strict
item = {"id": "x", "check": {"type": "contains_any", "values": ["compró"]},
        "require_korean": True, "rubric": "say it is correct"}
eq("score/pass", scoring.score(item, "고친 문장: compró")["passed"], True)
eq("score/fail-korean", scoring.score(item, "fixed: compró")["passed"], False)
eq("score/rubric-skipped", scoring.score(item, "고친 문장: compró")["needs_judge"], True)
eq("score/rubric-strict", scoring.score(item, "고친 문장: compró", "fail")["passed"], False)
eq("score/unknown-check", scoring.score({"check": {"type": "nope"}}, "x")["passed"], False)

# environment(): version and digests are recorded, and a failed lookup is None, not an exception
class Stub(ollama_http.Client):
    def _get(self, path):
        if path == "/api/version":
            return {"version": "0.0.1"}
        return {"models": [{"name": "m:latest", "digest": "abc123"}]}

env = Stub().environment(["m:latest", "other"])
eq("env/version", env["ollama_version"], "0.0.1")
eq("env/digest", env["model_digests"]["m:latest"], "abc123")
eq("env/missing-model", env["model_digests"]["other"], None)

class Down(ollama_http.Client):
    def _get(self, path):
        raise OSError("down")

eq("env/down", Down().environment(["m"]), {"ollama_version": None, "model_digests": {"m": None}})

# Client.generate(..., fmt=...) sends format in payload
class CaptureClient(ollama_http.Client):
    def __init__(self):
        super().__init__()
        self.calls = []

    def _post(self, path, payload):
        self.calls.append((path, payload))
        return {"response": '{"pass": true, "reason": "confirmed"}'}

cap = CaptureClient()
ans, _ = cap.generate("judge-m", "hi", system="sys", think=False, fmt=scoring.JUDGE_SCHEMA)
eq("client/fmt-response", ans, '{"pass": true, "reason": "confirmed"}')
eq("client/fmt-payload", cap.calls[0][1]["format"], scoring.JUDGE_SCHEMA)
eq("client/think-payload", cap.calls[0][1]["think"], False)

# parse_judge_response: plain JSON, fenced JSON, malformed
eq("judge-parse/plain", scoring.parse_judge_response('{"pass": true, "reason": "kept sentence"}'),
   (True, "kept sentence"))
eq("judge-parse/fenced", scoring.parse_judge_response('```json\n{"pass": false, "reason": "invented fix"}\n```'),
   (False, "invented fix"))
bad_ok, bad_why = scoring.parse_judge_response("not json at all")
eq("judge-parse/bad-ok", bad_ok, False)
eq("judge-parse/bad-prefix", bad_why.startswith("judge parse error:"), True)

# judge(): blind by default, includes fix only with show_fix=True
jcap = CaptureClient()
j_item = {"id": "j1", "prompt": "P", "fix": "SECRET_FIX"}
j_ok, j_why = scoring.judge("model answer", "rubric rule", j_item, client=jcap, model="gemma4:26b")
eq("judge/call-ok", (j_ok, j_why), (True, "confirmed"))
eq("judge/blind-default", "SECRET_FIX" in jcap.calls[0][1]["prompt"], False)
scoring.judge("model answer", "rubric rule", j_item, client=jcap, model="gemma4:26b", show_fix=True)
eq("judge/show-fix", "SECRET_FIX" in jcap.calls[1][1]["prompt"], True)

# score(rubric_mode="judge") and disagreement classification
ok_item = {
    "id": "t4-ok-1", "category": "correction_ok",
    "check": {"type": "says_correct", "values": ["Voy al banco."]},
    "require_korean": True, "rubric": "Sentence is already correct.",
}
res_agree = scoring.score(ok_item, "맞아요. 그대로 두세요.", rubric_mode="judge",
                          judge_fn=lambda a, r, i: (True, "confirmed"))
eq("score/judge-agree-pass", (res_agree["passed"], res_agree["needs_judge"], res_agree["disagreement"]),
   (True, False, None))

res_mp_jf = scoring.score(ok_item, "맞아요. 그대로 두세요.", rubric_mode="judge",
                          judge_fn=lambda a, r, i: (False, "false positive"))
eq("score/dis-mp-jf", (res_mp_jf["passed"], res_mp_jf["disagreement"]),
   (False, "mech_pass_judge_fail"))

res_mf_jp = scoring.score(ok_item, "Voy al banco. 좋은 문장이에요.", rubric_mode="judge",
                          judge_fn=lambda a, r, i: (True, "actually confirmed"))
eq("score/dis-mf-jp", (res_mf_jp["passed"], res_mf_jp["disagreement"]),
   (False, "mech_fail_judge_pass"))

# cohen_kappa()
eq("kappa/empty", scoring.cohen_kappa([])["n"], 0)
eq("kappa/perfect", scoring.cohen_kappa([(True, True), (False, False)])["kappa"], 1.0)
k_mid = scoring.cohen_kappa([(True, True)] * 4 + [(False, False)] * 4 + [(True, False), (False, True)])
eq("kappa/mid-agree", round(k_mid["agreement"], 2), 0.80)
eq("kappa/mid-val", round(k_mid["kappa"], 2), 0.60)

# select_calibration_sample() + calibrate --labels
fake_src = {
    "results": {
        "m1": [
            {"id": "a1", "prompt": "p1", "answer": "a1", "checks": [
                {"check": "says_correct", "passed": True, "reason": "ok"},
                {"check": "rubric", "passed": False, "reason": "dis"}]},
            {"id": "a2", "prompt": "p2", "answer": "a2", "checks": [
                {"check": "says_correct", "passed": True, "reason": "ok"},
                {"check": "rubric", "passed": True, "reason": "ok"}]},
            {"id": "a3", "prompt": "p3", "answer": "a3", "checks": [
                {"check": "says_correct", "passed": False, "reason": "no"},
                {"check": "rubric", "passed": False, "reason": "no"}]},
        ],
        "m2": [
            {"id": "a1", "prompt": "p1", "answer": "b1", "checks": [
                {"check": "says_correct", "passed": False, "reason": "no"},
                {"check": "rubric", "passed": True, "reason": "dis"}]},
            {"id": "a2", "prompt": "p2", "answer": "b2", "checks": [
                {"check": "says_correct", "passed": True, "reason": "ok"},
                {"check": "rubric", "passed": True, "reason": "ok"}]},
        ],
    }
}
sample = rubric.select_calibration_sample(fake_src, n=4, seed=7)
eq("cal/sample-len", len(sample), 4)
eq("cal/sample-disagreements", sum(1 for r in sample if r["disagreement"] is not None), 2)

with tempfile.TemporaryDirectory() as tmp:
    rpath = Path(tmp) / "res.json"
    lpath = Path(tmp) / "labels.jsonl"
    rpath.write_text(json.dumps(fake_src), encoding="utf-8")
    lpath.write_text("\n".join([
        json.dumps({"model": "m1", "id": "a1", "human_pass": False}),
        json.dumps({"model": "m1", "id": "a2", "human_pass": True}),
        json.dumps({"model": "m1", "id": "a3", "human_pass": False}),
        json.dumps({"model": "m2", "id": "a1", "human_pass": True}),
    ]) + "\n", encoding="utf-8")
    lookup = {(r["model"], r["id"]): r for r in rubric._judged_records(fake_src)}
    rows = [json.loads(l) for l in lpath.read_text(encoding="utf-8").splitlines()]
    joined = [(r["human_pass"], lookup[(r["model"], r["id"])]) for r in rows]
    eq("cal/human-judge-kappa", scoring.cohen_kappa([(hp, rec["judge_pass"]) for hp, rec in joined])["kappa"], 1.0)

if FAIL:
    print("\n".join("FAIL " + f for f in FAIL))
    raise SystemExit(1)
print("ok — all checks")
