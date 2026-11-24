"""Check implementations. This is the interesting half of the demo.

Each check answers one question about a model's answer and returns
(passed, reason). Keep them pure: no network, no I/O, so they are testable and
so a task file can be re-scored without re-running the model.

The mechanical checks are a port of the grader in gemma-spanish-tutor-wip/eval/run_eval.py,
the one behind the published 123/208 (stock gemma4:e2b) and 177/208 (v4) numbers. An earlier
version here was far more lenient on already-correct sentences (7/7 where this one gives 2/7
on the same answers), so numbers from it could not have reproduced those. tests.py pins the
cases that differed.

Shapes in eval/test_set_v4.jsonl:
  {"type": "contains_any", "values": ["compró un libro nuevo", ...]}
  {"type": "says_correct", "values": ["<the learner's sentence>"]}
  require_korean: true
  rubric: "<grading instruction>"   <- no implementation yet, see judge() below
"""

from __future__ import annotations

import json
import re
import unicodedata

LATIN_CHAR = r"[A-Za-zÀ-ÿ]"
LATIN = re.compile(LATIN_CHAR)
HANGUL = re.compile(r"[가-힣]")
ENGLISH_WORDS = re.compile(r"\b(the|is|are|and|you|this|that|with|verb|sentence)\b", re.IGNORECASE)


def norm(s: str) -> str:
    """NFC, markdown emphasis dropped, lowercased, whitespace collapsed. Accents are kept:
    in Spanish they are the answer."""
    s = unicodedata.normalize("NFC", s or "").replace("*", "").replace("`", "")
    return re.sub(r"\s+", " ", s).lower().strip()


def contains_phrase(text: str, phrase: str) -> bool:
    """Whole-word, accent-sensitive. "dé" must not match "de", "era" must not match "eras".
    Boundaries are Latin letters only, not \\w: Korean endings attach directly to the answer
    ("hable예요"), and \\w would count those Hangul syllables as part of the word."""
    p = norm(phrase)
    if not p:
        return False
    return re.search(rf"(?<!{LATIN_CHAR}){re.escape(p)}(?!{LATIN_CHAR})", norm(text)) is not None


# ---------------------------------------------------------------- verdict on already-correct sentences
# Grades the verdict, for items whose check values are [the learner's sentence]:
#   - the answer's own check block, if it opens with one ("쓴 형태: '…' ✗/✓"), doesn't flag the form;
#     a rewrite-first block ("다시 써 보면: …" / "바뀐 곳: …") counts as a check block too;
#   - it says the sentence is right: an affirmation in the first line after the check block, or
#     "틀린/고칠 곳이 없어요" anywhere, and no hedge ("거의 다 맞았어요", "고쳐 볼게요") in that line;
#   - nothing is changed: every "고친 문장:" / "다시 써 보면:" line is the learner's sentence, and
#     no 'x' → 'y' with x != y.
# Filling the template with the unchanged sentence is fine; inventing a fix is not.
# "맞게 쓰셨" and "잘 쓰셨" are not in run_eval.py's list. The honorific forms are how v4 confirms
# ("네, 맞게 쓰셨어요!", "잘 쓰셨어요! 👍"), and without them 3 of its confirmations failed in the
# 20 Nov run. Scores from this grader are one notch above run_eval.py's on such answers.
AFFIRM = ["맞아요", "맞습니다", "맞는 문장", "맞고", "맞게 썼", "맞게 쓰셨", "잘 쓰셨", "완벽",
          "자연스러운 문장", "자연스러워요"]
NO_ERROR = re.compile(r"(틀린|고칠)\s*(곳|부분|데)[은이가]?\s*(하나도\s*|전혀\s*)?없")
DENY = ["거의", "고쳐", "고치면", "틀렸", "안 맞", "맞지 않", "어색"]
FIX_LINE = re.compile(r"고친 (?:문장|글)\s*:\s*(.+)")
CHECK_LINE = re.compile(r"(확인할 점|쓴 형태|다시 써 보면|바뀐 곳)\s*:")
REWRITE_LINE = re.compile(r"다시 써 보면\s*:\s*(.+)")
TRAILING_GLOSS = re.compile(r"\s*\([^()]*[가-힣][^()]*\)\s*$")  # "Tengo una mano. (저는 손이…)" -> Spanish only
USED_LINE = re.compile(r"쓴 형태\s*:(.*)")
ARROW = re.compile(r"'([^'\n]+)'\s*(?:→|\$?\\rightarrow\$?)\s*'([^'\n]+)'")


def sentence_key(s: str) -> str:
    return re.sub(r"[\"'“”«».,!?¿¡\s]", "", norm(s))


def verdict_line(plain: str) -> str:
    """The first line after the check block that check-first and rewrite-first answers open with."""
    for line in plain.strip().split("\n"):
        if line.strip() and not CHECK_LINE.match(line.strip()):
            return line
    return ""


def rewrites(plain: str) -> list[str]:
    """The sentences on "다시 써 보면:" lines, Korean gloss dropped."""
    return [TRAILING_GLOSS.sub("", r).strip() for r in REWRITE_LINE.findall(plain)]


def affirms(answer: str) -> bool:
    """The verdict says the sentence is right, whatever the answer then changes in it."""
    plain = answer.replace("*", "")
    if any("✗" in m for m in USED_LINE.findall(plain)):
        return False  # its own check flagged the form, whatever the verdict line says
    first = norm(verdict_line(plain))
    yes = any(a in first for a in AFFIRM) or NO_ERROR.search(norm(answer))
    return bool(yes) and not any(d in first for d in DENY)


def says_correct(answer: str, sentence: str) -> bool:
    plain = answer.replace("*", "")
    if not affirms(answer):
        return False
    if any(sentence_key(f) != sentence_key(sentence) for f in FIX_LINE.findall(plain) + rewrites(plain)):
        return False
    return all(a.strip() == b.strip() for a, b in ARROW.findall(plain))


def learner_sentence(item: dict) -> str | None:
    """The one quoted sentence a `correction` item asks about, or None (paragraphs, other categories)."""
    quoted = re.findall(r'"([^"]+)"', item.get("prompt") or "")
    return quoted[0] if item.get("category") == "correction" and len(quoted) == 1 else None


# ---------------------------------------------------------------- checks

def check_says_correct(answer: str, spec: dict, sentence: str | None = None) -> tuple[bool, str]:
    """The sentence is already right and the tutor must say so, not invent an error."""
    values = spec.get("values") or []
    if not values:
        return False, "no sentence to check against"
    if says_correct(answer, values[0]):
        return True, "confirmed as correct, nothing changed"
    return False, "did not confirm it, or changed something in a correct sentence"


def _hits(answer: str, spec: dict) -> list[str]:
    return [v for v in spec.get("values") or [] if contains_phrase(answer, v)]


def _says_learner_is_right(answer: str, sentence: str | None) -> bool:
    return bool(sentence) and says_correct(answer, sentence)


def check_contains_any(answer: str, spec: dict, sentence: str | None = None) -> tuple[bool, str]:
    values = spec.get("values") or []
    if _says_learner_is_right(answer, sentence):
        # "틀린 곳이 없어요" must not pass because the fixed form shows up in a practice exercise
        return False, "said the learner's sentence was already correct"
    hits = _hits(answer, spec)
    if hits:
        return True, f"contains {hits[0]!r}"
    return False, f"missing any of {values!r}"


def check_contains_all(answer: str, spec: dict, sentence: str | None = None) -> tuple[bool, str]:
    values = spec.get("values") or []
    if _says_learner_is_right(answer, sentence):
        return False, "said the learner's sentence was already correct"
    missing = [v for v in values if not contains_phrase(answer, v)]
    return (not missing and bool(values)), ("contains all" if not missing else f"missing {missing!r}")


def check_not_contains(answer: str, spec: dict, sentence: str | None = None) -> tuple[bool, str]:
    """For items where the wrong form must disappear, not just the right one appear."""
    for v in spec.get("values") or []:
        if contains_phrase(answer, v):
            return False, f"still contains {v!r}"
    return True, "no forbidden strings"


def check_require_korean(answer: str, _spec: dict | None = None) -> tuple[bool, str]:
    """Enough Hangul relative to Latin letters, and not English prose. Answers legitimately
    contain Spanish examples, so 100% Hangul is not required."""
    answer = answer or ""
    hangul, latin = len(HANGUL.findall(answer)), len(LATIN.findall(answer))
    ratio = hangul / max(hangul + latin, 1)
    if ratio >= 0.25 and len(ENGLISH_WORDS.findall(answer)) <= 3:
        return True, "answer is in Korean"
    return False, f"not Korean enough (hangul share {ratio:.2f})"


CHECKS = {
    "contains_any": check_contains_any,
    "contains_all": check_contains_all,
    "not_contains": check_not_contains,
    "says_correct": check_says_correct,
}


JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "pass": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["pass", "reason"],
}

JUDGE_SYSTEM = (
    "You are a strict rubric grader for a Spanish tutor for Korean learners. "
    "Grade the model answer strictly against the rubric instruction and user prompt. "
    "Return JSON only: {\"pass\": true or false, \"reason\": \"one short sentence\"}."
)

_CODE_FENCE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL | re.IGNORECASE)


def judge_prompt(item: dict, answer: str, instruction: str, show_fix: bool = False) -> str:
    """Build the judge prompt. Blind to item['fix'] unless show_fix=True."""
    parts = [
        f"Prompt:\n{item.get('prompt', '')}",
        f"Rubric:\n{instruction}",
    ]
    if show_fix and item.get("fix"):
        parts.append(f"Reference fix:\n{item['fix']}")
    parts.append(f"Model answer:\n{answer}")
    return "\n\n".join(parts)


def parse_judge_response(raw: str) -> tuple[bool, str]:
    """Parse JSON {'pass': bool, 'reason': str}, tolerating code fences."""
    text = (raw or "").strip()
    m = _CODE_FENCE.match(text)
    if m:
        text = m.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return False, f"judge parse error: {(raw or '')[:120]}"
    if not isinstance(data, dict) or not isinstance(data.get("pass"), bool) or not isinstance(data.get("reason"), str):
        return False, f"judge parse error: {(raw or '')[:120]}"
    reason = data["reason"].strip() or ("passed" if data["pass"] else "failed")
    return data["pass"], reason


def judge(answer: str, instruction: str, item: dict, *, client=None, model: str = "",
          think: bool | None = False, show_fix: bool = False) -> tuple[bool, str]:
    """Grade a model answer against a rubric instruction using an Ollama model."""
    if client is None or not model:
        raise ValueError("judge() requires client and model")
    prompt = judge_prompt(item, answer, instruction, show_fix=show_fix)
    raw, _ = client.generate(
        model, prompt, temperature=0.0, system=JUDGE_SYSTEM, think=think, fmt=JUDGE_SCHEMA)
    return parse_judge_response(raw)


def disagreement(res: dict) -> str | None:
    """Classify disagreement between mechanical checks and the rubric judge."""
    checks = res.get("checks") or []
    rubric_checks = [c for c in checks if c.get("check") == "rubric"]
    mech_checks = [c for c in checks if c.get("check") != "rubric"]
    if not rubric_checks or not mech_checks:
        return None
    mech_ok = all(c.get("passed") for c in mech_checks)
    judge_ok = all(c.get("passed") for c in rubric_checks)
    if mech_ok and not judge_ok:
        return "mech_pass_judge_fail"
    if not mech_ok and judge_ok:
        return "mech_fail_judge_pass"
    return None


def cohen_kappa(pairs: list[tuple[bool, bool]]) -> dict:
    """Compute agreement and Cohen's kappa for binary (a, b) verdict pairs."""
    n = len(pairs)
    tp = sum(1 for a, b in pairs if a and b)
    tn = sum(1 for a, b in pairs if not a and not b)
    fp = sum(1 for a, b in pairs if a and not b)
    fn = sum(1 for a, b in pairs if not a and b)
    agree = tp + tn
    if n == 0:
        return {"n": 0, "tp": 0, "tn": 0, "fp": 0, "fn": 0,
                "agree": 0, "agreement": 0.0, "p_e": 0.0, "kappa": 0.0}
    p_o = agree / n
    p_a_yes = (tp + fp) / n
    p_b_yes = (tp + fn) / n
    p_e = p_a_yes * p_b_yes + (1.0 - p_a_yes) * (1.0 - p_b_yes)
    if abs(1.0 - p_e) < 1e-12:
        kappa = 1.0 if p_o == 1.0 else 0.0
    else:
        kappa = (p_o - p_e) / (1.0 - p_e)
    return {
        "n": n, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "agree": agree, "agreement": p_o, "p_e": p_e, "kappa": kappa,
    }


def score(item: dict, answer: str, rubric_mode: str = "skip", judge_fn=None) -> dict:
    """Apply every check the item declares. All must pass.

    rubric_mode:
      skip   (default) rubric-only items are recorded as ungraded, not failed
      fail   items carrying a rubric are failed until judged — the strict run
      judge  call judge_fn(answer, rubric, item) to grade rubric items
    """
    if rubric_mode not in ("skip", "fail", "judge"):
        raise ValueError(f"unknown rubric_mode: {rubric_mode!r}")

    results = []
    check = item.get("check") or {}
    ctype = check.get("type")
    if ctype:
        fn = CHECKS.get(ctype)
        if fn is None:
            results.append({"check": ctype, "passed": False, "reason": "unknown check type"})
        else:
            ok, why = fn(answer, check, learner_sentence(item))
            results.append({"check": ctype, "passed": ok, "reason": why})

    if item.get("require_korean"):
        ok, why = check_require_korean(answer)
        results.append({"check": "require_korean", "passed": ok, "reason": why})

    ungraded = False
    if item.get("rubric"):
        if rubric_mode == "fail":
            results.append({"check": "rubric", "passed": False, "reason": "judge not implemented"})
        elif rubric_mode == "judge":
            if judge_fn is None:
                raise ValueError("judge_fn is required when rubric_mode='judge'")
            ok, why = judge_fn(answer, item["rubric"], item)
            results.append({"check": "rubric", "passed": ok, "reason": why})
        else:
            ungraded = True

    out = {
        "id": item.get("id"),
        "category": item.get("category"),
        "passed": all(r["passed"] for r in results) if results else False,
        "needs_judge": ungraded,
        "checks": results,
    }
    out["disagreement"] = disagreement(out)
    return out
