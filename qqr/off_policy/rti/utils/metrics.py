from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ..ref.ref_scorer import RefScorer

def _softmax_from_logps(logps, temperature=1.0):
    x = np.array([lp if lp is not None else -1e9 for lp in logps], dtype=np.float64) / max(temperature, 1e-6)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / (np.sum(ex) + 1e-12)

def _kl(p, q):
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    return float(np.sum(p * (np.log(p) - np.log(q))))

def _js(p, q):
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)

def _ppl(token_logprobs):
    v = [lp for lp in (token_logprobs or []) if lp is not None and np.isfinite(lp)]
    if not v:
        return float("inf")
    avg_nll = -float(np.mean(v))
    return float(np.exp(avg_nll))

async def ref_kl_gate(
    *,
    args,
    sample_messages: List[Dict[str, Any]],
    original_response: str,
    corrected_response: str,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[bool, Dict[str, Any]]:
    if not getattr(args, "enable_ref_kl_gate", False):
        return True, {"enabled": False, "ok": True, "reason": "disabled"}

    method_str = (getattr(args, "ref_judge_method", "") or "ppl,js").strip().lower()
    methods = [m.strip() for m in method_str.split(",") if m.strip()]
    methods = [m for m in methods if m in {"ppl", "kl", "js"}]
    if not methods:
        return True, {"enabled": True, "ok": True, "reason": "empty_methods_allow"}

    ref_scorer = RefScorer(args)
    raw_score = await ref_scorer.score_completion_async(sample_messages, original_response, tools=tools)
    corr_score = await ref_scorer.score_completion_async(sample_messages, corrected_response, tools=tools)

    raw_ppl = _ppl(raw_score.get("token_logprobs"))
    corr_ppl = _ppl(corr_score.get("token_logprobs"))
    ppl_pass = corr_ppl < float(getattr(args, "ref_ppl_threshold", 2.0))

    L = min(len(raw_score.get("token_logprobs") or []), len(corr_score.get("token_logprobs") or []))
    raw_lps = (raw_score.get("token_logprobs") or [])[:L]
    corr_lps = (corr_score.get("token_logprobs") or [])[:L]

    kl_val = js_val = None
    kl_pass = js_pass = True
    if ("kl" in methods) or ("js" in methods):
        temp = float(getattr(args, "ref_kl_temperature", 1.0))
        p = _softmax_from_logps(raw_lps, temperature=temp)
        q = _softmax_from_logps(corr_lps, temperature=temp)
        kl_val = _kl(p, q)
        js_val = _js(p, q)
        kl_pass = (kl_val <= float(getattr(args, "ref_kl_threshold", 1.0)))
        js_pass = (js_val <= float(getattr(args, "ref_js_threshold", 0.3)))

    pass_map = {"ppl": ppl_pass, "kl": kl_pass, "js": js_pass}
    ok = all(pass_map[m] for m in methods)

    return ok, {
        "enabled": True,
        "ok": ok,
        "methods": methods,
        "common_prefix_len": L,
        "score":{
            "raw_score": raw_score, 
            "corr_score": corr_score
        },
        "values": {"raw_ppl": raw_ppl, "corr_ppl": corr_ppl,  "kl": kl_val, "js": js_val},
        "passes": {m: pass_map[m] for m in methods},
    }

async def get_completion_score(
    *,
    args,
    sample_messages: List[Dict[str, Any]],
    response: str,
    tools: Optional[List[Dict[str, Any]]] = None,
    url=None,
    model_id=None,
):
    ref_scorer = RefScorer(args, url=url, model_id=model_id)
    raw_score = await ref_scorer.score_completion_async(sample_messages, response, tools=tools)
    return raw_score
