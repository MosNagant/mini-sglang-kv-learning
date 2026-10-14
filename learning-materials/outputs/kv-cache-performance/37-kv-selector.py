"""Coach-written, standard-library-only frozen selection policy for M07-37.

This selects an entry name from already validated metadata. It does not validate
GPU index values, launch GPU work, import Torch/Triton, or time the selector.
Unknown correctness/environment/source context is rejected, never guessed.
Explicit research requests are not automatic performance promotion.
"""
import hashlib
import json
import math
from pathlib import Path

POLICY_CANONICAL_SHA256 = "8a6f451a0abd47280e7139c3181bdb081504bad0e68ea28c2bb282f63c0028cd"


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def fingerprint(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def meets_promotion(original_us, candidate_us):
    require(len(original_us) == len(candidate_us) == 3, "THREE_PAIRED_ROUNDS_REQUIRED")
    require(all(type(x) in (int, float) and math.isfinite(x) and x > 0
                for x in (*original_us, *candidate_us)), "INVALID_HISTORICAL_LATENCY")
    return all(candidate <= .97 * original
               for original, candidate in zip(original_us, candidate_us))


def validate_policy(policy):
    require(fingerprint(policy) == POLICY_CANONICAL_SHA256, "FROZEN_POLICY_CHANGED")
    require(policy["promoted_candidates"] == [] and policy["default"] == "original_cuda",
            "NO_CANDIDATE_PROMOTION_ACCEPTED")
    for row in policy["screening"]:
        require(not meets_promotion(row["original_us"], row["arm_us"]),
                "HISTORICAL_SCREENING_CHANGED")
    return policy


def load_policy(path=None):
    path = Path(path) if path else Path(__file__).with_name("37-kv-selection-policy.json")
    return validate_policy(json.loads(path.read_text(encoding="utf-8")))


def context_for_config(policy, config, execution_policy, environment=None):
    """Build a context for an already validated fixture; CPU tests may use it.

GPU callers must supply their observed environment and validate all tensors
before using this function. This helper itself is not tensor validation.
"""
    d, b, c = config["width"], config["batch"], config["capacity"]
    src = d + config["gap"]
    dst = (d + config["guard"]) * (2 if config["layout"] == "interleaved" else 1)
    return dict(environment=dict(environment or policy["environment"]),
                execution_policy=execution_policy, address_policy="reused_addresses_no_flush",
                dtype="float16", index_dtype="int32", device="cuda:0",
                batch=b, width=d, capacity=c, layout=config["layout"],
                input_gap=config["gap"], cache_guard=config["guard"],
                k_shape=[b, d], v_shape=[b, d], k_cache_shape=[c, d], v_cache_shape=[c, d],
                indices_shape=[b], indices_stride=[1],
                k_stride=[src, 1], v_stride=[src, 1],
                k_cache_stride=[dst, 1], v_cache_stride=[dst, 1],
                metadata_and_aliases_validated=True, unique_in_range_indices_validated=True,
                source_hashes=dict(policy["source_pins"]))


def validate_context(policy, ctx):
    require(ctx.get("environment") == policy["environment"], "UNKNOWN_ENVIRONMENT")
    require(ctx.get("execution_policy") in ("graph", "eager"), "UNKNOWN_EXECUTION_POLICY")
    require(ctx.get("address_policy") == "reused_addresses_no_flush", "UNKNOWN_ADDRESS_POLICY")
    require(ctx.get("source_hashes") == policy["source_pins"], "SOURCE_VERSION_CHANGED")
    require(ctx.get("dtype") == "float16" and ctx.get("index_dtype") == "int32", "UNSUPPORTED_DTYPE")
    require(ctx.get("device") == "cuda:0", "UNKNOWN_DEVICE")
    require(ctx.get("metadata_and_aliases_validated") is True and
            ctx.get("unique_in_range_indices_validated") is True, "VALIDATION_REQUIRED")
    integer_fields = ("batch", "width", "capacity", "input_gap", "cache_guard")
    require(all(type(ctx.get(k)) is int for k in integer_fields), "INVALID_INTEGER_METADATA")
    b, d, c = (ctx[k] for k in ("batch", "width", "capacity"))
    require(c in (512, 1024) and 1 <= b <= c, "BATCH_CAPACITY_SCOPE")
    require(d in (64, 128, 192), "WIDTH_SCOPE")
    require(ctx.get("layout") in ("interleaved", "separate"), "UNKNOWN_LAYOUT")
    require(ctx["input_gap"] in (0, 64), "INPUT_GAP_SCOPE")
    require(ctx["cache_guard"] == 0 or (d == 192 and ctx["cache_guard"] == 64), "CACHE_GUARD_SCOPE")
    cfg = dict(batch=b, width=d, capacity=c, layout=ctx["layout"],
               gap=ctx["input_gap"], guard=ctx["cache_guard"])
    expected = context_for_config(policy, cfg, ctx["execution_policy"], ctx["environment"])
    require(ctx == expected, "INCOMPLETE_OR_INCONSISTENT_CONTEXT")
    return cfg


def select(policy, context, request="auto"):
    validate_policy(policy)
    validate_context(policy, context)
    require(request == "auto" or request in policy["research_requests"], "UNKNOWN_REQUEST")
    research = request != "auto"
    return dict(policy_id=policy["id"], policy_fingerprint=POLICY_CANONICAL_SHA256,
                context_fingerprint=fingerprint(context), request=request,
                selected=request if research else "original_cuda",
                reason="explicit_research_not_promoted" if research else "no_historical_candidate_passed_frozen_gate",
                automatic_promotion=False, performance_claim=False,
                source=context["source_hashes"][policy["research_requests"][request]] if research
                else context["source_hashes"]["original_source/python/minisgl/kernel/store.py"])
