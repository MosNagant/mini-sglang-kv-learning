"""Experimental B1 greedy sampling inside the original decode CUDA Graph.

Opt-in only. Install BEFORE constructing one Engine with cuda_graph_bs=[1],
TP=1, and overlap scheduling disabled. No original minisgl file is edited.
The original model, FP32 Graph logits buffer, torch.argmax(dim=-1), and int32
conversion are preserved. The captured argmax/cast tensor is retained by the
runner; replay overwrites it. There is no extra logits copy or token copy.

Only a one-use ticket issued by the immediately preceding actual replay can
be consumed by Sampler.sample. Shape, stride, dtype, device, data pointer,
sampler configuration, and current stream must match. Other sampling calls use
the original implementation and record why. Intended decode replay misses are
not silently accepted: assert_usage() requires all expected replay tickets to
have been consumed. Engine's existing .to(int32) then performs no conversion.

Source hashes pin graph.py, engine.py, and sample.py. The capture method is
copied from that verified source and receives exactly two AST insertions:
one after the original warmup logits assignment and one after its captured
counterpart. No model arithmetic, existing logits assignment, or sampler RNG
semantics are rewritten. Non-greedy Graph replay computes an unused argmax,
then retains original sampling; it is outside the performance claim scope.

Lifetime: one engine, one stream, sequential forwards, no concurrent calls.
Keep the handle alive through model shutdown. Uninstall restores class methods
but does not rebuild already captured graphs: use a fresh model/process for an
original baseline. Engineering acceptance and speedup require external tests.
Coach implementation; learner mastery is not inferred.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import inspect
from pathlib import Path


SOURCE_SHA256 = {
    'graph.py': '2fc80c2d4debf56ccec70850dcc1060afa52da631f90f8172d6c87d5ca4b4115',
    'engine.py': '82bd58030c78dbfe249eaef857fbd48db9cb403e179dc1f64ade89cb3f8c699f',
    'sample.py': '6bb15e116a299929af9436df2c3d2e8331a0aa0adc63320cfe86e0993370ed0b',
}
_ACTIVE = None


def _require(condition, message):
    if not condition:
        raise RuntimeError('GRAPH_GREEDY: ' + message)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _signature(tensor):
    """CPU metadata only; deliberately does not inspect device tensor values."""
    return (tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype),
            str(tensor.device), int(tensor.data_ptr()), int(tensor.storage_offset()))


def _reuse_reason(ticket, logits, args, sampler, current_stream):
    """Pure metadata predicate, also exercised by the CPU controls."""
    if ticket is None:
        return 'no_replay_ticket'
    if args.temperatures is not None:
        return 'non_greedy'
    if _signature(logits) != ticket['signature']:
        return 'logits_metadata_mismatch'
    if tuple(logits.shape) != (1, sampler.vocab_size):
        return 'sampler_shape_mismatch'
    if str(logits.device) != str(sampler.device):
        return 'sampler_device_mismatch'
    if str(logits.dtype) != 'torch.float32':
        return 'logits_not_original_fp32_buffer'
    if current_stream != ticket['stream']:
        return 'stream_mismatch'
    expected = ((1,), (1,), 'torch.int32', str(logits.device))
    if _signature(ticket['tokens'])[:4] != expected:
        return 'captured_token_metadata_mismatch'
    if _signature(ticket['tokens']) != ticket['token_signature']:
        return 'captured_token_storage_changed'
    return None


def _transformed_capture(source):
    """Return a verified structural edit of _capture_graphs, never a new graph."""
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'GraphRunner')
    method = copy.deepcopy(next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                               and node.name == '_capture_graphs'))
    needle = ast.dump(ast.parse('self.buffer.logits[:bs] = model.forward()').body[0], include_attributes=False)

    class Insert(ast.NodeTransformer):
        count = 0

        def visit_Assign(self, node):
            if ast.dump(node, include_attributes=False) == needle:
                self.count += 1
                inserted = ast.parse('_graph_greedy_capture(self, bs)').body[0]
                return [node, ast.copy_location(inserted, node)]
            return node

    transform = Insert()
    method = transform.visit(method)
    _require(transform.count == 2, 'capture structure differs; expected exactly two logits assignments')
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), method], type_ignores=[])
    ast.fix_missing_locations(module)
    return module


class PatchHandle:
    """Owns reversible class patches and the routing evidence for one engine."""

    def __init__(self):
        import torch
        import minisgl.engine.graph as graph_module
        import minisgl.engine.sample as sample_module
        from minisgl.env import ENV

        _require(bool(ENV.DISABLE_OVERLAP_SCHEDULING), 'overlap scheduling must be disabled before minisgl import')
        source_root = Path(graph_module.__file__).resolve().parent
        _require(Path(sample_module.__file__).resolve().parent == source_root, 'mixed source roots')
        for name, digest in SOURCE_SHA256.items():
            _require(_sha(source_root / name) == digest, 'source hash differs: ' + name)
        runner_cls, sampler_cls = graph_module.GraphRunner, sample_module.Sampler
        originals = (runner_cls._capture_graphs, runner_cls.replay, sampler_cls.sample)
        for fn, filename, name in zip(originals, ('graph.py', 'graph.py', 'sample.py'),
                                      ('_capture_graphs', 'replay', 'sample')):
            raw = inspect.unwrap(fn)
            _require(raw.__name__ == name and Path(raw.__code__.co_filename).resolve() == source_root / filename,
                     'already modified runtime method: ' + name)
        self.torch, self.runner_cls, self.sampler_cls = torch, runner_cls, sampler_cls
        self.originals = originals
        self.runner = None
        self.pending = None
        self.installed = False
        self.source_root = str(source_root)
        self.counts = dict(capture_warmup_argmax=0, captured_argmax=0, replays=0,
                           sample_hits=0, original_samples=0, unconsumed_replay_tickets=0)
        self.fallback_reasons = {}

        def capture_argmax(runner, bs):
            _require(bs == 1, 'only B1 may be captured')
            _require(torch.cuda.current_stream(runner.device) == runner.stream, 'capture stream differs')
            logits = runner.buffer.logits[:bs]
            _require(logits.dtype == torch.float32 and logits.shape == (1, runner.buffer.logits.shape[1]),
                     'original FP32 logits buffer required')
            # These are precisely the original greedy operation and Engine cast.
            # The captured return tensor itself is retained: no copy_ is added.
            runner._graph_greedy_tokens = torch.argmax(logits, dim=-1).to(torch.int32)
            key = 'captured_argmax' if torch.cuda.is_current_stream_capturing() else 'capture_warmup_argmax'
            self.counts[key] += 1

        namespace = dict(vars(graph_module), _graph_greedy_capture=capture_argmax)
        transformed = _transformed_capture((source_root / 'graph.py').read_text(encoding='utf-8'))
        exec(compile(transformed, str(source_root / 'graph.py') + ':graph_greedy', 'exec'), namespace)
        edited_capture = namespace['_capture_graphs']

        def capture(runner, max_seq_len, vocab_size, model):
            _require(self.runner is None, 'one engine per installed handle')
            _require(runner.graph_bs_list == [1] and runner.max_graph_bs == 1, 'explicit cuda_graph_bs=[1] required')
            _require(graph_module.get_tp_info().size == 1, 'only TP1 is supported')
            self.runner = runner
            edited_capture(runner, max_seq_len, vocab_size, model)
            _require(self.counts['captured_argmax'] == 1 and self.counts['capture_warmup_argmax'] == 1,
                     'actual argmax capture was not observed exactly once')
            _require(_signature(runner._graph_greedy_tokens)[:4] == ((1,), (1,), 'torch.int32', str(runner.device)),
                     'captured token storage differs')

        def replay(runner, batch):
            _require(runner is self.runner, 'unregistered graph runner')
            _require(batch.is_decode and batch.size == batch.padded_size == 1, 'replay requires B1 decode')
            if self.pending is not None:
                self.counts['unconsumed_replay_tickets'] += 1
                self.pending = None
                raise RuntimeError('GRAPH_GREEDY: new replay before prior sample; sequential contract violated')
            _require(torch.cuda.current_stream(runner.device) == runner.stream, 'replay stream differs')
            logits = originals[1](runner, batch)
            self.counts['replays'] += 1
            tokens = runner._graph_greedy_tokens
            self.pending = {'signature': _signature(logits), 'stream': runner.stream,
                            'tokens': tokens, 'token_signature': _signature(tokens),
                            'generation': self.counts['replays']}
            return logits

        def sample(sampler, logits, args):
            current_stream = torch.cuda.current_stream(logits.device) if self.pending is not None else None
            matched, value = self._take_token(sampler, logits, args, current_stream)
            if matched:
                return value
            return originals[2](sampler, logits, args)

        self.replacements = (capture, replay, sample)
        runner_cls._capture_graphs, runner_cls.replay, sampler_cls.sample = self.replacements
        self.installed = True

    def _take_token(self, sampler, logits, args, current_stream):
        ticket, self.pending = self.pending, None  # consume on EVERY sample attempt
        reason = _reuse_reason(ticket, logits, args, sampler, current_stream)
        if reason is None:
            self.counts['sample_hits'] += 1
            return True, ticket['tokens']
        self.counts['original_samples'] += 1
        self.fallback_reasons[reason] = self.fallback_reasons.get(reason, 0) + 1
        return False, reason

    def report(self):
        return {'installed': self.installed, 'source_root': self.source_root,
                'source_sha256': dict(SOURCE_SHA256), 'counts': dict(self.counts),
                'fallback_reasons': dict(self.fallback_reasons), 'pending_ticket': self.pending is not None,
                'scope': 'B1 TP1 greedy, original FP32 Graph logits; Python routing counts, not hardware counters',
                'candidate_default': False}

    def assert_usage(self, *, replays, sample_hits, original_samples):
        for name, expected in [('replays', replays), ('sample_hits', sample_hits), ('original_samples', original_samples)]:
            _require(self.counts[name] == expected, f'{name}: {self.counts[name]} != {expected}')
        _require(self.pending is None and self.counts['unconsumed_replay_tickets'] == 0, 'stale/unconsumed replay ticket')
        _require(self.counts['captured_argmax'] == 1, 'missing candidate capture')
        return self.report()

    def uninstall(self):
        global _ACTIVE
        if not self.installed:
            return
        current = (self.runner_cls._capture_graphs, self.runner_cls.replay, self.sampler_cls.sample)
        _require(current == self.replacements, 'another class patch must be restored before uninstall')
        self.runner_cls._capture_graphs, self.runner_cls.replay, self.sampler_cls.sample = self.originals
        self.pending = None
        self.installed = False
        if _ACTIVE is self:
            _ACTIVE = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.uninstall()


def install():
    """Explicitly opt in before Engine construction; importing this file is inert."""
    global _ACTIVE
    _require(_ACTIVE is None, 'a patch is already installed')
    handle = PatchHandle()
    _ACTIVE = handle
    return handle
