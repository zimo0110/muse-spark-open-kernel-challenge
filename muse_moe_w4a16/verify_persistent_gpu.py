"""Hopper development validation for the persistent operator; never submits jobs.

Requires a real CUDA development GPU and the challenge dependencies. The
official `okc check` remains the authoritative correctness/timing measurement.
"""
import argparse
import importlib.util
from pathlib import Path
import secrets
import time

ROOT = Path(__file__).resolve().parent


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def capture(torch, fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
        fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    return graph, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, default=ROOT/'cuda_persistent_bf16.py')
    parser.add_argument('--shapes', type=int, nargs='+', default=[1,4,8,16,32,33,128,256,257])
    parser.add_argument('--replays', type=int, default=64)
    args = parser.parse_args()
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        parser.error('Requires a Hopper CUDA development GPU; no GPU check was run.')
    candidate = load(args.candidate.resolve(), 'persistent_candidate')
    reference = load(ROOT/'reference.py', 'development_reference')
    weights = reference.generate_weights(secrets.randbits(63))
    started = time.monotonic()
    state = candidate.prepare(weights)
    torch.cuda.synchronize()
    print(f'prepare_seconds={time.monotonic()-started:.3f}', flush=True)
    cases = {}
    for m in args.shapes:
        data = reference.generate_input(m, 'ep8', secrets.randbits(63))
        graph, output = capture(torch, lambda: candidate.kernel_fn(data, state))
        cases[m] = data, graph, output
    # Alternate shape-specific graphs sharing the same prepared scratch. This
    # exposes stale routing, counter-generation, and chunk-boundary assumptions.
    for replay in range(args.replays):
        order = args.shapes if replay % 2 else list(reversed(args.shapes))
        for m in order:
            data, graph, output = cases[m]
            fresh = reference.generate_input(m, 'ep8', secrets.randbits(63),
                                            dist='skewed' if replay % 2 else 'normal')
            if replay % 8 == 2:
                fresh['topk_ids'].fill_(255)  # empty local workload: exact zeros
            elif replay % 8 == 5:
                fresh['topk_ids'].copy_(torch.arange(8,device='cuda',dtype=torch.int32).expand(m,8))
            for key in ('hidden_states','topk_ids','topk_weights'):
                data[key].copy_(fresh[key])
            graph.replay()
            torch.cuda.synchronize()
            ok, message = reference.check_implementation(data, output, weights)
            if not ok:
                raise AssertionError((replay,m,message))
        if replay % 8 == 0:
            print(f'PASS replay_round={replay}: mixed shapes, fresh normal/skew/empty/all-local inputs',flush=True)
    print('Development replay correctness passed; no official score or leaderboard submission was made.')


if __name__ == '__main__':
    main()
