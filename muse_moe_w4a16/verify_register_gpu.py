"""Development validation on a real CUDA Hopper GPU; never submits to a leaderboard.

Run in the challenge dependency environment:
  python verify_register_gpu.py --mode both --soak 128
The official `okc check register_candidate.py` remains authoritative once quota resets.
"""
import argparse
import importlib.util
from pathlib import Path
import secrets
import time

ROOT = Path(__file__).resolve().parent


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def capture(torch, fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    return graph, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['persistent', 'grid', 'both'], default='both')
    parser.add_argument('--shapes', type=int, nargs='+', default=[1, 4, 8, 16, 32, 33, 128, 256, 257])
    parser.add_argument('--soak', type=int, default=128)
    parser.add_argument('--timing-rounds', type=int, default=30)
    parser.add_argument('--baseline', action='store_true', help='also capture the supplied SGLang baseline for paired timing')
    args = parser.parse_args()
    import torch
    if not torch.cuda.is_available():
        parser.error('CUDA GPU unavailable. Use verify_register_cpu.py for CPU semantics; this requires H100/H200.')
    if torch.cuda.get_device_capability()[0] != 9:
        parser.error('Use a Hopper (SM90) GPU for this development test.')
    ref = module(ROOT / 'reference.py', 'moe_reference_development_only')
    candidate = module(ROOT / 'register_candidate.py', 'register_candidate_development')
    weights = ref.generate_weights(secrets.randbits(63))
    started = time.monotonic()
    state = candidate.prepare(weights)
    torch.cuda.synchronize()
    print(f'packing: {time.monotonic()-started:.3f}s')
    baseline = module(ROOT / 'baseline.py', 'moe_baseline_development') if args.baseline else None
    baseline_state = baseline.prepare(weights) if baseline else None
    modes = [True, False] if args.mode == 'both' else [args.mode == 'persistent']
    flush = torch.ones(64 << 20, dtype=torch.float32, device='cuda')
    for persistent in modes:
        state['persistent'] = persistent
        for m in args.shapes:
            data = ref.generate_input(m, 'ep8', secrets.randbits(63))
            graph, output = capture(torch, lambda: candidate.kernel_fn(data, state))
            for step in range(0, max(args.soak, 1), 16):
                fresh = ref.generate_input(m, 'ep8', secrets.randbits(63), dist='skewed' if step // 16 % 2 else 'normal')
                for key in ('hidden_states', 'topk_ids', 'topk_weights'):
                    data[key].copy_(fresh[key])
                for _ in range(min(16, max(args.soak, 1)-step)): graph.replay()
                torch.cuda.synchronize()
                ok, message = ref.check_implementation(data, output, weights)
                if not ok: raise AssertionError(f'm={m} persistent={persistent} step={step}: {message}')
            # A reused graph must respond to new input values, not retain earlier routes.
            print(f'PASS m={m} persistent={persistent}: {message}')
            base_graph = capture(torch, lambda: baseline.kernel_fn(data, baseline_state))[0] if baseline else None
            times, base_times = [], []
            for rep in range(args.timing_rounds):
                order = [(graph, times), (base_graph, base_times)]
                if rep % 2: order.reverse()
                for g, target in order:
                    if g is None: continue
                    flush.sum()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record(); g.replay(); end.record(); end.synchronize()
                    target.append(start.elapsed_time(end) * 1000)
            candidate_us = sorted(times)[len(times)//2]
            text = f'DEV timing m={m} persistent={persistent}: {candidate_us:.2f}us (raw graph time)'
            if base_times:
                ratios = sorted(a/b for a, b in zip(base_times, times))
                text += f', median paired baseline/candidate={ratios[len(ratios)//2]:.3f}x'
            print(text)
    print('Development GPU validation finished; these timings are not an official challenge score.')


if __name__ == '__main__':
    main()
