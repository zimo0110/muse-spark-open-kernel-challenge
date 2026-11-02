
# Development-only prepare validation. No diagnostic data survives in state;
# official tests remain the independent mathematical correctness check.
def _codex_replay_stress(state, dev):
    shapes = (1, 4, 8, 16, 17, 32, 33, 65, 128, 256, 257, 385)
    stream = torch.cuda.Stream(device=dev)
    stream.wait_stream(torch.cuda.current_stream(dev))
    cases = []
    with torch.cuda.stream(stream):
        for m in shapes:
            data = _gen_input(m, 70001 + m, dev)
            kernel_fn(data, state)
            kernel_fn(data, state)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = kernel_fn(data, state)
            cases.append((m, data, graph, output))
            stream.synchronize()
            print(f"[codex replay] captured M={m}", flush=True)
    torch.cuda.current_stream(dev).wait_stream(stream)
    torch.cuda.synchronize(dev)
    for regime in range(4):
        print(f"[codex replay] regime={regime} begin", flush=True)
        expected = []
        for m, data, graph, output in cases:
            fresh = _gen_input(m, 91003 + 31*m + regime, dev)
            if regime == 1:  # no local work
                fresh['topk_ids'].fill_(255)
            elif regime == 2:  # all local, fat jobs and chunk boundary
                fresh['topk_ids'].copy_(torch.arange(8, device=dev).expand(m, 8))
            elif regime == 3:  # a deliberately unbalanced mixture
                ids = fresh['topk_ids']
                ids[:max(1,m//3)].copy_(torch.arange(8, device=dev).expand(max(1,m//3), 8))
            for key in ('hidden_states', 'topk_ids', 'topk_weights'):
                data[key].copy_(fresh[key])
            expected.append(kernel_fn(data, state).clone())
            torch.cuda.synchronize(dev)
            print(f"[codex replay] eager M={m} regime={regime}", flush=True)
        for turn in range(4):
            sequence = list(zip(cases, expected))
            if turn % 2:
                sequence.reverse()
            for (m, data, graph, output), want in sequence:
                graph.replay()
                torch.cuda.synchronize(dev)
                if not torch.isfinite(output).all().item():
                    raise AssertionError(('replay nonfinite', m, regime, turn))
                if regime == 1:
                    if torch.count_nonzero(output).item():
                        raise AssertionError(('empty routes not zero', m, turn))
                else:
                    diff = torch.linalg.vector_norm(output.float() - want.float())
                    norm = torch.linalg.vector_norm(want.float()).clamp_min(1e-20)
                    error = (diff/norm).item()
                    if error > 0.001:
                        raise AssertionError(('mixed graph replay differs from eager', m, regime, turn, error))
    print('[codex replay] PASS 192 mixed-shape graph replays; fresh normal/empty/all-local/skew; eager consistency, not independent reference', flush=True)

_codex_original_prepare = prepare

def prepare(weights):
    state = _codex_original_prepare(weights)
    _codex_replay_stress(state, weights['w13'].device)
    return state
