import torch

def random_flip(model, n=100, scale=0.1):
    params = [p for p in model.parameters() if p.requires_grad]
    for _ in range(n):
        p = params[torch.randint(len(params), (1,)).item()]
        idx = tuple(torch.randint(0, s, (1,)).item() for s in p.shape)
        p[idx] += torch.randn(1).item() * scale

def rank1(model, scale=0.05):
    w = model.head.weight
    u = torch.randn(w.shape[0], 1, device=w.device)
    v = torch.randn(1, w.shape[1], device=w.device)
    w += scale * (u @ v)

def adaptive(model, canary_x, canary_y, max_flips=5000, eps=1e-4):
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    changed = 0
    for _ in range(max_flips):
        p = params[torch.randint(len(params), (1,)).item()]
        idx = tuple(torch.randint(0, s, (1,)).item() for s in p.shape)
        delta = torch.randn(1).item() * 0.01
        original = p[idx].clone()
        p[idx] += delta
        with torch.no_grad():
            if not torch.allclose(model(canary_x), canary_y, atol=eps):
                p[idx] = original   # revert
            else:
                changed += 1
    print(f'adaptive: flipped {changed} coords while keeping canaries within {eps}')
