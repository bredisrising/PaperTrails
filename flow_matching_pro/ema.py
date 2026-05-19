import copy
import torch


class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for p_ema, p in zip(self.model.parameters(), model.parameters()):
            p_ema.lerp_(p.detach(), 1.0 - self.decay)
        for b_ema, b in zip(self.model.buffers(), model.buffers()):
            b_ema.copy_(b)

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, sd):
        self.model.load_state_dict(sd)
