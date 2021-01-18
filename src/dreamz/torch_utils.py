from torch import nn


class Lambda(nn.Module):
    def __init__(self, f):
        super(Lambda, self).__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


def adjust_learning_rate(optimizer, multiplier=None, new_lr=None):
    fact = (multiplier is not None) or (new_lr is not None)
    assert fact
    state_dict = optimizer.state_dict()
    if multiplier is not None:
        assert new_lr is None
        new_lr = next(iter(state_dict["param_groups"]))["lr"] * multiplier
    for param_group in state_dict["param_groups"]:
        param_group["lr"] = new_lr
    optimizer.load_state_dict(state_dict)
    print("Changed learning rate to {}".format(new_lr))
