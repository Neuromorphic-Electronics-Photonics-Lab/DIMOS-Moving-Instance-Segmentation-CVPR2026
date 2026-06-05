import torch

class GradientReverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

def grl(x, lambd=1.0):
    """
    Gradient Reversal Layer (GRL)
    Forward pass is identity mapping; backward pass multiplies gradient by -lambd
    """
    return GradientReverseFunction.apply(x, lambd)