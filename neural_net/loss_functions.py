import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def fem_residual_loss(A, F, u):
    residuo = torch.sparse.mm(A, u) - F.view_as(u)
    return torch.linalg.vector_norm(residuo, ord=2)
