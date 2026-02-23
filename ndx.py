import torch
from torch import exp, log, sigmoid

def loss_fun(self, X, y):
    """
    Given (batch) pairs X and y (noisy), return ˜lˆR,
    where l (the base loss) is assumed the logistic loss
    """
    ## Feed-forward
    h_x = self(X).squeeze(-1)
    f_x = sigmoid(h_x)

    ## Calculate distance-based probabilities of label flips
    # "self.sigmoid_scale" is \beta
    distances = h_x.abs()
    rho_x = 1 / (1 + exp(self.sigmoid_scale * distances))

    ## Compute the corrected loss
    # Baseloss (\ell; logistic) w.r.t. ˜y and -˜y
    normal_loss   = torch.where(y == +1, -log(f_x), -log(1 - f_x))
    opposite_loss = torch.where(y == +1, -log(1 - f_x), -log(f_x))

    # Approximate P(Y | X) and P(˜Y | X), form the modified loss
    pyy_x       = torch.where(y == +1, f_x, 1 - f_x)
    numerator   = (1 - pyy_x - rho_x) * pyy_x * normal_loss - rho_x * (1 - pyy_x) * opposite_loss
    denominator = pyy_x * (1 - pyy_x) - rho_x

    # "self.regularization_scale" is \lambda
    loss = numerator - self.regularization_scale * denominator
    return loss.mean()