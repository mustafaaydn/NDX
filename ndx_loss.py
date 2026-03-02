import torch


class NDX_Loss(torch.nn.Module):
    R"""
    Instance-only dependent loss correction for label noise in binary classification.

    Parameters
    ----------
    base_loss : function
        \ell in the paper, i.e., the loss to modify. Takes real-valued scores and discrete labels as arguments
        and returns a vector of loss values evaluated at them. `scores` is a B-length vector of real values
        and `labels` is a B-length of integers from the set {0, 1}. Both arguments to it will be torch.Tensors.

    sigmoid_scale : float
        \beta in the paper such that \rho_x = \sigma(-\beta * dists)

    regularization_scale : float
        \lambda in the paper such that ~\ell^R = ~\ell_{\text{numerator}} - \lambda ~\ell_{\text{denominator}}

    num_warmup_epochs : int
        Use the base loss function in the beginning for this many epochs. Please see Appendix A.5.
    """
    def __init__(self, base_loss, sigmoid_scale, regularization_scale, num_warmup_epochs):
        super().__init__()
        self.base_loss = base_loss
        self.sigmoid_scale = sigmoid_scale
        self.regularization_scale = regularization_scale
        self.num_warmup_epochs = num_warmup_epochs

    def forward(self, scores, labels, epoch_index):
        R"""
        Suffer the modified loss ~\ell^R.

        Parameters
        ----------
        scores : torch.FloatTensor of shape (B,)
            Real-valued scores, i.e., the ones before becoming P(~Y = +1 | X)
        labels : torch.LongTensor of shape (B,)
            Integer-valued labels from {0, 1}
        epoch_index : int
            1-based index of the current epoch. Used for warmup.

        Returns
        -------
        ~\ell^R : torch.float
            Modified (regularized) loss over the scores and labels
        """
        # If in warmup, suffer the normal loss
        if epoch_index <= self.num_warmup_epochs:
            return self.base_loss(scores, labels).mean()

        ## Calculate distance-based probabilities of label flips
        # "self.sigmoid_scale" is \beta
        distances = scores.detach().abs()
        rho_x = 1 / (1 + torch.exp(self.sigmoid_scale * distances))

        ## Compute the corrected loss
        # The base loss first: normal and labels flipped to be weighted
        normal_loss   = self.base_loss(scores, labels)
        opposite_loss = self.base_loss(scores, 1-labels)

        # Approximate P(Y | X) and P(˜Y | X), form the modified loss
        f_x         = torch.sigmoid(scores)
        pyy_x       = torch.where(labels == +1, f_x, 1 - f_x)
        numerator   = (1 - pyy_x - rho_x) * pyy_x * normal_loss - rho_x * (1 - pyy_x) * opposite_loss
        denominator = pyy_x * (1 - pyy_x) - rho_x

        # "self.regularization_scale" is \lambda
        loss = numerator - self.regularization_scale * denominator
        return loss.mean()
