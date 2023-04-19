# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import torch
import torch.nn.functional as F
from solo.utils.misc import gather, get_rank

def my_loss_func(z_list):

    loss = 0
    criterion = torch.nn.MSELoss()

    for one in z_list:

        noise = torch.mean(one[-2:],axis = 0)
        feature = one[:-2]
        batch_size = int(len(feature)/2)
        original = feature[:batch_size]
        diff = feature[batch_size:]
        # noise = noise.repeat(batch_size,1)
        # midpoint = (original+noise)/2
        loss += (torch.norm(noise) + 0.5*(torch.norm(diff)-torch.norm(feature)))/batch_size
        # loss += criterion(diff, midpoint)
    return loss

def simclr_loss_func(
    z: torch.Tensor, indexes: torch.Tensor, t_all:list, temperature: float = 0.1
) -> torch.Tensor:
    """Computes SimCLR's loss given batch of projected features z
    from different views, a positive boolean mask of all positives and
    a negative boolean mask of all negatives.

    Args:
        z (torch.Tensor): (N*views) x D Tensor containing projected features from the views.
        indexes (torch.Tensor): unique identifiers for each crop (unsupervised)
            or targets of each crop (supervised).

    Return:
        torch.Tensor: SimCLR loss.
    """


    gathered_indexes = gather(indexes)

    indexes = indexes.unsqueeze(0)
    gathered_indexes = gathered_indexes.unsqueeze(0)
    # positives
    pos_mask = indexes.t() == gathered_indexes
    pos_mask[:, z.size(0) * get_rank() :].fill_diagonal_(0)
    # negatives
    neg_mask = indexes.t() != gathered_indexes
    fake_negative_pair = (gathered_indexes < 0) * (gathered_indexes < 0)
    neg_mask = neg_mask*(~fake_negative_pair)

    t_all = torch.cat(t_all)

    temperature_ratio = torch.ones_like(neg_mask)
    temperature_ratio = temperature_ratio.to('cuda')

    batch_size = len(t_all)
    
    mask_tensor = torch.zeros((batch_size,batch_size)).to('cuda')
    mask_tensor += ((t_all == 0).unsqueeze(0) * (t_all == 5).unsqueeze(1)) * 4
    mask_tensor += ((t_all == 5).unsqueeze(0) * (t_all == 0).unsqueeze(1)) * 4
    mask_tensor += ((t_all == 0).unsqueeze(0) * (t_all == 7).unsqueeze(1)) * 2
    mask_tensor += ((t_all == 7).unsqueeze(0) * (t_all == 0).unsqueeze(1)) * 2

    mask_tensor = torch.where(mask_tensor==0, torch.tensor(1).to('cuda'), mask_tensor)


    # diff_mask = ((indexes.t() < 0) * (indexes.t() != -99999999)) * (gathered_indexes > 0)
    # temperature = temperature * torch.where(diff_mask, torch.tensor(4.0).to('cuda'), torch.tensor(1.0).to('cuda'))

    temperature = temperature * mask_tensor

    z = F.normalize(z, dim=-1)
    gathered_z = gather(z)
    tep = torch.einsum("if, jf -> ij", z, gathered_z)
    sim = torch.exp(tep/temperature)

    pos = torch.sum(sim * pos_mask, 1)
    neg = torch.sum(sim * neg_mask, 1)
    loss = -(torch.mean(torch.log(pos / (pos + neg))))
    return loss
