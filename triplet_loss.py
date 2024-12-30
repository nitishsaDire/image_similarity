import torch
import torch.nn as nn
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()  # check if GPU exists
device = "cuda" if use_cuda else "cpu"  # use CPU or GPU

eps = 1e-8 # an arbitrary small value to be used for numerical stability tricks

def euclidean_distance_matrix(x):
  """Efficient computation of Euclidean distance matrix

  Args:
    x: Input tensor of shape (batch_size, embedding_dim)
    
  Returns:
    Distance matrix of shape (batch_size, batch_size)
  """
  # step 1 - compute the dot product

  # shape: (batch_size, batch_size)
  dot_product = torch.mm(x, x.t())

  # step 2 - extract the squared Euclidean norm from the diagonal

  # shape: (batch_size,)
  squared_norm = torch.diag(dot_product)

  # step 3 - compute squared Euclidean distances

  # shape: (batch_size, batch_size)
  distance_matrix = squared_norm.unsqueeze(0) - 2 * dot_product + squared_norm.unsqueeze(1)

  # get rid of negative distances due to numerical instabilities
  distance_matrix = F.relu(distance_matrix)

  # step 4 - compute the non-squared distances
  
  # handle numerical stability
  # derivative of the square root operation applied to 0 is infinite
  # we need to handle by setting any 0 to eps
  mask = (distance_matrix == 0.0).float()

  # use this mask to set indices with a value of 0 to eps
  distance_matrix =distance_matrix.clone() + mask * eps

  # now it is safe to get the square root
  distance_matrix = torch.sqrt(distance_matrix.clone())

  # undo the trick for numerical stability
  distance_matrix = distance_matrix.clone() * (1.0 - mask)

  return distance_matrix


def get_triplet_mask(labels):
  """compute a mask for valid triplets

  Args:
    labels: Batch of integer labels. shape: (batch_size,)

  Returns:
    Mask tensor to indicate which triplets are actually valid. Shape: (batch_size, batch_size, batch_size)
    A triplet is valid if:
    `labels[i] == labels[j] and labels[i] != labels[k]`
    and `i`, `j`, `k` are different.
  """
  # step 1 - get a mask for distinct indices

  # shape: (batch_size, batch_size)
  indices_equal = torch.eye(labels.size()[0], dtype=torch.bool, device=labels.device)
  indices_not_equal = torch.logical_not(indices_equal)
  # shape: (batch_size, batch_size, 1)
  i_not_equal_j = indices_not_equal.unsqueeze(2)
  # shape: (batch_size, 1, batch_size)
  i_not_equal_k = indices_not_equal.unsqueeze(1)
  # shape: (1, batch_size, batch_size)
  j_not_equal_k = indices_not_equal.unsqueeze(0)
  # Shape: (batch_size, batch_size, batch_size)
  distinct_indices = torch.logical_and(torch.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)

  # step 2 - get a mask for valid anchor-positive-negative triplets

  # shape: (batch_size, batch_size)
  labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
  # shape: (batch_size, batch_size, 1)
  i_equal_j = labels_equal.unsqueeze(2)
  # shape: (batch_size, 1, batch_size)
  i_equal_k = labels_equal.unsqueeze(1)
  # shape: (batch_size, batch_size, batch_size)
  valid_indices = torch.logical_and(i_equal_j, torch.logical_not(i_equal_k))

  # step 3 - combine two masks
  mask = torch.logical_and(distinct_indices, valid_indices)

  return mask


def hard_example_mining(dist_mat, labels):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t()).to(device)
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t()).to(device)

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        (dist_mat * is_pos.float()).contiguous().view(N, -1), 1, keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    temp = dist_mat * is_neg.float()
    temp[temp == 0] = 10e5
    dist_an, relative_n_inds = torch.min(
        (temp).contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    return dist_ap, dist_an



class BatchAllTtripletLoss(nn.Module):
  """Uses all valid triplets to compute Triplet loss

  Args:
    margin: Margin value in the Triplet Loss equation
  """
  def __init__(self, margin=1., hard_mining=False):
    super().__init__()
    self.margin = margin
    self.hard_mining = hard_mining
    
  def forward(self, embeddings, labels):
    """computes loss value.

    Args:
      embeddings: Batch of embeddings, e.g., output of the encoder. shape: (batch_size, embedding_dim)
      labels: Batch of integer labels associated with embeddings. shape: (batch_size,)

    Returns:
      Scalar loss value.
    """
    # step 1 - get distance matrix
    # shape: (batch_size, batch_size)
    distance_matrix = euclidean_distance_matrix(embeddings).to(device)

    if not self.hard_mining:
        # step 2 - compute loss values for all triplets by applying broadcasting to distance matrix

        # shape: (batch_size, batch_size, 1)
        anchor_positive_dists = distance_matrix.unsqueeze(2)
        # shape: (batch_size, 1, batch_size)
        anchor_negative_dists = distance_matrix.unsqueeze(1)
        # get loss values for all possible n^3 triplets
        # shape: (batch_size, batch_size, batch_size)
        triplet_loss = anchor_positive_dists - anchor_negative_dists + self.margin

        # step 3 - filter out invalid or easy triplets by setting their loss values to 0

        # shape: (batch_size, batch_size, batch_size)
        mask = get_triplet_mask(labels).to(device)
        triplet_loss *= mask
        # easy triplets have negative loss values
        triplet_loss = F.relu(triplet_loss)

        # step 4 - compute scalar loss value by averaging positive losses
        num_positive_losses = (triplet_loss > eps).float().sum()
        triplet_loss = triplet_loss.sum() / (num_positive_losses + eps)
    else:
        ap, an=hard_example_mining(distance_matrix, labels)
        triplet_loss = F.relu(ap - an + self.margin)
        triplet_loss = triplet_loss.sum()/(triplet_loss>0).sum()
    return triplet_loss