# Copyright (c) 2023 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Author: Matthias Bitzer, matthias.bitzer3@de.bosch.com
import torch
import math
import numpy as np

from amorstructgp.nn.dataset_encoder import get_padded_dataset_and_masks


def cal_marg_likelihood(K, f, epsilon, kernel_mask, diagonal_mask, N, device):
    """
    Source: https://github.com/PrincetonLIPS/AHGP Licensed under the MIT License
    """
    # K: B X N X N
    # f: B X N X 1 (filled with zeros)
    diag_size = f.shape[1]
    K = (K + epsilon * torch.eye(diag_size).to(device).unsqueeze(0)) * kernel_mask  # fill the rest with zeros
    K = K + torch.eye(diag_size).to(device).unsqueeze(0) * (1 - kernel_mask)  # add ones to the diagonal
    L, info = torch.linalg.cholesky_ex(K)
    num_nonzero = torch.count_nonzero(info)
    if num_nonzero > 0:
        del L, num_nonzero, info, K
        return None, False
    # L = torch.cholesky(K)
    singular_values = L.diagonal(offset=0, dim1=1, dim2=2)
    logdet = torch.sum(torch.log(singular_values) * 2 * (1 - diagonal_mask), 1)
    data_fit = -(f.transpose(-1, -2)).matmul(torch.inverse(K)).matmul(f).squeeze(1).squeeze(1)
    AvgMLL = (0.5 * data_fit - 0.5 * logdet) / N - 0.5 * math.log(2 * math.pi)
    return AvgMLL, True


def cal_marg_likelihood_batch_noise(K, f, noise_variances, kernel_mask, diagonal_mask, N, device):
    # K: B X N X N
    # f: B X N X 1 (filled with zeros)
    # noise_variances: B x 1
    diag_size = f.shape[1]
    batch_size = f.shape[0]
    identities = torch.eye(diag_size).to(device).unsqueeze(0).expand(batch_size, diag_size, diag_size)  # B x N x N
    noise_variance_terms = identities * noise_variances[:, None]
    K = (K + noise_variance_terms) * kernel_mask  # fill the rest with zeros
    K = K + torch.eye(diag_size).to(device).unsqueeze(0) * (1 - kernel_mask)  # add ones to the diagonal
    L, info = torch.linalg.cholesky_ex(K)
    num_nonzero = torch.count_nonzero(info)
    if num_nonzero > 0:
        del num_nonzero, L, K, info
        return None, False
    # L = torch.cholesky(K)
    singular_values = L.diagonal(offset=0, dim1=1, dim2=2)
    logdet = torch.sum(torch.log(singular_values) * 2 * (1 - diagonal_mask), 1)
    data_fit = -(f.transpose(-1, -2)).matmul(torch.inverse(K)).matmul(f).squeeze(1).squeeze(1)
    AvgMLL = (0.5 * data_fit - 0.5 * logdet) / N - 0.5 * math.log(2 * math.pi)
    return AvgMLL, True


def GP_noise(y1, K11, K12, K22, epsilon_noise, device=torch.device("cpu")):
    """
    Source: https://github.com/PrincetonLIPS/AHGP Licensed under the MIT License
    Calculate the posterior mean and covariance matrix for y2 based on the noisy observations y1 and the given kernel matrix
    """
    # y1: N_train x 1
    # K11: N_train x N_train
    # K12: N_train x N_test
    # K22: N_test x N_test
    # Kernel of the noisy observations
    K11 = K11 + epsilon_noise * torch.eye(K11.shape[0]).to(device)
    solved = torch.linalg.solve(K11, K12)
    # Compute posterior mean
    mu_2 = torch.matmul(solved.T, y1)
    var_2 = K22 - torch.matmul(solved.T, K12)
    return mu_2, var_2  # mean, covariance
