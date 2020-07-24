import numpy as np


def squared_op(vec, ndims, op=np.add):
    ones = np.matmul(np.ones_like(vec), np.transpose(np.ones_like(vec))) / ndims
    ones = ones[..., np.newaxis]
    ones = np.dstack([ones] * ndims)
    vec = vec[:, np.newaxis, ...]
    vec_squared = vec * ones
    
    return op(vec_squared, np.transpose(vec_squared, axes=[1, 0, 2]))

def diagonalize(sigma, ndims):
    eye = np.eye(ndims)[..., np.newaxis]
    sigma = np.transpose(sigma)[:, np.newaxis, ...]
    return eye * sigma

def squared_sigma(diag, op=np.add):
    ones = np.ones_like(diag)[..., np.newaxis]
    print(ones.shape)
    ones = ones * np.transpose(ones, axes=[0, 1, 3, 2])
    
    ext = ones * diag[..., np.newaxis]
    summed = op(ext, np.transpose(ext, axes=[0, 1, 3, 2]))
    summed = np.transpose(summed, axes=[2, 3, 0, 1])
    return summed

def diag_inverse(diag):
    mask = np.where(np.greater(diag, 0.0), np.ones_like(diag), np.zeros_like(diag))
    return 1.0 / (diag + 1.0 - mask) * mask

def det_diag(diag, keepdims=False):
    diag = np.diagonal(diag, axis1=-2, axis2=-1)
    return np.prod(diag, axis=-1, keepdims=keepdims)

def pairwise_bhattacharyya(mu, std, diagonal=False):
    ndim = int(mu.shape[1])
    smu = squared_op(mu, ndim, op=np.subtract)
    if diagonal:
        diag = diagonalize(std, ndim)
    else:
        raise NotImplementedError()
            
    diagsq = squared_sigma(diag, op=lambda x, y: np.add(x, y) / 2.0)
    diaginv = diag_inverse(diagsq)

    # Compute first term
    smu_v = smu[:, :, np.newaxis, ...]
    smu_h = smu[..., np.newaxis]
    dist = np.matmul(np.matmul(smu_v, diaginv), smu_h)
    term_a = 1 / 8.0 * dist
    term_a = np.squeeze(np.squeeze(term_a, axis=-1), axis=-1)
    
    det_all = det_diag(diagsq)
    det_single = det_diag(np.transpose(diag, axes=[2, 0, 1]), keepdims=True)
    det_sq = np.matmul(det_single, np.transpose(det_single))
    
    frac_det = np.sqrt(det_sq)
    frac_det_mask = np.where(np.greater(frac_det, 0.0), np.ones_like(frac_det), np.zeros_like(frac_det))
    
    frac = det_all / (frac_det + (1.0 - frac_det_mask)) * frac_det_mask
    frac_mask = np.where(np.greater(frac, 0.0), np.zeros_like(frac), np.ones_like(frac))
    
    log = np.log(frac + frac_mask)
    term_b = 1 / 2.0 * log
    
    return term_a + term_b

def battacharyya(mu1, std1, mu2, std2):
    sigma = (std1 + std2) / 2
    mus = (mu1 - mu2)
    return 1 / 8 * mus.T @ np.diag(1 / sigma) @ mus + 1 / 2 * np.log(np.prod(sigma) / np.sqrt(np.prod(std1) * np.prod(std2)))
