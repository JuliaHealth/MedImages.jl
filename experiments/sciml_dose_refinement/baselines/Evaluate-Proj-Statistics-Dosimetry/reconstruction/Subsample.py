from pytomography.io.SPECT.shared import subsample_projections, subsample_metadata
from torch import Tensor
from pytomography.metadata.SPECT import SPECTObjectMeta, SPECTProjMeta
from typing import Tuple, Dict, Any

import torch
from torch.distributions import Binomial

def binomial_thinning(projections: torch.Tensor, t_reduction_factor: float) -> torch.Tensor:
    """
    Apply binomial thinning to a batch of 2D projection images.
    
    For each pixel (count = x), we draw y ~ Binomial(x, p = t_reduction_factor).
    This simulates 'thinning' the counts to match a shorter acquisition time.
    
    Args:
        projections (torch.Tensor): A 3D tensor of shape (N_proj, N_x, N_y) containing
                                    integer-valued pixel counts for each projection.
        t_reduction_factor (float): Fraction by which counts are thinned. 
                                    E.g., 0.666... to reduce from 15s to 10s.
                                    Must be between 0.0 and 1.0 inclusive.
    
    Returns:
        torch.Tensor: A 3D tensor of the same shape, with each pixel count
                      replaced by a binomially thinned value.
    """
    if not (0.0 < t_reduction_factor <= 1.0):
        raise ValueError("t_reduction_factor must be in (0, 1].")

    # Convert to integer type if not already (Binomial requires integer total_count).
    # This truncates any fractional part. In real data, projection counts are usually integers.
    counts = projections.to(torch.int64)

    # Create a Binomial distribution object. total_count is the pixel value, p is the fraction.
    # Note: This is vectorized across the entire tensor.
    dist = Binomial(total_count=counts, probs=t_reduction_factor)

    # Sample from the distribution to get the thinned counts.
    thinned_counts = dist.sample()

    return thinned_counts


def subsample_projections_number(
    photopeak: Tensor,
    scatter: Tensor,
    object_meta: SPECTObjectMeta,
    proj_meta: SPECTProjMeta,
    parameters: Dict[str, Any]) -> Tuple[Tensor, Tensor, SPECTObjectMeta, SPECTProjMeta]:
    """_summary_

    Parameters
    ----------
    photopeak : Tensor
        _description_
    scatter : Tensor
        _description_
    object_meta : SPECTObjectMeta
        _description_
    proj_meta : SPECTProjMeta
        _description_
    parameters : Dict[Any]
        _description_
        
    Returns
    -------
    Tuple[Tensor, Tensor, SPECTObjectMeta, SPECTProjMeta]
        _description_
    """
    
    if "Angle_Reduction" not in parameters or "Angle_Start_Index" not in parameters:
        raise AssertionError(f"Missing adequate parameters to subsample projecttions. Found {parameters}")
    
    object_meta, proj_meta = subsample_metadata(
        object_meta=object_meta,
        proj_meta=proj_meta,
        N_pixel=1,
        N_angle=parameters["Angle_Reduction"], 
        N_angle_start=parameters["Angle_Start_Index"]
    )
    
    # Photopeak
    photopeak = subsample_projections(
        projections=photopeak,
        N_pixel=1,
        N_angle=parameters["Angle_Reduction"],
        N_angle_start=parameters["Angle_Start_Index"],
                                    
    )
    
    # Scatter
    scatter = subsample_projections(projections=scatter,
                                    N_pixel=1,
                                    N_angle=parameters["Angle_Reduction"],
                                    N_angle_start=parameters["Angle_Start_Index"],
                                    )
    
    return photopeak, scatter, object_meta, proj_meta


def subsample_projections_time(projections: torch.Tensor, t_reduction_factor: float) -> torch.Tensor:
    """
    Apply binomial thinning to a batch of 2D projection images.
    
    For each pixel (count = x), we draw y ~ Binomial(x, p = t_reduction_factor).
    This simulates 'thinning' the counts to match a shorter acquisition time.
    
    Args:
        projections (torch.Tensor): A 3D tensor of shape (N_proj, N_x, N_y) containing
                                    integer-valued pixel counts for each projection.
        t_reduction_factor (float): Fraction by which counts are thinned. 
                                    E.g., 0.666... to reduce from 15s to 10s.
                                    Must be between 0.0 and 1.0 inclusive.
    
    Returns:
        torch.Tensor: A 3D tensor of the same shape, with each pixel count
                      replaced by a binomially thinned value.
    """
    if not (0.0 <= t_reduction_factor <= 1.0):
        raise ValueError("t_reduction_factor must be in [0, 1].")

    # Convert to integer type if not already (Binomial requires integer total_count).
    # This truncates any fractional part. In real data, projection counts are usually integers.
    counts = projections.float()

    # Create a Binomial distribution object. total_count is the pixel value, p is the fraction.
    # Note: This is vectorized across the entire tensor.
    dist = Binomial(total_count=counts, probs=t_reduction_factor)

    # Sample from the distribution to get the thinned counts.
    thinned_counts = dist.sample()

    return thinned_counts