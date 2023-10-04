# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
The renderer is a module that takes in rays, decides where to sample along each
ray, and computes pixel colors using the volume rendering equation.
"""
import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

rot_eta = lambda eta : torch.Tensor([
    [np.cos(eta),np.sin(eta),0,0],
    [-np.sin(eta), np.cos(eta),0,0],
    [0,0,1,0],
    [0,0,0,1]]).float()

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    eta = 90
    c2w = rot_eta(eta / 180. * np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],
                                 [0,0,1,0],
                                 [0,1,0,0],
                                 [0,0,0,1]])) @ c2w
    return c2w.numpy()



import math

import numpy as np
import torch
import torch.nn as nn

from training.volumetric_rendering.ray_marcher import MipRayMarcher2
from training.volumetric_rendering import math_utils

def generate_planes():
    """
    Defines planes by the three vectors that form the "axes" of the
    plane. Should work with arbitrary number of planes and planes of
    arbitrary orientation.
    """
    return torch.tensor([[[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]],
                            [[1, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0]],
                            [[0, 0, 1],
                            [1, 0, 0],
                            [0, 1, 0]]], dtype=torch.float32)

def project_onto_planes(planes, coordinates):
    """
    Does a projection of a 3D point onto a batch of 2D planes,
    returning 2D plane coordinates.

    Takes plane axes of shape n_planes, 3, 3
    # Takes coordinates of shape N, M, 3
    # returns projections of shape N*n_planes, M, 2
    """
    N, M, C = coordinates.shape
    n_planes, _, _ = planes.shape
    coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1).reshape(N*n_planes, M, 3)
    inv_planes = torch.linalg.inv(planes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N*n_planes, 3, 3)
    projections = torch.bmm(coordinates, inv_planes)
    return projections[..., :2]

def sample_from_planes(plane_axes, plane_features, coordinates, mode='bilinear', padding_mode='zeros', box_warp=None):
    assert padding_mode == 'zeros'
    N, n_planes, C, H, W = plane_features.shape
    _, M, _ = coordinates.shape
    plane_features = plane_features.view(N*n_planes, C, H, W)

    coordinates = (2/box_warp) * coordinates # TODO: add specific box bounds

    projected_coordinates = project_onto_planes(plane_axes, coordinates).unsqueeze(1)
    output_features = torch.nn.functional.grid_sample(plane_features, projected_coordinates.float(), mode=mode, padding_mode=padding_mode, align_corners=False).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
    return output_features

def sample_from_3dgrid(grid, coordinates):
    """
    Expects coordinates in shape (batch_size, num_points_per_batch, 3)
    Expects grid in shape (1, channels, H, W, D)
    (Also works if grid has batch size)
    Returns sampled features of shape (batch_size, num_points_per_batch, feature_channels)
    """
    batch_size, n_coords, n_dims = coordinates.shape
    sampled_features = torch.nn.functional.grid_sample(grid.expand(batch_size, -1, -1, -1, -1),
                                                       coordinates.reshape(batch_size, 1, 1, -1, n_dims),
                                                       mode='bilinear', padding_mode='zeros', align_corners=False)
    N, C, H, W, D = sampled_features.shape
    sampled_features = sampled_features.permute(0, 4, 3, 2, 1).reshape(N, H*W*D, C)
    return sampled_features


class RenderNetwork(torch.nn.Module):
    def __init__(
            self,
            input_size,
            dir_count
    ):
        super().__init__()
        self.input_size = 3 * input_size + input_size * 2
        self.layers_main = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, 64),
            torch.nn.ReLU(),
            # torch.nn.Linear(256, 256),
            # torch.nn.ReLU(),
            # torch.nn.Linear(256, 256),
            # torch.nn.ReLU(),
            # torch.nn.Linear(256, 256),
            # torch.nn.ReLU(),
            # torch.nn.Linear(256, 256),
            # torch.nn.ReLU(),
        )
        # decoder_lr_mul
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, 64),
            torch.nn.Softplus(),
            torch.nn.Linear(64, 32 + 1),

        )

        self.layers_main_2 = torch.nn.Sequential(
            torch.nn.Linear(64 + self.input_size, 64),
            torch.nn.ReLU(),
            # torch.nn.Linear(256, 256),
            # torch.nn.ReLU(),
            # torch.nn.Linear(256, 256),
            # torch.nn.ReLU(),
            # torch.nn.Linear(256, 256),
            # torch.nn.ReLU(),
        )

        self.layers_sigma = torch.nn.Sequential(
            torch.nn.Linear(64 + self.input_size, 64),  # dodane wejscie tutaj moze cos pomoze
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )

        self.layers_rgb = torch.nn.Sequential(
            torch.nn.Linear(64 + self.input_size + dir_count, 64),
            torch.nn.ReLU(),
            # torch.nn.Linear(256, 256),
            # torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
        )

    def forward(self, sampled_features, ray_directions):
        # Aggregate features
        # sampled_features = sampled_features.mean(1)
        x = sampled_features

        # N, M, C = x.shape
        # x = x.view(N * M, C)

        x = self.net(x)
        # x = x.view(N, M, -1)
        rgb = torch.sigmoid(x[..., 1:]) * (1 + 2 * 0.001) - 0.001  # Uses sigmoid clamping from MipNeRF
        sigma = x[..., 0:1]
        return {'rgb': rgb, 'sigma': sigma}

    # def forward(self, triplane_code, dirs):
    #     x = self.layers_main(triplane_code)
    #     x1 = torch.concat([x, triplane_code], dim=1)
    #
    #     x = self.layers_main_2(x1)
    #     xs = torch.concat([x, triplane_code], dim=1)
    #
    #     sigma = self.layers_sigma(xs)
    #     x = torch.concat([x, triplane_code, dirs], dim=1)
    #     rgb = self.layers_rgb(x)
    #     return {'rgb': rgb,
    #             'sigma': sigma}


class ImagePlanes(torch.nn.Module):

    def __init__(self, focal, poses, images, count=np.inf, device='cuda'):
        super(ImagePlanes, self).__init__()

        self.pose_matrices = []
        self.K_matrices = []
        self.images = []

        self.focal = focal
        for i in range(min(count, poses.shape[0])):
            M = poses[i]
            M = torch.from_numpy(M)
            M = M @ torch.Tensor([[-1, 0, 0, 0],
                                  [0, 1, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]]).to(M.device)
            M = torch.inverse(M)
            M = M[0:3]
            self.pose_matrices.append(M)

            image = images[i]
            # image = torch.from_numpy(image)
            self.images.append(image)#.permute(2, 0, 1))
            self.size = float(image.shape[0])
            K = torch.Tensor(
                [[1.0254, 0, 0.5],
                 [0, 1.0254, 0.5],
                 [0, 0, 1]])
            self.K_matrices.append(K)

        self.pose_matrices = torch.stack(self.pose_matrices).to(device)
        self.K_matrices = torch.stack(self.K_matrices).to(device)
        self.image_plane = torch.stack(self.images).to(device)

    def forward(self, points=None):
        if points.shape[0] == 1:
            points = points[0]

        points = torch.concat([points, torch.ones(points.shape[0], 1).to(points.device)], 1).to(points.device)
        points_in_camera_coords = self.pose_matrices @ points.T
        # camera-origin distance is equal to 1 in points_in_camera_coords
        ps = self.K_matrices @ points_in_camera_coords
        pixels = (ps / ps[:, None, 2])[:, 0:2, :]
        # pixels = pixels / self.size
        # print("Pixels")
        # p = pixels.flatten()
        # print(pixels.min(), torch.quantile(p, 0.05),torch.quantile(p, 0.5), pixels.max())
        pixels = torch.clamp(pixels, 0, 1)
        # print("Pixels")
        # print(pixels)
        pixels = pixels * 2.0 - 1.0
        pixels = pixels.permute(0, 2, 1)

        feats = []
        for img in range(self.image_plane.shape[0]):
            feat = torch.nn.functional.grid_sample(
                self.image_plane[img].unsqueeze(0),
                pixels[img].unsqueeze(0).unsqueeze(0), mode='bilinear', padding_mode='zeros', align_corners=False)
            feats.append(feat)
        feats = torch.stack(feats).squeeze(1)
        pixels = pixels.permute(1, 0, 2)
        pixels = pixels.flatten(1)
        feats = feats.permute(2, 3, 0, 1)
        feats = feats.flatten(2)
        # print(feats[0].shape) # torch.Size([262144, 96])
        # print(pixels.shape) # torch.Size([262144, 6])
        feats = torch.cat((feats[0], pixels), 1)
        return feats


class MultiImageNeRF(torch.nn.Module):

    def __init__(self, image_plane):
        super(MultiImageNeRF, self).__init__()
        self.image_plane = image_plane

        # self.input_ch_views = dir_count

    def forward(self, input_pts, input_views, renderer_network):
        # input_pts, input_views = torch.split(x, [3, self.input_ch_views], dim=-1)
        x = self.image_plane(input_pts)
        return renderer_network(x, input_views)

import math

def cart2sph(x,y,z):
    XsqPlusYsq = x**2 + y**2
    elev = math.atan2(z,math.sqrt(XsqPlusYsq))     # theta
    az = math.atan2(y,x)                           # phi
    return elev, az

def fibonacci_sphere(samples=1000):

    points = []
    phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append(cart2sph(x,y,z))

    return points


class ImportanceRenderer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ray_marcher = MipRayMarcher2()
        self.plane_axes = generate_planes()
        count = 32
        dir_count = 3
        self.render_network = RenderNetwork(count, dir_count)

    def forward(self, planes, decoder, ray_origins, ray_directions, rendering_options):
        outs = []
        nerfs = []
        depths = []
        for plane, origin, direction in zip(planes, ray_origins, ray_directions):
            origin = origin.unsqueeze(0)
            direction = direction.unsqueeze(0)
            poses = [pose_spherical(theta, phi, -1.307) for phi, theta in fibonacci_sphere(32)]
            # poses = [pose_spherical((i - 8) / 8 * 180, 0, 2) for i in range(16)]
            # poses += [pose_spherical(0, (i - 8) / 8 * 180, 2) for i in range(16)]
            image_plane = ImagePlanes(focal=torch.Tensor([10.0]),
                        poses=np.stack(poses),
                        images=plane.view(32, 3, plane.shape[-2], plane.shape[-1]))
            plane_nerf = MultiImageNeRF(image_plane=image_plane)
            nerfs.append(plane_nerf)
            self.plane_axes = self.plane_axes.to(ray_origins.device)

            if rendering_options['ray_start'] == rendering_options['ray_end'] == 'auto':
                ray_start, ray_end = math_utils.get_ray_limits_box(origin, ray_directions, box_side_length=rendering_options['box_warp'])
                is_ray_valid = ray_end > ray_start
                if torch.any(is_ray_valid).item():
                    ray_start[~is_ray_valid] = ray_start[is_ray_valid].min()
                    ray_end[~is_ray_valid] = ray_start[is_ray_valid].max()
                depths_coarse = self.sample_stratified(origin, ray_start, ray_end, rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])
            else:
                # Create stratified depth samples
                depths_coarse = self.sample_stratified(origin, rendering_options['ray_start'], rendering_options['ray_end'], rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])

            batch_size, num_rays, samples_per_ray, _ = depths_coarse.shape
            depths.append(depths_coarse)
            # Coarse Pass
            sample_coordinates = (origin.unsqueeze(-2) + depths_coarse * direction.unsqueeze(-2)).reshape(batch_size, -1, 3)
            sample_directions = direction.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1).reshape(batch_size, -1, 3)

            out = plane_nerf(sample_coordinates.squeeze(0), sample_directions.squeeze(0), self.render_network)
            outs.append(out)

        out = {k: torch.stack([out[k] for out in outs]) for k in outs[0].keys()}
        batch_size = planes.shape[0]
        depths_coarse = torch.cat(depths, dim=0)

        colors_coarse = out['rgb'] # output from MultiPlane
        if rendering_options.get('density_noise', 0) > 0:
            out['sigma'] += torch.randn_like(out['sigma']) * rendering_options['density_noise']
        densities_coarse = out['sigma'] # output from MultiPlane
        colors_coarse = colors_coarse.reshape(batch_size, num_rays, samples_per_ray, colors_coarse.shape[-1])
        densities_coarse = densities_coarse.reshape(batch_size, num_rays, samples_per_ray, 1)

        # Fine Pass
        N_importance = rendering_options['depth_resolution_importance']
        if N_importance > 0:
            outs = []
            depths_ = []
            for directions, origins, colors, densities, depths, plane_nerf in zip(ray_directions,ray_origins,colors_coarse, densities_coarse, depths_coarse, nerfs):
                colors = colors.unsqueeze(0)
                densities = densities.unsqueeze(0)
                depths = depths.unsqueeze(0)
                directions = directions.unsqueeze(0)
                origins = origins.unsqueeze(0)

                _, _, weights = self.ray_marcher(colors, densities, depths, rendering_options)

                depths_fine = self.sample_importance(depths, weights, N_importance)

                sample_directions = directions.unsqueeze(-2).expand(-1, -1, N_importance, -1).reshape(1, -1, 3)
                sample_coordinates = (origins.unsqueeze(-2) + depths_fine * directions.unsqueeze(-2)).reshape(1, -1, 3)

                out = plane_nerf(sample_coordinates.squeeze(0), sample_directions.squeeze(0), self.render_network)
                outs.append(out)
                depths_.append(depths_fine)

            out = {k: torch.stack([out[k] for out in outs]) for k in outs[0].keys()}
            colors_fine = out['rgb']

            if rendering_options.get('density_noise', 0) > 0:
                out['sigma'] += torch.randn_like(out['sigma']) * rendering_options['density_noise']
            densities_fine = out['sigma'] # output from MultiPlane
            colors_fine = colors_fine.reshape(batch_size, num_rays, N_importance, colors_fine.shape[-1])
            densities_fine = densities_fine.reshape(batch_size, num_rays, N_importance, 1)
            depths_fine = torch.cat(depths_, dim=0)
            all_depths, all_colors, all_densities = self.unify_samples(depths_coarse, colors_coarse, densities_coarse,
                                                                      depths_fine, colors_fine, densities_fine)

            # Aggregate
            rgb_final, depth_final, weights = self.ray_marcher(all_colors, all_densities, all_depths, rendering_options)
        else:
            rgb_final, depth_final, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)

        # feature_samples, depth_samples, weights_samples
        return rgb_final, depth_final, weights.sum(2)

    def run_model(self, planes, decoder, sample_coordinates, sample_directions, options):
        outs = []
        nerfs = []
        # print("run_model shapes", planes.shape, sample_coordinates.shape)
        # [8, 3, 32, 256, 256]) ([8, 2000, 3])
        for plane, sample_coordinates_ in zip(planes, sample_coordinates):
            poses = [pose_spherical(theta, phi, -1.307) for phi, theta in fibonacci_sphere(32)]
            image_plane = ImagePlanes(focal=torch.Tensor([10.0]),
                                      poses=np.stack(poses),
                                      images=plane.view(32, 3, plane.shape[-2], plane.shape[-1]))
            plane_nerf = MultiImageNeRF(image_plane=image_plane)
            nerfs.append(plane_nerf)
            # self.plane_axes = self.plane_axes.to(ray_origins.device)
            out = plane_nerf(sample_coordinates_.squeeze(0), sample_directions.squeeze(0), self.render_network)
            outs.append(out)

        out = {k: torch.stack([out[k] for out in outs]) for k in outs[0].keys()}
        # batch_size = planes.shape[0]
        # depths_coarse = torch.cat(depths, dim=0)
        #
        # out = decoder(sampled_features, sample_directions)
        return out

    def sort_samples(self, all_depths, all_colors, all_densities):
        _, indices = torch.sort(all_depths, dim=-2)
        all_depths = torch.gather(all_depths, -2, indices)
        all_colors = torch.gather(all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
        all_densities = torch.gather(all_densities, -2, indices.expand(-1, -1, -1, 1))
        return all_depths, all_colors, all_densities

    def unify_samples(self, depths1, colors1, densities1, depths2, colors2, densities2):
        all_depths = torch.cat([depths1, depths2], dim = -2)
        all_colors = torch.cat([colors1, colors2], dim = -2)
        all_densities = torch.cat([densities1, densities2], dim = -2)

        _, indices = torch.sort(all_depths, dim=-2)
        all_depths = torch.gather(all_depths, -2, indices)
        all_colors = torch.gather(all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
        all_densities = torch.gather(all_densities, -2, indices.expand(-1, -1, -1, 1))

        return all_depths, all_colors, all_densities

    def sample_stratified(self, ray_origins, ray_start, ray_end, depth_resolution, disparity_space_sampling=False):
        """
        Return depths of approximately uniformly spaced samples along rays.
        """
        N, M, _ = ray_origins.shape
        if disparity_space_sampling:
            depths_coarse = torch.linspace(0,
                                    1,
                                    depth_resolution,
                                    device=ray_origins.device).reshape(1, 1, depth_resolution, 1).repeat(N, M, 1, 1)
            depth_delta = 1/(depth_resolution - 1)
            depths_coarse += torch.rand_like(depths_coarse) * depth_delta
            depths_coarse = 1./(1./ray_start * (1. - depths_coarse) + 1./ray_end * depths_coarse)
        else:
            if type(ray_start) == torch.Tensor:
                depths_coarse = math_utils.linspace(ray_start, ray_end, depth_resolution).permute(1,2,0,3)
                depth_delta = (ray_end - ray_start) / (depth_resolution - 1)
                depths_coarse += torch.rand_like(depths_coarse) * depth_delta[..., None]
            else:
                depths_coarse = torch.linspace(ray_start, ray_end, depth_resolution, device=ray_origins.device).reshape(1, 1, depth_resolution, 1).repeat(N, M, 1, 1)
                depth_delta = (ray_end - ray_start)/(depth_resolution - 1)
                depths_coarse += torch.rand_like(depths_coarse) * depth_delta

        return depths_coarse

    def sample_importance(self, z_vals, weights, N_importance):
        """
        Return depths of importance sampled points along rays. See NeRF importance sampling for more.
        """
        with torch.no_grad():
            batch_size, num_rays, samples_per_ray, _ = z_vals.shape

            z_vals = z_vals.reshape(batch_size * num_rays, samples_per_ray)
            weights = weights.reshape(batch_size * num_rays, -1) # -1 to account for loss of 1 sample in MipRayMarcher

            # smooth weights
            weights = torch.nn.functional.max_pool1d(weights.unsqueeze(1).float(), 2, 1, padding=1)
            weights = torch.nn.functional.avg_pool1d(weights, 2, 1).squeeze()
            weights = weights + 0.01

            z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:])
            importance_z_vals = self.sample_pdf(z_vals_mid, weights[:, 1:-1],
                                             N_importance).detach().reshape(batch_size, num_rays, N_importance, 1)
        return importance_z_vals

    def sample_pdf(self, bins, weights, N_importance, det=False, eps=1e-5):
        """
        Sample @N_importance samples from @bins with distribution defined by @weights.
        Inputs:
            bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
            weights: (N_rays, N_samples_)
            N_importance: the number of samples to draw from the distribution
            det: deterministic or not
            eps: a small number to prevent division by zero
        Outputs:
            samples: the sampled samples
        """
        N_rays, N_samples_ = weights.shape
        weights = weights + eps # prevent division by zero (don't do inplace op!)
        pdf = weights / torch.sum(weights, -1, keepdim=True) # (N_rays, N_samples_)
        cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
        cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1)
                                                                   # padded to 0~1 inclusive

        if det:
            u = torch.linspace(0, 1, N_importance, device=bins.device)
            u = u.expand(N_rays, N_importance)
        else:
            u = torch.rand(N_rays, N_importance, device=bins.device)
        u = u.contiguous()

        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.clamp_min(inds-1, 0)
        above = torch.clamp_max(inds, N_samples_)

        inds_sampled = torch.stack([below, above], -1).view(N_rays, 2*N_importance)
        cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
        bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

        denom = cdf_g[...,1]-cdf_g[...,0]
        denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                             # anyway, therefore any value for it is fine (set to 1 here)

        samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
        return samples