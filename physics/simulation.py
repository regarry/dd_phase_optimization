# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import pi
import torch.fft
import skimage.io
import matplotlib.pyplot as plt
from datetime import datetime
import os
from dd_phase_optimization.data.io import load_tiff_sequence

# nohup python mask_learning.py &> ./logs/01-31-25-09-38.txt &

class TotalVariationLoss(nn.Module):
    """
    A PyTorch module to calculate the anisotropic total variation loss.
    """
    def __init__(self, weight=1.0):
        """
        Args:
            weight (float): The weight for the TV loss in the total loss calculation.
        """
        super(TotalVariationLoss, self).__init__()
        self.weight = weight

    def forward(self, img):
        """
        Calculates the TV loss for the input image tensor.
        
        Args:
            img (torch.Tensor): The input image tensor. 
                                Expected shape: (N, C, H, W)
        
        Returns:
            torch.Tensor: The weighted total variation loss.
        """
        # Get the image dimensions
        #if dim == 4:
    
        if img.dim() == 2:
            # unsqueeze to add batch and channel dimensions
            img = img.unsqueeze(0).unsqueeze(0)
            
        elif img.dim() == 3:
            img = img.unsqueeze(0)
            
        batch_size, channels, height, width = img.shape

        # Calculate the horizontal and vertical differences using slicing
        # This is more memory-efficient than using convolutions.
        horizontal_diff = img[:, :, :, 1:] - img[:, :, :, :-1]
        vertical_diff = img[:, :, 1:, :] - img[:, :, :-1, :]
        
        # Calculate the L1 norm of the differences
        # We sum over all dimensions to get a single scalar loss value.
        tv_loss = torch.sum(torch.abs(horizontal_diff)) + torch.sum(torch.abs(vertical_diff))
        
        # Normalize by the number of elements in the batch and apply the weight
        return self.weight * tv_loss / batch_size

# This function creates a 2D gaussian filter with std=1, without normalization.
# during training this filter is scaled with a random std to simulate different blur per emitter
def gaussian2D_unnormalized(shape=(7, 7), sigma=1.0):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    hV = torch.from_numpy(h).type(torch.FloatTensor)
    return hV



def NextPowerOfTwo(number):
    # Returns next power of two following 'number'
    return 2 ** math.ceil(math.log(number, 2))

class BlurLayer(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.gauss = gaussian2D_unnormalized(shape=(7, 7)).to(device)
        self.std_min = 0.8
        self.std_max = 1.2

    def forward(self, img_4d, device):
        # number of the input PSF images
        Nbatch = img_4d.size(0)
        Nemitters = img_4d.size(1)
        # generate random gaussian blur for each emitter
        RepeatedGaussian = self.gauss.expand(1, Nemitters, 7, 7)
        stds = (self.std_min + (self.std_max - self.std_min) * torch.rand((Nemitters, 1))).to(device)
        MultipleGaussians = torch.zeros_like(RepeatedGaussian)
        for i in range(Nemitters):
            MultipleGaussians[:, i, :, :] = 1 / (2 * pi * stds[i] ** 2) * torch.pow(RepeatedGaussian[:, i, :, :],
                                                                                    1 / (stds[i] ** 2))
        # blur each emitter with slightly different gaussian
        images4D_blur = F.conv2d(img_4d, MultipleGaussians, padding=(2, 2))
        return images4D_blur


# ================================
# Cropping layer: keeps only the center part of the FOV to prevent unnecessary processing
# ==============================
class Croplayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, images4D):
        H = images4D.size(2)
        mid = int((H - 1) / 2)
        images4D_crop = images4D[:, :, mid - 20:mid + 21, mid - 20:mid + 21]
        return images4D_crop


class imgs4dto3d(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, images4D, xyz):
        Nbatch, Nemitters, H, W = images4D.shape[0], images4D.shape[1], images4D.shape[2], images4D.shape[3]
        img = torch.zeros((Nbatch, 1, 200, 200)).type(torch.FloatTensor).to(self.device)
        #img.requires_grad_()
        for i in range(Nbatch):
            for j in range(Nemitters):
                x = int(xyz[i, j, 0])
                y = int(xyz[i, j, 1])
                img[i, 0, x - 15:x + 16, y - 15: y + 16] += images4D[i, j]
        return img


class poisson_noise_approx(nn.Module):
    def __init__(self, device, Nimgs, conv3d):
        super().__init__()
        self.H, self.W = 200, 200
        self.device = device
        self.mean = 3e8
        self.std = 2e8
        self.Nimgs = Nimgs
        self.conv3d = conv3d

    def forward(self, input):
        # number of images
        Nbatch = input.size(0)
        # approximate the poisson noise using CLT and reparameterization
        input = input + 1e5 + (self.std * torch.randn(input.size()) + self.mean).type(torch.FloatTensor).to(self.device)
        input[input <= 0] = 0
        if self.conv3d == True:
            input_poiss = input + torch.tensor(100) * torch.sqrt(input) * torch.randn(Nbatch, 1, self.Nimgs, self.H, self.W).type(
                torch.FloatTensor).to(self.device)
        else:
            #input_poiss = input + torch.tensor(100) * torch.sqrt(input) * torch.randn(Nbatch, self.Nimgs, self.H, self.W).type(
            #    torch.FloatTensor).to(self.device)
            input_poiss = input + torch.tensor(100) * torch.sqrt(input) * torch.randn(Nbatch, 1, self.H, self.W).type(
                torch.FloatTensor).to(self.device) # same noise applied to each z depth
        
        # if torch.isnan(input_poiss).any():
        #     print('yes')

        # result
        return input_poiss


class NoiseLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # As requested, these parameters are directly assigned as attributes
        # and NOT registered as buffers.
        # WARNING: This means they will NOT be automatically moved to the correct
        # device with .to(device) and will NOT be saved/loaded with state_dict().
        # You will need to manually ensure they are on the correct device when used.
        self.quantum_efficiency = torch.tensor(config['quantum_efficiency'], dtype=torch.float64)
        self.dark_current_mean = torch.tensor(config['dark_current_mean'], dtype=torch.float64)
        self.read_noise_std = torch.tensor(config['read_noise_std'], dtype=torch.float64)
        self.camera_gain = torch.tensor(config['camera_gain'], dtype=torch.float64)
        self.camera_max_adu = torch.tensor(config['camera_max_adu'], dtype=torch.float64)

    def forward(self, ideal_image_volume):
        """
        Simulates common camera noise components (Photon, Dark Current, Read Noise)
        on an ideal image volume (PyTorch Tensor), converting the signal to camera ADU.
        Uses differentiable approximations for noise.

        Args:
            ideal_image_volume (torch.Tensor): The ideal, noiseless image volume (e.g., photon counts).
                                               Expected dtype is torch.float.

        Returns:
            torch.Tensor: The noisy image volume, in Analog-to-Digital Units (ADU),
                          with simulated camera noise.
                          The returned tensor will be on the same device as `ideal_image_volume`.
        """

        # Ensure ideal_image_volume is float64 for numerical stability
        ideal_image_volume = ideal_image_volume.to(torch.float64)

        # 1. Convert incident photon signal to mean electrons generated by signal.
        # Ensure 'self.quantum_efficiency' is on the same device as 'ideal_image_volume'
        signal_electrons_mean = ideal_image_volume * self.quantum_efficiency.to(ideal_image_volume.device)

        # 2. Add Dark Current Electrons (Differentiable Approximation of Poisson).
        #    Poisson noise variance = mean. Approximate with Gaussian noise: N(mean, sqrt(mean)^2)
        # Ensure 'self.dark_current_mean' is on the same device
        dark_current_std = torch.sqrt(self.dark_current_mean.to(ideal_image_volume.device))
        # Generate Gaussian noise for dark current with the correct mean and std
        dark_current_noise = torch.randn(ideal_image_volume.shape, device=ideal_image_volume.device, dtype=torch.float64) * dark_current_std
        # Ensure the mean is added to the noise after it's scaled by std
        dark_current_electrons = self.dark_current_mean.to(ideal_image_volume.device) + dark_current_noise

        # 3. Calculate the total mean electrons accumulated per pixel.
        total_electrons_mean = signal_electrons_mean + dark_current_electrons
        total_electrons_mean = torch.clamp_min(total_electrons_mean, 0.0) # Ensure non-negative mean for sqrt

        # 4. Apply Photon Noise (Shot Noise) - Differentiable Approximation of Poisson.
        #    Variance of photon noise is equal to its mean.
        photon_noise_std = torch.sqrt(total_electrons_mean)
        photon_noise = torch.randn(ideal_image_volume.shape, device=ideal_image_volume.device, dtype=torch.float64) * photon_noise_std
        # The total signal + photon noise is the mean signal + Gaussian noise scaled by sqrt(mean)
        noisy_electrons_poisson_approx = total_electrons_mean + photon_noise

        # 5. Add Read Noise - remains Gaussian.
        #    This is independently generated Gaussian noise. Its generation doesn't depend on
        #    `ideal_image_volume` in a way that breaks the graph for `ideal_image_volume` itself.
        # Ensure 'self.read_noise_std' is on the same device
        read_noise = torch.randn(ideal_image_volume.shape, device=ideal_image_volume.device, dtype=torch.float64) * self.read_noise_std.to(ideal_image_volume.device)
        
        # 6. Sum all electron components, including all noise types, before ADC.
        #    Here we sum the *approximated* noisy electron counts.
        electrons_after_noise = noisy_electrons_poisson_approx + read_noise

        # 7. Apply Camera Gain (Analog-to-Digital Conversion).
        # Ensure 'self.camera_gain' is on the same device
        noisy_image_adu = electrons_after_noise * self.camera_gain.to(ideal_image_volume.device)

        # 8. Clip values to the valid ADU range.
        # Ensure 'self.camera_max_adu' is on the same device
        noisy_image_adu = torch.clamp(noisy_image_adu, min=0.0, max=self.camera_max_adu.to(ideal_image_volume.device))

        return noisy_image_adu.type(torch.float32) # Convert back to float32 for consistency

class Normalize01(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, result_noisy):
        Nbatch = result_noisy.size(0)
        result_noisy_01 = torch.zeros_like(result_noisy)
        #min_val = (result_noisy[0, 0, :, :]).min()
        min_val = 0
        #max_val = (result_noisy[:, :, :, :]).max()
        #print(max_val)
        max_val = 4e9
        # if torch.isnan(result_noisy).any():
        #     print('yes')
        result_noisy[result_noisy <= 10] = 1
        result_noisy[result_noisy >= max_val] = max_val
        # for i in range(Nbatch):

        #     result_noisy_01[i, :, :, :] = (result_noisy[i, :, :, :] - min_val) / (max_val - min_val)
        result_noisy_01 = (result_noisy) / (max_val)
        return result_noisy_01


# ==================================================
# Physical encoding layer, from 3D to 2D:
# this layer takes in the learnable parameter "mask"
# and output the resulting 2D image corresponding to the emitters location.
# ===================================================

class OpticsSimulation(nn.Module):
    def __init__(self, config):
        super(OpticsSimulation, self).__init__()
        #unpack the config
        self.config = config
        self.bfp_dir = config["bfp_dir"]

        self.px = config['px']  #the pixel size used
        self.wavelength = config['wavelength']
        self.focal_length_1 = config['focal_length_1']
        #psf_width_pixels = config['psf_width_pixels']
        #psf_edge_remove = config['psf_edge_remove']
        laser_beam_FWHC = config['laser_beam_FWHC']
        self.refractive_index = config['refractive_index']
        #max_defocus = config['max_defocus']
        data_path = self.config['data_path']
        image_volume = config['image_volume']
        psf_keep_radius = config['psf_keep_radius']
        training_results_dir = config['training_results_dir']
        #device = config['device']
        self.lens_approach = config['lens_approach']
        if self.lens_approach == 'fresnel':
            max_intensity = config.get('max_intensity_fresnel', 5.0e+4)
        elif self.lens_approach == 'convolution':
            max_intensity = config.get('max_intensity_conv', 8.0e+10)
        else:
            max_intensity = config.get('max_intensity_conv', 8.0e+10)    
        #self.device = device
        self.psf_keep_radius = psf_keep_radius
        self.max_intensity = torch.tensor(max_intensity)
        self.counter = 0
        self.focal_length_2 = config['focal_length_2']  # for 4f approach
        self.focal_length_3 = config['focal_length_3']  # for 9f approach
        self.focal_length_4 = config['focal_length_4']  # for 9f approach
        self.focal_length_5 = config['focal_length_5']  # for 9f approach
        self.illumination_scaling_factor = config.get('illumination_scaling_factor')  # scaling factor for the illumination
        self.camera_max_adu = config.get('camera_max_adu')  # maximum ADU for the camera
        self.lenless_prop_distance = config.get('lenless_prop_distance', 1.0e-3)  # distance for lensless propagation
            
        #self.power_2 = config['power_2']
        #self.pad_to_power_2 = self.power_2-N
        self.pad = 500
        self.datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.conv3d = config.get('conv3d', False)
        self.aperature = config.get('aperature', False)
        self.phase_mask_upsample_factor = config.get('phase_mask_upsample_factor', 1)  # scale factor for the phase mask upsampling
        self.N = config['N']
        self.scale_factor = config['scale_factor']
        self.phase_mask_pixel_size = config['phase_mask_pixel_size']    
        #self.N = self.phase_mask_upsample_factor * self.phase_mask_pixel_size # grid size for the physics calculations
        #self.z_spacing = config.get('z_spacing', 0)
        #self.z_img_mode = config.get('z_img_mode', 'edgecenter')
        #if self.z_img_mode == 'everyother':
            #self.z_depth_list = list(range(0,self.Nimgs,2))
        #if self.z_img_mode == 'edgecenter' and self.z_spacing > 0:
        #    self.z_depth_list = [-self.z_spacing, 0, self.z_spacing]
        #else:
        #    self.z_depth_list = list(range(-self.z_spacing,self.z_spacing+1))
        
        self.Nimgs = config['Nimgs']
        self.z_depth_list = config['z_depth_list']

        # to transer the physical size of the imaging volume
        self.image_volume_size_px = image_volume

        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        x = list(range(-self.N // 2, self.N // 2))
        y = list(range(-self.N // 2, self.N // 2))
        [X, Y] = np.meshgrid(x, y)
        X = X * self.px
        Y = Y * self.px
        
        xx = list(range(-self.N + 1, self.N + 1))
        yy = list(range(-self.N + 1, self.N + 1))
        [XX, YY] = np.meshgrid(xx, yy)
        self.XX = XX * self.px
        self.YY = YY * self.px

        
        # initialize phase mask
        incident_gaussian = 1 * np.exp(-(np.square(X) + np.square(Y)) / (2 * laser_beam_FWHC ** 2))
        incident_gaussian = torch.from_numpy(incident_gaussian).type(torch.FloatTensor)
        
        # --- Lens 1 (B1) and Propagation Kernel 1 (Q1) ---
        
        C1 = (np.pi / (self.wavelength * self.focal_length_1) * (np.square(X) + np.square(Y))) % (
            2 * np.pi)  # lens function lens as a phase transformer
        B1 = np.exp(-1j * C1)
        B1 = torch.from_numpy(B1).type(torch.cfloat)
    
        # refractive index is of the medium 
        #Q1 = np.exp(1j * (np.pi * self.refractive_index / (self.wavelength * self.focal_length_1)) * (
        #            np.square(XX) + np.square(YY)))  # Fresnel diffraction equation at distance = focal length
        #self.Q1 = torch.from_numpy(Q1).type(torch.cfloat).to(device)

        # --- Lens 2 (B2) and Propagation Kernel 2 (Q2) ---
        C2 = (np.pi / (self.wavelength * self.focal_length_2) * (np.square(X) + np.square(Y))) % (2 * np.pi)
        B2 = np.exp(-1j * C2)
        B2 = torch.from_numpy(B2).type(torch.cfloat)
        
        # lens 3
        C3 = (np.pi / (self.wavelength * self.focal_length_3) * (np.square(X) + np.square(Y))) % (2 * np.pi)
        B3 = np.exp(-1j * C3)
        B3 = torch.from_numpy(B3).type(torch.cfloat)
        
        # lens 4
        C4 = (np.pi / (self.wavelength * self.focal_length_4) * (np.square(X) + np.square(Y))) % (2 * np.pi)
        B4 = np.exp(-1j * C4)
        B4 = torch.from_numpy(B4).type(torch.cfloat)
        
        # lens 5
        C5 = (np.pi / (self.wavelength * self.focal_length_5) * (np.square(X) + np.square(Y))) % (2 * np.pi)
        B5 = np.exp(-1j * C5)
        B5 = torch.from_numpy(B5).type(torch.cfloat)
        

        ## Q2 for propagation over self.focal_length_2
        #Q2_val = np.exp(1j * (np.pi * self.refractive_index / (self.wavelength * self.focal_length_2)) * (np.square(self.XX) + np.square(self.YY)))
        #self.Q2 = torch.from_numpy(Q2_val).type(torch.cfloat).to(device)
        
        # angular specturm
        k = 2 * self.refractive_index * np.pi / self.wavelength
        self.k = k
       
        # phy_x = self.N * self.px  # physical width (meters)
        # phy_y = self.N * self.px  # physical length (meters)
        # obj_size = [self.N, self.N]
        # # generate meshgrid
        # Fs_x = obj_size[1] / phy_x
        # Fs_y = obj_size[0] / phy_y
        # dFx = Fs_x / obj_size[1]
        # dFy = Fs_y / obj_size[0]
        # Fx = np.arange(-Fs_x / 2, Fs_x / 2, dFx)
        # Fy = np.arange(-Fs_y / 2, Fs_y / 2, dFy)
        # alpha and beta (wavenumber components) 
        
        # Generate spatial frequencies using np.fft.fftfreq
        # This function correctly produces N frequency bins.
        # The 'd' argument is the spatial sampling interval (px).
        Fx_raw = np.fft.fftfreq(self.N, d=self.px)
        Fy_raw = np.fft.fftfreq(self.N, d=self.px)

        # Shift the zero-frequency component to the center of the array
        # This reorders the frequencies to be from -Fs/2 to Fs/2 - dF
        Fx = np.fft.fftshift(Fx_raw)
        Fy = np.fft.fftshift(Fy_raw)
        alpha = self.refractive_index * self.wavelength * Fx
        beta = self.refractive_index * self.wavelength * Fy
        [ALPHA, BETA] = np.meshgrid(alpha, beta)
        # go over and make sure that it is not complex
        gamma_cust = np.zeros_like(ALPHA)
        for i in range(len(ALPHA)):
            for j in range(len(ALPHA[0])):
                if 1 - np.square(ALPHA[i][j]) - np.square(BETA[i][j]) > 0:
                    gamma_cust[i, j] = np.sqrt(1 - np.square(ALPHA[i][j]) - np.square(BETA[i][j]))
        gamma_cust = torch.from_numpy(gamma_cust).type(torch.FloatTensor)


        # read bead defocus stack
        # Load all TIFF images sequentially 
        bead_defocus_dir = os.path.join(training_results_dir, 'bead_defocus_imgs')
        self.bead_defocus_imgs = load_tiff_sequence(bead_defocus_dir)
        # tiff_files = sorted([f for f in os.listdir(data_path) if f.lower().endswith('.tiff')])
        # print(f"Found {len(tiff_files)} TIFF files in {data_path}:")
        # for idx, fname in enumerate(tiff_files):
        #     img_path = os.path.join(data_path, fname)
        #     img = skimage.io.imread(img_path)
        #     self.psf_imgs.append(img)
        #     print(f"Loaded [{idx}] {fname} with shape {img.shape}")
        # print(f"Total images loaded into self.imgs: {len(self.psf_imgs)}")
        # total_bytes = sum(img.nbytes for img in self.psf_imgs)
        # print(f"self.psf_imgs memory size: {total_bytes / 1024**2:.2f} MB")
            #plt.imshow(img[center_img + -psf_keep_radius:center_img+psf_keep_radius+1,\
            #                           center_img + -psf_keep_radius:center_img+psf_keep_radius+1])
            #plt.axis('off')  # Turn off axis labels
            #plt.show()



        #self.blur = BlurLayer(device)
        #self.crop = Croplayer()
        #self.img4dto3d = imgs4dto3d(device)
        self.noise = NoiseLayer(config)
        #self.norm01 = Normalize01()
        # Convert the list of images into a single 3D numpy array first
        imgs_array = np.array(self.bead_defocus_imgs) 

        
        #---- REGISTER BUFFERS ---
        
        # 1. Register the Incident Gaussian Beam
        self.register_buffer('incident_gaussian', incident_gaussian)
        
        self.register_buffer('gpu_psfs', torch.from_numpy(imgs_array).float())
        
        self.register_buffer('B1', B1)
        self.register_buffer('B2', B2)
        self.register_buffer('B3', B3)
        self.register_buffer('B4', B4)
        self.register_buffer('B5', B5)
        
        
        # 3. Register Angular Spectrum Propagation Constant
        self.register_buffer('gamma_cust', gamma_cust)

        # 4. Register the PSF Image Library
        # This converts the list of images (self.imgs) into one 3D GPU tensor
        # Shape: [Num_PSFs, Height, Width]
        psf_stack = np.array(self.bead_defocus_imgs).astype('float32')
        self.register_buffer('gpu_psfs', torch.from_numpy(psf_stack))

        # 5. Register the Z-Depth List
        # This allows the forward loop to find PSF indices on the GPU
        z_depths = torch.tensor(self.config['z_depth_list']).long()
        self.register_buffer('z_depth_buffer', z_depths)

        # 6. Pre-calculate k (Wave number)
        self.k = 2 * self.config['refractive_index'] * np.pi / self.wavelength
        
    def circular_aperature(self, arr):
        """
        Sets elements outside a centered circle to zero for a PyTorch tensor.

        Args:
        arr: A 2D square PyTorch tensor.

        Returns:
        A new PyTorch tensor with elements outside the circle set to zero.
        """
        h = arr.shape[0]
        if arr.shape[1] != h:
            raise ValueError("Input tensor must be square.")

        center_x = (h - 1) / 2
        center_y = (h - 1) / 2
        radius = h / 2

        # Create a meshgrid of coordinates
        x = torch.arange(h)
        y = torch.arange(h)
        xx, yy = torch.meshgrid(x, y, indexing='ij')  # Use indexing='ij' for correct meshgrid

        # Calculate the distance from each point to the center
        distances = torch.sqrt((xx - center_x)**2 + (yy - center_y)**2)

        # Create a mask where True indicates points inside the circle
        mask = distances <= radius

        # Apply the mask to the array
        result = arr * mask

        return result
    
    @staticmethod
    def _visualize_step(title, tensor):
        arr = tensor.cpu().detach().numpy()
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.title(f"{title} - Phase")
        plt.imshow(np.angle(arr), cmap='twilight')
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.title(f"{title} - Magnitude")
        plt.imshow(np.abs(arr), cmap='viridis')
        plt.colorbar()
        plt.tight_layout()
        plt.show(block=False)
        
    @staticmethod
    def _visualize_real(title, tensor):
        arr = tensor.cpu().detach().numpy()
        plt.figure(figsize=(6, 5))
        plt.title(title)
        plt.imshow(arr, cmap='viridis')
        plt.colorbar()
        plt.tight_layout()
        plt.show(block=False)
        
    def angular_spectrum_propagation(self, input_field, z, debug=False):
        """
        Angular spectrum propagation for a given output_layer and z pixel distance.

        Args:
            output_layer (torch.Tensor): The complex field to propagate.
            z (float or int): The z pixel distance for propagation.

        Returns:
            torch.Tensor: The propagated intensity (real-valued).
        """
        # Step 1: FFT2
        fft2_field = torch.fft.fft2(input_field)
        if debug:
            self._visualize_step("FFT2(output_layer)", fft2_field)
            
            
        # Step 2: Propagation kernel
        prop_kernel = torch.exp(1j * self.k * self.gamma_cust * z * self.px)
        
        if debug:
         self._visualize_step("Propagation Kernel", prop_kernel)
         
         # Step 3: FFTSHIFT
        prop_kernel_shift = torch.fft.fftshift(prop_kernel)
        if debug:
            self._visualize_step("FFTSHIFT(prop_kernel)", prop_kernel_shift)
         
        # Step 4: Multiply
        mult = fft2_field * prop_kernel_shift
        if debug:
            self._visualize_step("Multiplied Spectrum", mult)
            
        # # Step 5: IFFTSHIFT
        # ifftshifted = torch.fft.ifftshift(mult)
        # if debug:
        #     self._visualize_step("IFFTSHIFT(Multiplied)", ifftshifted)
            
        # Step 6: IFFT2
        U1 = torch.fft.ifft2(mult)
        if debug:
            self._visualize_step("IFFT2", U1)
            self._visualize_real("Intensity (real(U1 * conj(U1)))", torch.real(U1 * torch.conj(U1)))
        
        #U1_old = torch.fft.ifft2(
        #        torch.fft.fft2(output_layer)
        #        * torch.fft.fftshift(torch.exp(1j * self.k * self.gamma_cust * z * self.px))
        #)
        
        #if debug:
        #    self._visualize_step("U1_old", U1_old)
        #    self._visualize_real("Intensity (real(U1_old * conj(U1_old)))", torch.real(U1_old * torch.conj(U1_old)))
        #U1 = torch.real(U1 * torch.conj(U1))
        return U1
    
    def fresnel_propagation(self, input_img, z, debug=False):
        """
        Fresnel propagation for a given input_img and z pixel distance.

        Args:
            input_img (torch.Tensor): The complex field to propagate.
            z (float or int): The z pixel distance for propagation.
            debug (bool, optional): If True, visualizes intermediate steps for debugging. Defaults to False.

        Returns:
            torch.Tensor: The propagated intensity (real-valued).
        """
        
        Q = np.exp(1j * self.k / (2 * z * self.px) * (
                    np.square(self.XX) + np.square(self.YY)))  # Fresnel diffraction equation
        Q = torch.from_numpy(Q).type(torch.cfloat).to(input_img.device)

        if debug:
            # Visualize the phase and magnitude of Q for debugging
            Q_np = Q.cpu().numpy()
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.title("Phase of Q")
            plt.imshow(np.angle(Q_np), cmap='twilight')
            plt.colorbar()
            plt.subplot(1, 2, 2)
            plt.title("Magnitude of Q")
            plt.imshow(np.abs(Q_np), cmap='viridis')
            plt.colorbar()
            plt.tight_layout()
            plt.show(block=False)
        
        Uo_pad = F.pad(input_img, (0, self.N, 0, self.N), 'constant', 0) # padded to interpolate with fft
        
        if debug:
            # Visualize the phase and magnitude of Uo_pad for debugging
            Uo_pad_np = Uo_pad.cpu().detach().numpy()
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.title("Phase of Uo_pad")
            plt.imshow(np.angle(Uo_pad_np), cmap='twilight')
            plt.colorbar()
            plt.subplot(1, 2, 2)
            plt.title("Magnitude of Uo_pad")
            plt.imshow(np.abs(Uo_pad_np), cmap='viridis')
            plt.colorbar()
            plt.tight_layout()
            plt.show(block=False)

        U1 = torch.fft.ifft2(torch.fft.fft2(Uo_pad) * torch.fft.fft2(Q)) # light directly incident of the lens
        
        if debug:
            # Visualize the phase and magnitude of U1 for debugging
            U1_np = U1.cpu().detach().numpy()
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.title("Phase of U1")
            plt.imshow(np.angle(U1_np), cmap='twilight')
            plt.colorbar()
            plt.subplot(1, 2, 2)
            plt.title("Magnitude of U1")
            plt.imshow(np.abs(U1_np), cmap='viridis')
            plt.colorbar()
            plt.tight_layout()
            plt.show(block=False)
        #check if there are two singeltons in U1
        
        U1_cropped = U1[...,-self.N:, -self.N:]
        U1_intensity = torch.real(U1_cropped * torch.conj(U1_cropped))
        
        if debug:
            plt.figure(figsize=(6, 5))
            plt.title("U1 Intensity (Fresnel Propagation)")
            plt.imshow(U1_intensity.cpu().detach().numpy(), cmap='viridis')
            plt.colorbar()
            plt.tight_layout()
            plt.show(block=False)
            breakpoint()  # Pause for debugging

        return U1_intensity
    
    def against_lens(self, phase_mask):
        Ta = torch.exp(1j * phase_mask) # amplitude transmittance (in our case the slm reflectance)
        Ta = Ta[None, None, :] 
        Ul = self.incident_gaussian * Ta # light directly behind the SLM (or in our case reflected from the SLM)
        Ul_prime = Ul * self.B1 # light after the lens
        output_layer = self.fresnel_propagation(Ul_prime, self.focal_length_1/self.px, debug=False) # light at the back focal plane of the lens
        #Ul_prime_pad = F.pad(Ul_prime, (self.N//2, self.N//2, self.N//2, self.N//2), 'constant', 0) # padded to interpolate with fft
        #Uf = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fft2(Ul_prime_pad) * torch.fft.fft2(self.Q1))) # light at the back focal plane of the lens   
        #output_layer = Uf[:, :, self.N // 2:3 * self.N // 2, self.N // 2:3 * self.N // 2]
        return output_layer

    def fourier_lens(self, phase_mask):
        Ta = torch.exp(1j * phase_mask) # amplitude transmittance (in our case the slm reflectance)
        Uo = self.incident_gaussian * Ta # light directly behind the SLM (or in our case reflected from the SLM)
        Ul = self.angular_spectrum_propagation(Uo, self.focal_length_1/self.px) # light directly infront of the lens
        Ul_prime = Ul * self.B1 # light after the lens
        Uf = self.angular_spectrum_propagation(Ul_prime, self.focal_length_1/self.px) # light at the back focal plane of the lens
        output_layer = Uf[None, None, :, :] # light at the back focal plane of the lens
        return output_layer

    def lensless(self, phase_mask):
        Ta = torch.exp(1j * phase_mask) # amplitude transmittance (in our case the slm reflectance)
        Ta = Ta[None, None, :]
        Uo = self.incident_gaussian * Ta # light directly behind the SLM (or in our case reflected from the SLM)
        output_layer = Uo
        return output_layer

    def fourf(self, phase_mask):
        Ta = torch.exp(1j * phase_mask).to(phase_mask.device) # amplitude transmittance (in our case the slm reflectance)
        Uo = self.incident_gaussian * Ta # light directly behind the SLM (or in our case reflected from the SLM)
        Ul1 = self.angular_spectrum_propagation(Uo, self.focal_length_1/self.px, debug = False) # light directly infront of the lens
        Ul1_prime = Ul1 * self.B1 # light after the lens
        Ul2 = self.angular_spectrum_propagation(Ul1_prime, (self.focal_length_1+self.focal_length_2)/self.px, debug = False) # light at the back focal plane of the lens
        #Ul2 = self.angular_spectrum_propagation(Uf1, self.focal_length_2) # light directly infront of the lens
        Ul2_prime = Ul2 * self.B2 # light after the 2nd lens
        Uf2 = self.angular_spectrum_propagation(Ul2_prime, self.focal_length_2/self.px, debug = False) # light at the back focal plane of the lens
        output_layer = Uf2[None, None, :, :] # light at the back focal plane of the lens
       
        return output_layer
    
    def lazy_fourf(self, phase_mask):
        Ta = torch.exp(1j * phase_mask).to(phase_mask.device) # amplitude transmittance (in our case the slm reflectance)
        Uo = self.incident_gaussian * Ta # light directly behind the SLM (or in our case reflected from the SLM)
        U1 = F.interpolate(
            Uo[None, None, :, :], 
            scale_factor= self.scale_factor, 
            mode='bilinear', 
            align_corners=False
            )   
        return U1
    
    def fivef(self, phase_mask):
        Ta = torch.exp(1j * phase_mask) # amplitude transmittance (in our case the slm reflectance)
        Uo = self.incident_gaussian * Ta # light directly behind the SLM (or in our case reflected from the SLM)
        Ul1 = self.angular_spectrum_propagation(Uo, self.focal_length_1/self.px, debug = False) # light directly infront of the lens
        Ul1_prime = Ul1 * self.B1 # light after the lens
        Ul2 = self.angular_spectrum_propagation(Ul1_prime, (self.focal_length_1+self.focal_length_2)/self.px, debug = False) # light at the back focal plane of the lens
        #Ul2 = self.angular_spectrum_propagation(Uf1, self.focal_length_2) # light directly infront of the lens
        Ul2_prime = Ul2 * self.B2 # light after the 2nd lens
        Ul3 = self.angular_spectrum_propagation(Ul2_prime, (self.focal_length_2+self.focal_length_3)/self.px, debug = False) # light at the back focal plane of the lens
        Ul3_prime = Ul3 * self.B3 # light after the 3rd lens
        Uf3 = self.angular_spectrum_propagation(Ul3_prime, self.focal_length_3/self.px, debug = False) # light at the back focal plane of the lens
        output_layer = Uf3[None, None, :, :] # light at the back focal plane of the lens
       
        return output_layer
    
    def ninef(self, phase_mask):
        Ta = torch.exp(1j * phase_mask) # amplitude transmittance (in our case the slm reflectance)
        Uo = self.incident_gaussian * Ta # light directly behind the SLM (or in our case reflected from the SLM)
        Ul1 = self.angular_spectrum_propagation(Uo, self.focal_length_1/self.px, debug = False) # light directly behind the lens
        Ul1_prime = Ul1 * self.B1 # light after the lens
        Ul2 = self.angular_spectrum_propagation(Ul1_prime, (self.focal_length_1+self.focal_length_2)/self.px, debug = False) 
        Ul2_prime = Ul2 * self.B2 # light after the 2nd lens
        Ul3 = self.angular_spectrum_propagation(Ul2_prime, (self.focal_length_2+self.focal_length_3)/self.px, debug = False) 
        Ul3_prime = Ul3 * self.B3 # light after the 3rd lens
        Ul4 = self.angular_spectrum_propagation(Ul3_prime, (self.focal_length_3+self.focal_length_4)/self.px, debug = False) 
        Ul4_prime = Ul4 * self.B4 # light after the 4th ens
        Ul5 = self.angular_spectrum_propagation(Ul4_prime, (self.focal_length_4+self.focal_length_5)/self.px, debug = False) 
        Ul5_prime = Ul5 * self.B5 # light after the 5th lens
        Uf5 = self.angular_spectrum_propagation(Ul5_prime, self.focal_length_5/self.px, debug = False) 
        output_layer = Uf5[None, None, :, :] 
       
        return output_layer
    
    
    def test_fourf(self, input_field):
        Ul1 = self.angular_spectrum_propagation(input_field, self.focal_length_1/self.px, debug = False) # light directly infront of the lens
        Ul1_prime = Ul1 * self.B1 # light after the lens
        Ul2 = self.angular_spectrum_propagation(Ul1_prime, (self.focal_length_1 + self.focal_length_2)/self.px, debug = False) # light at the back focal plane of the lens
        #Ul2 = self.angular_spectrum_propagation(Uf1, self.focal_length_2) # light directly infront of the lens
        Ul2_prime = Ul2 * self.B2 # light after the 2nd lens
        Uf2 = self.angular_spectrum_propagation(Ul2_prime, self.focal_length_2/self.px, debug = False)
        return Uf2
    

    @staticmethod
    def standardize_and_scale_to_uint16(img: np.ndarray) -> np.ndarray:
        """
        Standardizes an image (zero mean, unit variance) and rescales to uint16 [0, 65535].
        Args:
            img (np.ndarray): The input image array.
        Returns:
            np.ndarray: The standardized and scaled image as uint16.
        """
        # If tensor, convert to numpy
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        img_float = img.astype(np.float64)
        mean = img_float.mean()
        std = img_float.std()
        if std > 0:
            img_float = (img_float - mean) / std
        else:
            img_float = img_float - mean  # Avoid division by zero

        # Min-max scale to [0, 1]
        img_float -= img_float.min()
        max_val = img_float.max()
        if max_val > 0:
            img_float /= max_val

        # Scale to uint16
        return (img_float * 65535).astype(np.uint16)
    
    def generate_beam_cross_section(self, initial_field: np.ndarray, output_folder: str, z_range_px: tuple, y_slice_px: tuple, asm: bool = True) -> np.ndarray:
        """
        Generates a 2D cross-section of the beam by stacking 1D slices along the z-axis.
        
        Args:
            initial_field (np.ndarray): The complex field at z=0.
            output_folder (str): root Folder to save 2D intensity profiles at each z-step.
            z_range_px (tuple): (min, max) propagation distance in pixels.
            y_slice_px (tuple): (min, max) slice of the y-axis in pixels.
            
        Returns:
            np.ndarray: A 2D array representing the beam's cross-section (ZY plane).
        """
        output_beam_sections_dir = os.path.join(output_folder, "beam_sections")
        os.makedirs(output_beam_sections_dir, exist_ok=True)
        z_min_px, z_max_px, z_step = z_range_px
        y_min_px, y_max_px = y_slice_px
        
        num_z_steps = len(range(z_min_px, z_max_px, z_step))
        # Initialize the cross-section profile array accounting for step size

        cross_section_profile = np.zeros((num_z_steps, y_max_px - y_min_px))
        # init empty tensor to hold z min to z max with z step of width and height depednig on intial_field size'
        height, width = initial_field.shape
        intensity_at_z = torch.zeros((num_z_steps, height, width))

        print(f"Generating beam cross-section for {num_z_steps} slices...")
        if asm:
            print("using angular spectrum")
        else:
            print("using fresnel approximation")
            
        for i, z_px in enumerate(range(z_min_px, z_max_px, z_step)):
            if z_px == 0:
                z_px = 1.0e-3 # avoid zero division
        
            # Propagate field and get intensity
            if asm:
                output_field = self.angular_spectrum_propagation(initial_field, z_px)
                intensity_at_z[i] = torch.real(output_field * torch.conj(output_field))
            else:
                output_field = self.fresnel_propagation(initial_field, z_px)
                intensity_at_z[i] = torch.real(output_field * torch.conj(output_field))

            # Extract the central 1D slice along the y-axis
            center_row_idx = intensity_at_z[i].shape[0] // 2
            center_col_idx = intensity_at_z[i].shape[1] // 2
            cross_section_profile[i,:] = intensity_at_z[i][ center_row_idx, center_col_idx + y_min_px : center_col_idx + y_max_px]
        
        # normalize intensity_at_z to uint16
        #intensity_at_z = self.normalize_to_uint16(intensity_at_z[:])
        intensity_at_z = self.standardize_and_scale_to_uint16(intensity_at_z[:])
        for i, z_px in enumerate(range(z_min_px, z_max_px, z_step)):
            # Save the full 2D intensity profile at the current z-step
            z_mm = z_px*self.px*1.0e3
            save_path = os.path.join(output_beam_sections_dir, f'intensity_{i:04d}_{z_mm:.2f}.tiff')
            #print(f'{i} {z_px}')
            skimage.io.imsave(save_path, intensity_at_z[i])
        
        def visualize_column_sums(P: np.ndarray, output_folder) -> np.ndarray:
            """
            Calculates the sum of columns for each image in a 3D NumPy array
            and visualizes the results as a new "image".

            Args:
                P (np.ndarray): A 3D NumPy array of shape (num_images, height, width),
                                where 'height' and 'width' are typically 2048.
                                The data type is expected to be uint16.

            Returns:
                np.ndarray: A 2D NumPy array of shape (num_images, width), where
                            each row represents an original image and its column sums.
                            The data type will be uint32 to prevent overflow.
            """
            if P.ndim != 3:
                raise ValueError("Input array P must be 3-dimensional (num_images, height, width).")
            if P.shape[1] != 2048 or P.shape[2] != 2048:
                print(f"Warning: Expected image dimensions of 2048x2048, but got {P.shape[1]}x{P.shape[2]}.")

            # Sum columns for each image.
            # Axis=2 refers to the column dimension in a (num_images, rows, columns) array.
            # Using uint32 to prevent potential overflow as sums can exceed uint16 max.
            column_sums_per_image = np.sum(P, axis=1, dtype=np.uint32)
            column_visual_path = os.path.join(output_folder,'column_sums_visualization.png')
            # Visualize the new "image"
            plt.figure(figsize=(12, 6))
            plt.imshow(column_sums_per_image, aspect='auto', cmap='viridis')
            plt.colorbar(label='Summed Column Value')
            plt.title('Column Sums for Each Image (New "Image")')
            plt.xlabel('Column Index')
            plt.ylabel('Original Image Index')
            plt.savefig(column_visual_path) # Save the plot as an image file
            plt.close() # Close the plot to prevent it from displaying immediately in some environments

            return column_sums_per_image
        
        column_sums_image = visualize_column_sums(intensity_at_z, output_folder)
    
        # save cross_section_profile as tiff in 32 bit float format
        #cross_section_save_path = os.path.join(output_folder, 'cross_section_profile.tiff')
        #skimage.io.imsave(cross_section_save_path, cross_section_profile.astype(np.float32))
        print("Cross-section generation complete.")
        return cross_section_profile, column_sums_image

    @staticmethod
    def expand_matrix_kron_torch(matrix, scale_factor):
        """
        Expands a matrix by a given scale factor using the Kronecker product.
        This function works with torch.autograd.
        """
        # Create the block tensor of ones.
        # The dtype must match the input matrix's dtype for autograd to work smoothly.
        block = torch.ones(scale_factor, scale_factor, dtype=matrix.dtype, device=matrix.device)
        
        # Compute the Kronecker product.
        expanded_matrix = torch.kron(matrix, block)
        return expanded_matrix
    
    def upsample_pixel_size(matrix, scale_factor):
        return matrix.repeat_interleave(scale_factor, dim=0).repeat_interleave(scale_factor, dim=1)

    def forward(self, phase_mask, xyz):
        Nbatch, Nemitters = xyz.shape[0], xyz.shape[1]
        if not self.lens_approach == 'lazy_4f':
            if self.phase_mask_upsample_factor > 1:
                phase_mask_upsampled = self.expand_matrix_kron_torch(phase_mask, self.phase_mask_upsample_factor)
            else:
                phase_mask_upsampled = phase_mask
        """    
        if self.lens_approach == 'fresnel':
            mask_param = self.incident_gaussian * torch.exp(1j * phase_mask_upsampled)
            mask_param = mask_param[None, None, :]
            #multiply the mask with the phase function of the lens (B1)
            B1 = self.B1 * mask_param

            # AG - Need to check if the FFT will be faster if padded to 1024
            # pad_to_power_2 = NextPowerOfTwo(B1.shape[0])-B1.shape[0]
            
            E1 = F.pad(B1, (self.N//2, self.N//2, self.N//2, self.N//2), 'constant', 0)
            # Goodman book equation 4-14, convolution method - lens kernel (Q1) and the image after mask and lens function (E2)
            E2 = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fft2(E1) * torch.fft.fft2(self.Q1)))
            output_layer = E2[:, :, self.N // 2:3 * self.N // 2, self.N // 2:3 * self.N // 2]
        """
        if self.lens_approach == 'against_lens':
            output_layer = self.against_lens(phase_mask_upsampled)
            
        elif self.lens_approach == 'fourier_lens' or self.lens_approach == 'convolution':
            output_layer = self.fourier_lens(phase_mask_upsampled)
            
        elif self.lens_approach == 'lensless':
            output_layer = self.lensless(phase_mask_upsampled)
            # propagate to prop distance  pixels infront of lens
            output_layer = self.angular_spectrum_propagation(output_layer, self.lenless_prop_distance/self.px) # infront of lens
            
        elif self.lens_approach == '5f':
            output_layer = self.fivef(phase_mask_upsampled)
                
        elif self.lens_approach == '4f':
            output_layer = self.fourf(phase_mask_upsampled)
            
        elif self.lens_approach == 'lazy_4f':
            output_layer = self.lazy_fourf(phase_mask) 
               
        elif self.lens_approach == '9f':
            output_layer = self.ninef(phase_mask_upsampled)
            
        else:
            raise ValueError('lens approach not supported')
            
        #if self.counter == 0 and not self.training:  
        debug = False
        if debug:  # debugging 4f 
            from data_utils import save_png
            save_png(phase_mask, self.bfp_dir, "input phase mask", self.config)
            #save_output_layer(output_layer, self.bfp_dir, self.lens_approach, self.counter, self.datetime, self.config)

        self.counter += 1
        
        if self.conv3d == False:
            # make a 4D tensor to store the 2D images
            imgs3D = torch.zeros(Nbatch, self.Nimgs, self.image_volume_size_px[0], self.image_volume_size_px[1]).type(torch.FloatTensor).to(phase_mask.device)
        elif self.conv3d == True and self.Nimgs > 1:
            #make a 5D tensor to store the 3D images
            imgs3D = torch.zeros(Nbatch, 1, self.Nimgs, self.image_volume_size_px[0], self.image_volume_size_px[1]).type(torch.FloatTensor).to(phase_mask.device)
        else:
            raise ValueError('Nimgs must be > 1 to use conv3d')
        # find all of the unique x values and compute intensity where x = xyz[i, j, 0].type(torch.LongTensor) - self.image_volume_size_px[0]//2 
        # x = xyz[i, j, 0].type(torch.LongTensor) - self.image_volume_size_px[0]//2
        
        for l in range(self.Nimgs):
            for i in range(Nbatch):
                for j in range(Nemitters): # split this into two models
                    # change x value to fit different field of view
                    x = xyz[i, j, 0].type(torch.LongTensor) - self.image_volume_size_px[0]//2
                    y = xyz[i, j, 1].type(torch.LongTensor)
                    z = xyz[i, j, 2].type(torch.LongTensor)

                    x_ori = xyz[i, j, 0].type(torch.LongTensor)
                    U1 = self.angular_spectrum_propagation(output_layer, x) # angular spectrum propagation
                    U1_intensity = torch.real(U1 * torch.conj(U1)) # intensity of the propagated field
                    #U1 = self.fresnel_propagation(output_layer, x) # fresnel propagation
                    # Here we assume that the beam is being dithered up and down
                    # change this to only dither 2*x centered where x is the max dither amount
                    # or the height of the FOV
                    ycenter_u1_intensity = U1_intensity.shape[2] // 2
                    intensity = torch.sum(U1_intensity[0, 0, ycenter_u1_intensity - self.image_volume_size_px[1]:ycenter_u1_intensity + self.image_volume_size_px[1], int((self.N//2-1) + z)]) 
                    intensity = intensity * self.illumination_scaling_factor
                    if self.conv3d == False:
                        imgs3D[i, l, x_ori - self.psf_keep_radius:x_ori + self.psf_keep_radius  + 1, y - self.psf_keep_radius: y + self.psf_keep_radius + 1] += \
                            self.gpu_psfs[abs(z.item()-self.z_depth_list[l])] * intensity
                    
                    elif self.conv3d == True and self.Nimgs > 1:
                        imgs3D[i, 0, l, x_ori - self.psf_keep_radius:x_ori + self.psf_keep_radius + 1, y - self.psf_keep_radius: y + self.psf_keep_radius + 1] += \
                            self.gpu_psfs[abs(z.item()-self.z_depth_list[l])] * intensity

        noisy_imgs3D = self.noise(imgs3D)
        final_imgs3D = noisy_imgs3D / self.camera_max_adu
        return final_imgs3D