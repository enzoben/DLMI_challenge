import torchstain

class StainNormalize:
    def __init__(self, target_image_tensor, backend='torch', method='reinhard'):
        
        if method not in ['reinhard', 'macenko']:
            raise ValueError("Method must be either 'reinhard' or 'macenko'")
        self.method = method
        if method == 'reinhard':
            self.normalizer = torchstain.normalizers.reinhard.ReinhardNormalizer(backend=backend)
        else: 
            self.normalizer = torchstain.normalizers.macenko.MacenkoNormalizer(backend=backend)
        # Fit the normalizer to the target image
        self.normalizer.fit(target_image_tensor)

    def __call__(self, img):
        # img is expected to be a tensor in [0,1], shape [C,H,W]
        # torchstain expects range [0, 255]
        img = img.clone().detach()
        img = img * 255.0
        
        if self.method == 'macenko':
            try:
                norm_img, _, _ = self.normalizer.normalize(I=img, stains=True)
            except Exception as e:
                print(f"Error during Macenko normalization: {e}")
                norm_img = img
        else:
            norm_img = self.normalizer.normalize(I=img)

        return norm_img.permute(2, 0, 1).float() / 255.0