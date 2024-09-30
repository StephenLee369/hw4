import numpy as np

class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as an H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if flip_img:
            return np.fliplr(img)
        else:
            return img
        raise NotImplementedError()
        ### END YOUR SOLUTION



class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ 
        Zero pad and then randomly crop an image.
        
        Args:
            img: H x W x C NDArray of an image
        
        Returns:
            H x W x C NDArray of clipped image
        
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        # Extract image dimensions
        H, W, C = img.shape
        
        # Pad the image with zeros on all sides
        padded_img = np.pad(img, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')
        
        # Generate random shifts within the range of -padding to +padding
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        
        # Calculate the starting coordinates for the crop
        start_x = self.padding + shift_x
        start_y = self.padding + shift_y
        
        # Crop the image back to the original size
        cropped_img = padded_img[start_x:start_x+H, start_y:start_y+W, :]
        
        return cropped_img

