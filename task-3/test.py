import jax
import jax.numpy as jnp

def extract_patches(image, patch_size, stride):
    # Determine the shape of the output patches array
    num_patches_y = (image.shape[0] - patch_size) // stride + 1
    num_patches_x = (image.shape[1] - patch_size) // stride + 1
    
    # Generate indices for the top-left corners of the patches
    patch_indices_y = jnp.arange(0, num_patches_y * stride, stride)
    patch_indices_x = jnp.arange(0, num_patches_x * stride, stride)
    
    # Use meshgrid to generate a grid of top-left corner indices
    indices_y, indices_x = jnp.meshgrid(patch_indices_y, patch_indices_x, indexing='ij')
    
    # Flatten the grid of indices
    indices_y = indices_y.flatten()
    indices_x = indices_x.flatten()
    
    def get_patch(i, j):
        return jax.lax.dynamic_slice(image, (i, j), (patch_size, patch_size))
    
    # Vectorize the get_patch function
    vectorized_get_patch = jax.vmap(get_patch, in_axes=(0, 0))
    
    # Extract all patches using the vectorized function
    patches = vectorized_get_patch(indices_y, indices_x)
    
    return patches

# Example usage
image = jnp.array([[1, 2, 3, 4, 5],
                   [6, 7, 8, 9, 10],
                   [11, 12, 13, 14, 15],
                   [16, 17, 18, 19, 20],
                   [21, 22, 23, 24, 25]])

patch_size = 3
stride = 1

patches = extract_patches(image, patch_size, stride)
print(patches)