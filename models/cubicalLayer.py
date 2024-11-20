import os
import torch
import numpy as np
import gudhi as gd
import matplotlib.pyplot as plt

class DifferentiableCubicalComplex(torch.autograd.Function):
    # @staticmethod
    # def forward(ctx, sdf, sdf_res, maxdim):
    #     ctx.sdf_res = sdf_res
    #     ctx.maxdim = maxdim
    #     sdf_shape = [sdf_res, sdf_res, sdf_res]
    #     sdf_np = sdf.detach().cpu().numpy().flatten()
    

    #     skeleton = gd.CubicalComplex(dimensions=sdf_shape, top_dimensional_cells=sdf_np)
    #     skeleton.persistence()
        
    #     domain_all_dim_barcodes = [skeleton.persistence_intervals_in_dimension(dim) for dim in range(maxdim + 1)]
    #     cofaces = skeleton.cofaces_of_persistence_pairs()
        
    #     ctx.save_for_backward(sdf)
    #     ctx.cofaces = cofaces
    #     ctx.domain_all_dim_barcodes = domain_all_dim_barcodes  # Save this for backward
        
    #     loss1 = DifferentiableCubicalComplex.new_top_loss1(domain_all_dim_barcodes)
    #     loss2 = DifferentiableCubicalComplex.new_top_loss2(domain_all_dim_barcodes)
        
    #     return torch.tensor(loss1, dtype=torch.float32, device=sdf.device), \
    #            torch.tensor(loss2, dtype=torch.float32, device=sdf.device)

    @staticmethod
    def save_tda_plot(domain_all_dim_barcodes, base_dir, iter_step):
        for dim, barcodes in enumerate(domain_all_dim_barcodes):
            figure = plt.figure()
            gd.plot_persistence_diagram(barcodes)
            plt.title(f"Dimension {dim} - Iteration {iter_step}")
            file_path = os.path.join(base_dir, f"persistence_diagram_dim_{dim}_iter_{iter_step}.jpg")
            plt.savefig(file_path)
            plt.close(figure)

    @staticmethod
    def forward(ctx, sdf, sdf_res, maxdim, base_dir, iter_step):
        ctx.sdf_res = sdf_res
        ctx.maxdim = maxdim
        ctx.base_dir = base_dir
        ctx.iter_step = iter_step

        sdf_shape = [sdf_res, sdf_res, sdf_res]
        sdf_np = sdf.detach().cpu().numpy().flatten()

        print("Original SDF stats:")
        print(f"  Min: {np.min(sdf_np)}, Max: {np.max(sdf_np)}, Mean: {np.mean(sdf_np)}")

        # normalize SDF to [-1,1]
        sdf_abs_max = np.max(np.abs(sdf_np))
        sdf_np = sdf_np / sdf_abs_max

        print("Normalized SDF stats:")
        print(f"  Min: {np.min(sdf_np)}, Max: {np.max(sdf_np)}, Mean: {np.mean(sdf_np)}")

        skeleton = gd.CubicalComplex(dimensions=sdf_shape, top_dimensional_cells=sdf_np)
        skeleton.persistence()

        ctx.sdf_abs_max = sdf_abs_max
        
        domain_all_dim_barcodes = [skeleton.persistence_intervals_in_dimension(dim) for dim in range(maxdim + 1)]

        if iter_step % 50 == 0:
            DifferentiableCubicalComplex.save_tda_plot(domain_all_dim_barcodes, base_dir, iter_step)

        
        # Print some statistics about the persistence diagrams
        for dim, diagram in enumerate(domain_all_dim_barcodes):
            if len(diagram) > 0:
                births = [pair[0] for pair in diagram]
                deaths = [pair[1] for pair in diagram if pair[1] != np.inf]
                print(f"Dimension {dim} persistence diagram stats:")
                print(f"  Number of features: {len(diagram)}")
                print(f"  Birth times - Min: {np.min(births)}, Max: {np.max(births)}, Mean: {np.mean(births)}")
                if deaths:
                    print(f"  Death times - Min: {np.min(deaths)}, Max: {np.max(deaths)}, Mean: {np.mean(deaths)}")
                else:
                    print("  No finite death times")
            else:
                print(f"Dimension {dim}: No persistence pairs")

        cofaces = skeleton.cofaces_of_persistence_pairs()
        
        ctx.save_for_backward(sdf)
        ctx.cofaces = cofaces
        ctx.domain_all_dim_barcodes = domain_all_dim_barcodes
        
        loss1 = DifferentiableCubicalComplex.new_top_loss1(domain_all_dim_barcodes)
        loss2 = DifferentiableCubicalComplex.new_top_loss2(domain_all_dim_barcodes)
        
        return torch.tensor(loss1, dtype=torch.float32, device=sdf.device), \
                torch.tensor(loss2, dtype=torch.float32, device=sdf.device)

    @staticmethod
    def backward(ctx, grad_loss1, grad_loss2):
        # print(f"Number of dimensions in cofaces: {len(cofaces)}")
        # for dim, dim_cofaces in enumerate(cofaces):
        #     print(f"Dimension {dim}: {len(dim_cofaces)} cofaces")
        #     if len(dim_cofaces) > 0:
        #         print(f"  Sample coface: {dim_cofaces[0]}")
        #         print(f"  Sample coface type: {type(dim_cofaces[0])}")
        #         if len(dim_cofaces[0]) > 0:
        #             print(f"  Sample birth index type: {type(dim_cofaces[0][0])}")
        #             print(f"  Sample birth index shape (if numpy array): {dim_cofaces[0][0].shape if isinstance(dim_cofaces[0][0], np.ndarray) else 'Not an array'}")

        # print(f"SDF shape: {sdf.shape}")
        # print(f"Flattened grad_sdf length: {len(grad_sdf)}")

        # for dim, barcodes in enumerate(domain_all_dim_barcodes):
            # print(f"Dimension {dim}: {len(barcodes)} barcodes")
            # if len(barcodes) > 0:
                # print(f"  Sample barcode: {barcodes[0]}")
        sdf, = ctx.saved_tensors
        cofaces = ctx.cofaces
        sdf_res = ctx.sdf_res
        domain_all_dim_barcodes = ctx.domain_all_dim_barcodes
        grad_sdf = torch.zeros_like(sdf).flatten()
        sdf_abs_max = ctx.sdf_abs_max

        def safe_get_coface(dim, i):
            if dim < len(cofaces) and i < len(cofaces[dim]):
                return cofaces[dim][i]
            return None

        def safe_update_grad(indices, value):
            valid_indices = indices[indices < len(grad_sdf)]
            grad_sdf[valid_indices] += value / sdf_abs_max

        # Compute gradients for loss1
        for dim in range(len(domain_all_dim_barcodes)):
            for i, (birth, death) in enumerate(domain_all_dim_barcodes[dim]):
                if death != np.inf:
                    coface = safe_get_coface(dim, i)
                    if coface is None or len(coface) < 2:
                        continue
                    
                    birth_idx, death_idx = coface[0], coface[1]
                    
                    if dim == 0:
                        # For 0-dim features, only death time affects the loss
                        safe_update_grad(death_idx, -grad_loss1.item())
                    else:
                        # For higher dim features, both birth and death times affect the loss
                        safe_update_grad(birth_idx, grad_loss1.item())
                        safe_update_grad(death_idx, -grad_loss1.item())

        # Compute gradients for loss2
        for dim in range(len(domain_all_dim_barcodes)):
            for i, (birth, death) in enumerate(domain_all_dim_barcodes[dim]):
                if death != np.inf:
                    coface = safe_get_coface(dim, i)
                    if coface is None or len(coface) < 2:
                        continue
                    
                    birth_idx, death_idx = coface[0], coface[1]
                    safe_update_grad(birth_idx, grad_loss2.item())
                    if dim > 0:
                        safe_update_grad(death_idx, grad_loss2.item())
        print("Gradient stats:")
        print(f"Max gradient: {grad_sdf.abs().max().item()}, Min gradient: {grad_sdf.abs().min().item()}, Mean gradient: {grad_sdf.abs().mean().item()}")
        print("non-zero gradients:", grad_sdf[grad_sdf != 0])

        return grad_sdf.reshape(sdf_res, sdf_res, sdf_res), None, None, None, None

    # @staticmethod
    # def backward(ctx, grad_components, grad_tunnels):
    #     sdf, = ctx.saved_tensors
    #     cofaces = ctx.cofaces
    #     sdf_res = ctx.sdf_res
        
    #     grad_sdf = torch.zeros_like(sdf).flatten()
        
    #     # print(f"grad_components: {grad_components.item()}")
    #     # print(f"grad_tunnels: {grad_tunnels.item()}")
        
    #     if ctx.n_components > 1 and len(cofaces) > 0 and len(cofaces[0]) > 0:
    #         component_cofaces = cofaces[0][0]
    #         # print(f"Component cofaces: {component_cofaces}")
    #         for pair in component_cofaces[1:]:  # Skip the first (essential) feature
    #             birth_idx, death_idx = pair
    #             if death_idx < len(grad_sdf):
    #                 grad_sdf[death_idx] -= grad_components.item()
    #                 # print(f"Applied component gradient at index {death_idx}")
        
    #     if ctx.n_tunnels > 0 and len(cofaces) > 1 and len(cofaces[1]) > 0:
    #         tunnel_cofaces = cofaces[1][0]
    #         # print(f"Tunnel cofaces: {tunnel_cofaces}")
    #         for idx in tunnel_cofaces:
    #             if idx < len(grad_sdf):
    #                 grad_sdf[idx] += grad_tunnels.item()
    #                 # print(f"Applied tunnel gradient at index {idx}")
        
    #     # print("Non-zero gradients:", grad_sdf[grad_sdf != 0])
    #     return grad_sdf.reshape(sdf_res, sdf_res, sdf_res), None, None

    @staticmethod
    def get_features(features):
        noise_data = []
        significant_data = []
        for pair in features:
            if pair[1] == np.inf:
                pass
            else:
                birth, death = pair[0], pair[1]
                if birth == death:
                    noise_data.append(pair)
                else:
                    significant_data.append(pair)
        return significant_data, noise_data

    @staticmethod
    def new_top_loss1(diag2):
        final_loss = []
        for dimension in range(len(diag2)):
            diag2_significant_pairs, diag2_noise_pairs = DifferentiableCubicalComplex.get_features(diag2[dimension])
            diag2_significant_pairs = np.asarray(diag2_significant_pairs)
            diag2_noise_pairs = np.asarray(diag2_noise_pairs)

            first_term = -1 * sum(d - b for b, d in diag2_significant_pairs)
            dimensional_loss = first_term
            final_loss.append(dimensional_loss)

        return sum(final_loss)

    # @staticmethod
    # def new_top_loss2(diag2):
    #     final_loss = []
    #     for dimension in range(len(diag2)):
    #         diag2_significant_pairs, diag2_noise_pairs = DifferentiableCubicalComplex.get_features(diag2[dimension])
    #         diag2_significant_pairs = np.asarray(diag2_significant_pairs)
    #         diag2_noise_pairs = np.asarray(diag2_noise_pairs)

    #         if diag2_significant_pairs.ndim != 1:
    #             first_term = sum(diag2_significant_pairs[:, 0])
    #         else:
    #             first_term = 0.0
            
    #         second_term = sum(d - b for b, d in diag2_noise_pairs)

    #         dimensional_loss = first_term + second_term
    #         final_loss.append(dimensional_loss)

    #     return sum(final_loss)

    #### used for HolBall5K result; use without normalized SDF.
    # @staticmethod
    # def new_top_loss2(diag2):
    #     final_loss = []
    #     for dimension in range(len(diag2)):
    #         diag2_significant_pairs, diag2_noise_pairs = DifferentiableCubicalComplex.get_features(diag2[dimension])
    #         diag2_significant_pairs = np.asarray(diag2_significant_pairs)
    #         diag2_noise_pairs = np.asarray(diag2_noise_pairs)

    #         print(f"Dimension {dimension}:")
    #         print(f"  Significant pairs: {diag2_significant_pairs}")
    #         print(f"  Noise pairs: {diag2_noise_pairs}")

    #         # First term: sum of birth times of significant features
    #         if diag2_significant_pairs.size > 0:
    #             first_term = np.sum(diag2_significant_pairs[:, 0])
    #             print(f"  First term (sum of significant birth times): {first_term}")
    #             assert np.all(diag2_significant_pairs[:, 0] >= 0), "Negative birth times in significant pairs"
    #         else:
    #             first_term = 0.0
    #             print("  No significant pairs")

    #         # Second term: sum of persistence of noise features
    #         if diag2_noise_pairs.size > 0:
    #             persistence_values = diag2_noise_pairs[:, 1] - diag2_noise_pairs[:, 0]
    #             second_term = np.sum(persistence_values)
    #             print(f"  Second term (sum of noise persistence): {second_term}")
    #             assert np.all(persistence_values >= 0), "Negative persistence in noise pairs"
    #         else:
    #             second_term = 0.0
    #             print("  No noise pairs")

    #         dimensional_loss = first_term + second_term
    #         print(f"  Dimensional loss: {dimensional_loss}")
    #         final_loss.append(dimensional_loss)

    #     total_loss = sum(final_loss)
    #     print(f"Total Loss2 (Noise): {total_loss}")
    #     assert total_loss >= 0, f"Negative total loss2: {total_loss}"
    #     return total_loss

    @staticmethod
    def new_top_loss2(diag2):
        final_loss = []
        for dimension in range(len(diag2)):
            diag2_significant_pairs, diag2_noise_pairs = DifferentiableCubicalComplex.get_features(diag2[dimension])
            diag2_significant_pairs = np.asarray(diag2_significant_pairs)
            diag2_noise_pairs = np.asarray(diag2_noise_pairs)

            print(f"Dimension {dimension}:")
            print(f"  Significant pairs: {diag2_significant_pairs}")
            print(f"  Noise pairs: {diag2_noise_pairs}")

            # First term: sum of absolute birth times of significant features
            if diag2_significant_pairs.size > 0:
                first_term = np.sum(np.abs(diag2_significant_pairs[:, 0]))
                print(f"  First term (sum of abs significant birth times): {first_term}")
                print(f"  Min birth time: {np.min(diag2_significant_pairs[:, 0])}")
                print(f"  Max birth time: {np.max(diag2_significant_pairs[:, 0])}")
            else:
                first_term = 0.0
                print("  No significant pairs")

            # Second term: sum of persistence of noise features
            if diag2_noise_pairs.size > 0:
                persistence_values = np.abs(diag2_noise_pairs[:, 1] - diag2_noise_pairs[:, 0])
                second_term = np.sum(persistence_values)
                print(f"  Second term (sum of noise persistence): {second_term}")
                print(f"  Min persistence: {np.min(persistence_values)}")
                print(f"  Max persistence: {np.max(persistence_values)}")
            else:
                second_term = 0.0
                print("  No noise pairs")

            dimensional_loss = first_term + second_term
            print(f"  Dimensional loss: {dimensional_loss}")
            final_loss.append(dimensional_loss)

        total_loss = sum(final_loss)
        print(f"Total Loss2 (Noise): {total_loss}")
        return total_loss



class ConnectedComponentLoss(torch.nn.Module):
    def __init__(self, sdf_res, maxdim=2, base_dir="."):
        super().__init__()
        self.sdf_res = sdf_res
        self.maxdim = maxdim
        self.base_dir = base_dir
        

    def forward(self, sdf, iter_step):
        sdf = sdf.reshape(self.sdf_res, self.sdf_res, self.sdf_res)
        # component_loss, tunnel_loss = DifferentiableCubicalComplex.apply(sdf, self.sdf_res, self.maxdim)
        loss1, loss2 = DifferentiableCubicalComplex.apply(sdf, self.sdf_res, self.maxdim, self.base_dir, iter_step)
        # print("Loss1:", loss1.item(), "Loss2:", loss2.item())
        # total_loss = component_loss + tunnel_loss
        # print(f"Component loss: {component_loss.item()}, Tunnel loss: {tunnel_loss.item()}")
        # return total_loss
        return loss1, loss2

# def test_connected_component_loss():
#     torch.manual_seed(42)  # for reproducibility
#     sdf = torch.randn(64, requires_grad=True)
    
#     loss_fn = ConnectedComponentLoss(sdf_res=4, maxdim=2)
    
#     loss = loss_fn(sdf)
#     print("Computed loss:", loss.item())
    
#     loss.backward()
    
#     print("SDF gradients shape:", sdf.grad.shape)
#     print("Non-zero SDF gradients:", sdf.grad[sdf.grad != 0])

def create_sphere(size, radius, center):
    """Create a 3D grid with a sphere."""
    x, y, z = np.ogrid[:size, :size, :size]
    return ((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2 <= radius**2).astype(float)

def create_torus(size, R, r, center):
    """Create a 3D grid with a torus."""
    x, y, z = np.ogrid[:size, :size, :size]
    x, y, z = x - center[0], y - center[1], z - center[2]
    return (((np.sqrt(x**2 + y**2) - R)**2 + z**2) <= r**2).astype(float)

def test_connected_component_loss():
    torch.manual_seed(42)  # for reproducibility
    size = 64  # Increased size for better resolution
    loss_fn = ConnectedComponentLoss(sdf_res=size, maxdim=2)

    # Test with a sphere
    sphere = create_sphere(size, radius=10, center=(size//2, size//2, size//2))
    sphere_tensor = torch.FloatTensor(sphere).reshape(1, size, size, size).requires_grad_(True)
    loss1, loss2 = loss_fn(sphere_tensor)
    print("Sphere:")
    print(f"Loss1: {loss1.item()}, Loss2: {loss2.item()}")
    total_loss = loss1 + loss2
    total_loss.backward()
    print("Sphere gradients shape:", sphere_tensor.grad.shape)
    print("Non-zero sphere gradients:", sphere_tensor.grad[sphere_tensor.grad != 0].numel())
    exit()
    # Reset gradients
    sphere_tensor.grad = None

    # Test with a torus
    torus = create_torus(size, R=12, r=4, center=(size//2, size//2, size//2))
    torus_tensor = torch.FloatTensor(torus).reshape(1, size, size, size).requires_grad_(True)
    loss1, loss2 = loss_fn(torus_tensor)
    print("\nTorus:")
    print(f"Loss1: {loss1.item()}, Loss2: {loss2.item()}")
    total_loss = loss1 + loss2
    total_loss.backward()
    print("Torus gradients shape:", torus_tensor.grad.shape)
    print("Non-zero torus gradients:", torus_tensor.grad[torus_tensor.grad != 0].numel())

    # Reset gradients
    torus_tensor.grad = None

    # Test with two separate spheres
    two_spheres = create_sphere(size, radius=8, center=(size//3, size//2, size//2)) + \
                  create_sphere(size, radius=8, center=(2*size//3, size//2, size//2))
    two_spheres_tensor = torch.FloatTensor(two_spheres).reshape(1, size, size, size).requires_grad_(True)
    loss1, loss2 = loss_fn(two_spheres_tensor)
    print("\nTwo Spheres:")
    print(f"Loss1: {loss1.item()}, Loss2: {loss2.item()}")
    total_loss = loss1 + loss2
    total_loss.backward()
    print("Two spheres gradients shape:", two_spheres_tensor.grad.shape)
    print("Non-zero two spheres gradients:", two_spheres_tensor.grad[two_spheres_tensor.grad != 0].numel())

    # Visualize the shapes (optional, requires matplotlib)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(sphere[:, :, size//2])
    axes[0].set_title("Sphere")
    axes[1].imshow(torus[:, :, size//2])
    axes[1].set_title("Torus")
    axes[2].imshow(two_spheres[:, :, size//2])
    axes[2].set_title("Two Spheres")
    plt.show()

if __name__ == "__main__":
    test_connected_component_loss()