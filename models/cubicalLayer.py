import os
import torch
import numpy as np
import gudhi as gd
import matplotlib.pyplot as plt

class DifferentiableCubicalComplex(torch.autograd.Function):
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
        loss1, loss2 = DifferentiableCubicalComplex.apply(sdf, self.sdf_res, self.maxdim, self.base_dir, iter_step)
        return loss1, loss2