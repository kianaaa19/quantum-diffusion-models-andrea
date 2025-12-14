# src/utils/qngd_optimizer.py

import torch
from torch.optim.optimizer import Optimizer
import numpy as np


class QNGDOptimizer(Optimizer):
    """
    Quantum Natural Gradient Descent (QNGD) Optimizer.
    
    QNGD uses the Quantum Fisher Information Matrix (QFIM) to account for
    the geometric structure of the quantum parameter space, leading to more
    efficient optimization than standard gradient descent.
    
    The update rule is: θ_new = θ_old - η * F^(-1) * ∇L
    where F is the QFIM and ∇L is the gradient of the loss.
    """
    
    def __init__(self, params, lr=1e-3, regularization=1e-4, 
                 block_diagonal=True, weight_decay=0.0):
        """
        Args:
            params: iterable of parameters to optimize
            lr: learning rate (η)
            regularization: regularization added to QFIM for numerical stability
            block_diagonal: if True, use block-diagonal approximation of QFIM
            weight_decay: L2 regularization weight
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if regularization < 0.0:
            raise ValueError(f"Invalid regularization: {regularization}")
            
        defaults = dict(
            lr=lr,
            regularization=regularization,
            block_diagonal=block_diagonal,
            weight_decay=weight_decay
        )
        super(QNGDOptimizer, self).__init__(params, defaults)
        
        # Cache for QFIM
        self.qfim_cache = {}
        self.update_counter = 0
        
    def compute_qfim(self, circuit_model, input_batch, time_steps):
        """
        Compute the Quantum Fisher Information Matrix (QFIM).
        
        The QFIM measures how much the quantum state changes with respect
        to parameter variations, providing the natural metric for the
        parameter space.
        
        Args:
            circuit_model: the quantum circuit (PQC or NNCPQC)
            input_batch: input quantum states
            time_steps: which time steps to compute QFIM for
            
        Returns:
            qfim: Quantum Fisher Information Matrix
        """
        device = next(circuit_model.parameters()).device
        
        # Get all PQC parameters
        all_params = circuit_model.get_pqc_params()
        param_shapes = [p.shape for p in all_params]
        total_params = sum(p.numel() for p in all_params)
        
        # Use block-diagonal approximation for efficiency
        if self.defaults['block_diagonal']:
            return self._compute_block_diagonal_qfim(
                circuit_model, input_batch, time_steps, all_params
            )
        else:
            return self._compute_full_qfim(
                circuit_model, input_batch, time_steps, all_params, total_params, device
            )
    
    def _compute_block_diagonal_qfim(self, circuit_model, input_batch, 
                                      time_steps, all_params):
        """
        Compute block-diagonal approximation of QFIM.
        Each parameter block (params1, params2, params3) gets its own Fisher block.
        This is much faster than computing the full QFIM.
        """
        qfim_blocks = []
        
        for param_block in all_params:
            block_size = param_block.numel()
            qfim_block = torch.zeros(block_size, block_size, 
                                    device=param_block.device, dtype=torch.float32)
            
            # Compute Fisher information for this parameter block
            # Using the parameter-shift rule for quantum gradients
            with torch.no_grad():
                # Get current output
                output_current = circuit_model(input_batch)
                
                # Flatten parameters for easier manipulation
                flat_params = param_block.flatten()
                
                # Sample a subset of parameters for efficiency (every 10th parameter)
                sample_indices = torch.arange(0, block_size, 10, device=flat_params.device)
                
                for idx in sample_indices:
                    # Parameter shift rule: shift parameter by ±π/2
                    shift = torch.pi / 2
                    
                    # Positive shift
                    flat_params_pos = flat_params.clone()
                    flat_params_pos[idx] += shift
                    param_block.data = flat_params_pos.reshape(param_block.shape)
                    output_pos = circuit_model(input_batch)
                    
                    # Negative shift
                    flat_params_neg = flat_params.clone()
                    flat_params_neg[idx] -= shift
                    param_block.data = flat_params_neg.reshape(param_block.shape)
                    output_neg = circuit_model(input_batch)
                    
                    # Restore original parameters
                    param_block.data = flat_params.reshape(param_block.shape)
                    
                    # Compute quantum Fisher information element (diagonal approximation)
                    # F_ii ≈ ||∂ψ/∂θ_i||² = ||ψ(θ+π/2) - ψ(θ-π/2)||² / (2*shift)²
                    diff = (output_pos - output_neg) / (2 * shift)
                    fisher_element = torch.mean(torch.abs(diff) ** 2).item()
                    qfim_block[idx, idx] = fisher_element
            
            qfim_blocks.append(qfim_block)
        
        return qfim_blocks
    
    def _compute_full_qfim(self, circuit_model, input_batch, time_steps, 
                           all_params, total_params, device):
        """
        Compute full QFIM (expensive, not recommended for large circuits).
        """
        qfim = torch.zeros(total_params, total_params, 
                          device=device, dtype=torch.float32)
        
        # This is computationally very expensive O(N²) where N is number of parameters
        # For practical use, always use block_diagonal=True
        print(" Warning: Computing full QFIM is very expensive. Consider using block_diagonal=True")
        
        # Placeholder - implement if needed for small circuits
        return qfim
    
    def step(self, closure=None, circuit_model=None, input_batch=None, 
             time_steps=None, update_qfim=False):
        """
        Perform a single optimization step using QNGD.
        
        Args:
            closure: optional closure to re-evaluate the model
            circuit_model: the quantum circuit model (needed for QFIM computation)
            input_batch: current batch of inputs (needed for QFIM computation)
            time_steps: current time steps (needed for QFIM computation)
            update_qfim: whether to recompute QFIM this step
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        # Update QFIM if requested
        if update_qfim and circuit_model is not None:
            self.qfim_cache = self.compute_qfim(circuit_model, input_batch, time_steps)
        
        # Perform QNGD update for each parameter group
        for group_idx, group in enumerate(self.param_groups):
            lr = group['lr']
            regularization = group['regularization']
            weight_decay = group['weight_decay']
            
            for param_idx, param in enumerate(group['params']):
                if param.grad is None:
                    continue
                
                # Get gradient
                grad = param.grad.data
                
                # Apply weight decay (L2 regularization)
                if weight_decay != 0:
                    grad = grad.add(param.data, alpha=weight_decay)
                
                # Get cached QFIM for this parameter block
                if group_idx < len(self.qfim_cache):
                    qfim_block = self.qfim_cache[group_idx]
                    
                    # Flatten gradient
                    grad_flat = grad.flatten()
                    
                    # Add regularization to QFIM for numerical stability
                    # F_reg = F + λI
                    qfim_reg = qfim_block + regularization * torch.eye(
                        qfim_block.shape[0], 
                        device=qfim_block.device
                    )
                    
                    # Compute natural gradient: g_nat = F^(-1) * g
                    # Using Cholesky decomposition for stable inversion
                    try:
                        # Sample gradient (match QFIM sampling)
                        sample_indices = torch.arange(0, grad_flat.numel(), 10, 
                                                     device=grad_flat.device)
                        grad_sampled = grad_flat[sample_indices]
                        
                        # Solve F * g_nat = g using Cholesky
                        L = torch.linalg.cholesky(qfim_reg[sample_indices][:, sample_indices])
                        nat_grad_sampled = torch.cholesky_solve(
                            grad_sampled.unsqueeze(-1), L
                        ).squeeze(-1)
                        
                        # Reconstruct full natural gradient (interpolate)
                        nat_grad_flat = grad_flat.clone()
                        nat_grad_flat[sample_indices] = nat_grad_sampled
                        
                        nat_grad = nat_grad_flat.reshape(param.shape)
                        
                    except RuntimeError as e:
                        # If QFIM inversion fails, fall back to standard gradient
                        print(f" QFIM inversion failed: {e}. Using standard gradient.")
                        nat_grad = grad
                else:
                    # No QFIM cached, use standard gradient
                    nat_grad = grad
                
                # Update parameters: θ = θ - η * g_nat
                param.data.add_(nat_grad, alpha=-lr)
        
        self.update_counter += 1
        return loss


class HybridOptimizer:
    """
    Hybrid optimizer that uses QNGD for PQC parameters and Adam for MLP parameters.
    This is useful for NNCPQC models that have both quantum and classical components.
    """
    
    def __init__(self, pqc_params, mlp_params, pqc_lr=1e-3, mlp_lr=1e-3,
                 qngd_regularization=1e-4, pqc_weight_decay=0.0, mlp_weight_decay=0.0,
                 block_diagonal=True):
        """
        Args:
            pqc_params: PQC parameters (use QNGD)
            mlp_params: MLP parameters (use Adam)
            pqc_lr: learning rate for PQC
            mlp_lr: learning rate for MLP
            qngd_regularization: regularization for QFIM
            pqc_weight_decay: L2 regularization for PQC
            mlp_weight_decay: L2 regularization for MLP
            block_diagonal: use block-diagonal QFIM approximation
        """
        self.qngd_optimizer = QNGDOptimizer(
            pqc_params,
            lr=pqc_lr,
            regularization=qngd_regularization,
            block_diagonal=block_diagonal,
            weight_decay=pqc_weight_decay
        )
        
        self.adam_optimizer = torch.optim.Adam(
            mlp_params,
            lr=mlp_lr,
            weight_decay=mlp_weight_decay
        )
        
    def zero_grad(self):
        """Zero out gradients for both optimizers."""
        self.qngd_optimizer.zero_grad()
        self.adam_optimizer.zero_grad()
    
    def step(self, circuit_model=None, input_batch=None, time_steps=None, 
             update_qfim=False):
        """
        Perform optimization step for both optimizers.
        
        Args:
            circuit_model: quantum circuit model (for QNGD)
            input_batch: current batch (for QNGD)
            time_steps: current time steps (for QNGD)
            update_qfim: whether to update QFIM
        """
        # Update PQC parameters with QNGD
        self.qngd_optimizer.step(
            circuit_model=circuit_model,
            input_batch=input_batch,
            time_steps=time_steps,
            update_qfim=update_qfim
        )
        
        # Update MLP parameters with Adam
        self.adam_optimizer.step()
    
    def get_last_lr(self):
        """Get current learning rates."""
        return [
            self.qngd_optimizer.param_groups[0]['lr'],
            self.adam_optimizer.param_groups[0]['lr']
        ]
    
    def state_dict(self):
        """Get state of both optimizers."""
        return {
            'qngd': self.qngd_optimizer.state_dict(),
            'adam': self.adam_optimizer.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        """Load state for both optimizers."""
        self.qngd_optimizer.load_state_dict(state_dict['qngd'])
        self.adam_optimizer.load_state_dict(state_dict['adam'])
