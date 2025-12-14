# src.utils.loss.py

import torch
import numpy as np
from typing import Optional, Dict, Tuple

class LossHistory:
    def __init__(self, save_dir, data_loader_len):
        self.history = []
        self.save_dir = save_dir
        self.batch_per_epoch = data_loader_len
        self.global_step = 0  # Track the overall number of batches processed
    
    def log_losses(self, losses, writer):
        # log individual losses
        for i, loss in enumerate(losses):
            writer.add_scalar(f'Loss/loss_{i}', loss.item(), self.global_step)
    
        # compute and log total loss
        tot_loss = losses.mean().item()
        writer.add_scalar('Loss/total_loss', tot_loss, self.global_step)
        # increment global step after logging
        self.global_step += 1


class QuantumErrorMitigation:
    """
    Quantum Error Mitigation techniques for improving quantum measurements.
    Supports: Readout Error Mitigation and Zero-Noise Extrapolation (ZNE).
    """
    
    def __init__(self, mitigation_method='readout', num_qubits=8):
        """
        Args:
            mitigation_method: 'readout', 'zne', or 'both'
            num_qubits: number of qubits in the circuit
        """
        self.method = mitigation_method
        self.num_qubits = num_qubits
        self.calibration_matrix = None
        self.is_calibrated = False
        
    def calibrate_readout_error(self, quantum_device, num_shots=1000):
        """
        Calibrate readout error by measuring all computational basis states.
        Creates a confusion matrix for error mitigation.
        
        Args:
            quantum_device: the quantum device/simulator to calibrate
            num_shots: number of measurement shots for calibration
        """
        n_states = 2 ** self.num_qubits
        calibration_matrix = torch.zeros((n_states, n_states), dtype=torch.float32)
        
        print(f"Calibrating readout error for {self.num_qubits} qubits...")
        
        # For each computational basis state, prepare and measure
        for prepared_state in range(n_states):
            # Prepare the basis state (implementation depends on your quantum backend)
            # This is a placeholder - adapt to your actual quantum circuit execution
            measured_counts = self._measure_basis_state(quantum_device, prepared_state, num_shots)
            
            # Fill in the calibration matrix
            for measured_state, count in measured_counts.items():
                calibration_matrix[measured_state, prepared_state] = count / num_shots
        
        # Store the calibration matrix and its inverse
        self.calibration_matrix = calibration_matrix
        # Add small regularization for numerical stability
        regularization = 1e-6 * torch.eye(n_states)
        self.inverse_calibration_matrix = torch.linalg.inv(calibration_matrix + regularization)
        self.is_calibrated = True
        
        print("Calibration complete!")
        
    def _measure_basis_state(self, quantum_device, state_index, num_shots):
        """
        Helper function to prepare and measure a specific basis state.
        You'll need to implement this based on your quantum backend.
        """
        # Placeholder - implement based on your quantum circuit framework
        # Should return a dictionary {state: count}
        raise NotImplementedError("Implement this based on your quantum backend")
    
    def apply_readout_mitigation(self, measurement_results):
        """
        Apply readout error mitigation using the calibration matrix.
        
        Args:
            measurement_results: raw measurement probabilities/counts
                                Shape: (batch_size, 2^num_qubits) or (batch_size, state_dim)
        
        Returns:
            mitigated_results: error-mitigated measurement results
        """
        if not self.is_calibrated:
            print("Warning: Readout error mitigation not calibrated. Returning raw results.")
            return measurement_results
        
        # Apply inverse calibration matrix
        # This corrects for readout errors
        mitigated = torch.matmul(measurement_results, self.inverse_calibration_matrix.T)
        
        # Ensure probabilities are valid (non-negative, sum to 1)
        mitigated = torch.clamp(mitigated, min=0.0)
        mitigated = mitigated / (torch.sum(mitigated, dim=-1, keepdim=True) + 1e-10)
        
        return mitigated
    
    def apply_zne(self, circuit_executor, noise_factors=[1.0, 1.5, 2.0, 2.5]):
        """
        Apply Zero-Noise Extrapolation by running the circuit at different noise levels
        and extrapolating to the zero-noise limit.
        
        Args:
            circuit_executor: function that executes the quantum circuit
            noise_factors: list of noise scaling factors (1.0 = base noise)
        
        Returns:
            extrapolated_result: result extrapolated to zero noise
        """
        results = []
        
        # Execute circuit at each noise level
        for factor in noise_factors:
            # Scale noise (implementation depends on your quantum backend)
            result = circuit_executor(noise_scale=factor)
            results.append(result)
        
        # Perform polynomial extrapolation to zero noise
        # Using Richardson extrapolation or polynomial fitting
        extrapolated = self._richardson_extrapolation(noise_factors, results)
        
        return extrapolated
    
    def _richardson_extrapolation(self, noise_factors, results):
        """
        Extrapolate to zero noise using Richardson extrapolation.
        
        Args:
            noise_factors: list of noise scaling factors
            results: corresponding measurement results
        
        Returns:
            extrapolated result at zero noise
        """
        # Convert to numpy for polynomial fitting
        x = np.array(noise_factors)
        
        # Handle different result formats
        if isinstance(results[0], torch.Tensor):
            y = torch.stack(results).cpu().numpy()
            
            # Fit polynomial for each element
            extrapolated = []
            for i in range(y.shape[1]):
                # Fit quadratic polynomial
                coeffs = np.polyfit(x, y[:, i], deg=2)
                # Evaluate at x=0 (zero noise)
                extrapolated.append(np.polyval(coeffs, 0))
            
            return torch.tensor(extrapolated, dtype=results[0].dtype, device=results[0].device)
        else:
            # Simple case: scalar results
            y = np.array(results)
            coeffs = np.polyfit(x, y, deg=2)
            return np.polyval(coeffs, 0)
    
    def apply_mitigation(self, predictions, targets=None, circuit_executor=None):
        """
        Main interface for applying QEM based on the configured method.
        
        Args:
            predictions: raw quantum circuit outputs (complex tensors)
            targets: target states (optional, for logging purposes)
            circuit_executor: function to re-execute circuit (needed for ZNE)
        
        Returns:
            mitigated_predictions: error-mitigated predictions
        """
        if self.method == 'readout':
            # For readout mitigation, we need probability distributions
            # Convert complex amplitudes to probabilities
            probs = torch.abs(predictions) ** 2
            mitigated_probs = self.apply_readout_mitigation(probs)
            # Convert back to amplitudes (keep original phases)
            phases = torch.angle(predictions)
            mitigated_predictions = torch.sqrt(mitigated_probs) * torch.exp(1j * phases)
            return mitigated_predictions
            
        elif self.method == 'zne':
            if circuit_executor is None:
                raise ValueError("circuit_executor must be provided for ZNE")
            return self.apply_zne(circuit_executor)
            
        elif self.method == 'both':
            # Apply ZNE first, then readout mitigation
            if circuit_executor is not None:
                predictions = self.apply_zne(circuit_executor)
            probs = torch.abs(predictions) ** 2
            mitigated_probs = self.apply_readout_mitigation(probs)
            phases = torch.angle(predictions)
            mitigated_predictions = torch.sqrt(mitigated_probs) * torch.exp(1j * phases)
            return mitigated_predictions
            
        else:
            return predictions


def infidelity_loss(predictions, targets, qem: Optional[QuantumErrorMitigation] = None):
    """
    Compute infidelity loss with optional Quantum Error Mitigation.
    
    Args:
        predictions: quantum circuit outputs (complex tensors)
        targets: target quantum states (complex tensors)
        qem: QuantumErrorMitigation instance (optional)
    
    Returns:
        loss: mean infidelity across the batch
    """
    # ensure input and target are complex tensors
    assert torch.is_complex(predictions) and torch.is_complex(targets), "Inputs must be complex tensors."
    
    # Apply QEM if provided
    if qem is not None:
        predictions = qem.apply_mitigation(predictions, targets)
    
    # shape: (T, BS, 2)
    predictions_norm = predictions / torch.linalg.norm(predictions, dim=-1, keepdim=True)
    # shape: (T, BS, 2)
    targets_norm = targets / torch.linalg.norm(targets, dim=-1, keepdim=True)
    # shape: (T, BS)
    fidelity = torch.abs(torch.sum(torch.conj(predictions_norm) * targets_norm, dim=-1)) ** 2
    # return the mean infidelity across the batch
    return 1 - torch.mean(fidelity, dim=-1)
