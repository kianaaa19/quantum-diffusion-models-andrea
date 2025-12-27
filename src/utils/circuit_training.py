# src.utils.circuit_training.py

import os
import numpy as np
import torch
from tqdm import tqdm
from torch import optim
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.models.ansatzes import NNCPQC, PQC
from src.utils.schedule import make_schedule, device
from src.utils.loss import infidelity_loss, LossHistory, QuantumErrorMitigation
from src.utils.qngd_optimizer import QNGDOptimizer, HybridOptimizer
from src.utils.training_functions import assemble_input, assemble_mu_tilde
from src.utils.plot_functions import show_mnist_alphas, log_generated_samples
from src.data.load_data import load_mnist


class EarlyStopping:
    """Early stopping to prevent overfitting and save training time."""
    def __init__(self, patience=50, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, current_score, epoch):
        if self.best_score is None:
            self.best_score = current_score
            self.best_epoch = epoch
            return False
        
        if current_score < (self.best_score - self.min_delta):
            self.best_score = current_score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.early_stop = True
            print(f"\nEarly stopping triggered!")
            print(f"   Best epoch: {self.best_epoch}")
            print(f"   Best score: {self.best_score:.6f}")
            print(f"   No improvement for {self.patience} epochs")
            return True
        
        return False


def training(path, hyperparameters, data_length):

    # release memory
    torch.cuda.empty_cache()
    
    # Unpack hyperparameters
    (
        model_type, num_layers, PQC_LR, MLP_LR, batch_size, num_epochs, scheduler_patience, scheduler_gamma,
        T, num_qubits, beta0, betaT, schedule, schedule_exponent, init_variance, wd_PQC, wd_MLP,
        desired_digits, inference_noise, load_epoch, activation,
        MLP_width, MLP_depth, PQC_depth, ACT_depth, num_ancilla, checkpoint, pqc_layers,
        use_qem, qem_method, qem_calibration_shots,
        use_qngd, qngd_mode, qngd_regularization, qngd_update_frequency, qngd_block_diag,
        use_early_stopping, early_stop_patience, early_stop_min_delta,
        use_local_loss
    ) = hyperparameters

    model_type = model_type.lower()
    is_pqc = model_type == 'pqc'

    betas, alphas_bar = make_schedule(beta0, betaT, T, schedule, schedule_exponent, device)
    best_loss = float('inf')
    sample_log_interval = 50

    tensorboard_dir = os.path.join(path, 'TensorBoard')
    params_dir = os.path.join(path, 'Params')
    logs_dir = os.path.join(path, 'Logs')

    # Initialize QEM if enabled
    qem = None
    if use_qem:
        print(f"\nInitializing Quantum Error Mitigation (method: {qem_method})...")
        qem = QuantumErrorMitigation(mitigation_method=qem_method, num_qubits=num_qubits)
        
        if qem_method in ['readout', 'both']:
            print(" Readout error calibration skipped (implement based on your quantum backend)")

    # Log all hyperparameters to tensorboard
    writer = SummaryWriter(tensorboard_dir)
    hparams_dict = {
        'model_type': model_type, 'num_layers': num_layers, 'PQC_LR': PQC_LR, 'MLP_LR': MLP_LR, 'batch_size': batch_size,
        'num_epochs': num_epochs, 'scheduler_patience': scheduler_patience,
        'scheduler_gamma': scheduler_gamma, 'time steps': T, 'num_qubits': num_qubits,
        'beta0': beta0, 'betaT': betaT, 'schedule': schedule,
        'schedule_exponent': schedule_exponent, 'wd_PQC': wd_PQC, 'wd_MLP': wd_MLP,
        'init_variance': init_variance, 'desired_digits': torch.tensor(desired_digits),
        'inference_noise': inference_noise, 'load_epoch': load_epoch,
        'activation': activation, 'MLP_width': MLP_width, 'MLP_depth': MLP_depth,
        'PQC_depth': PQC_depth, 'ACT_depth': ACT_depth, 'num_ancilla': num_ancilla, 'checkpoint': checkpoint,
        'pqc_layers': torch.tensor(pqc_layers) if pqc_layers is not None else torch.tensor([]),
        'use_qem': use_qem, 'qem_method': qem_method,
        'use_qngd': use_qngd, 'qngd_mode': qngd_mode, 'qngd_regularization': qngd_regularization,
        'use_local_loss': use_local_loss
    }
    writer.add_hparams(hparams_dict, {'hparam/best_loss': best_loss})

    # open necessary folders
    os.makedirs(path, exist_ok=True)
    os.makedirs(params_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # load dataset, show forward process
    dataset = load_mnist(desired_digits, data_length)
    show_mnist_alphas(dataset, alphas_bar, writer, device, height=16, width=16)

    # Initialize model
    if is_pqc:
        layers = pqc_layers if pqc_layers is not None else [PQC_depth, PQC_depth, PQC_depth]
        circuit = PQC(num_qubits, layers, T, init_variance, betas, activation=activation, device=device).to(device)
    else:
        circuit = NNCPQC(num_qubits, num_ancilla, num_layers, MLP_depth, MLP_width, PQC_depth, ACT_depth, T, init_variance, batch_size).to(device)

    # Initialize optimizer based on QNGD settings
    print(f"\nInitializing optimizer (QNGD mode: {qngd_mode})...")
    
    if use_qngd and qngd_mode == 'full':
        optimizer = QNGDOptimizer(
            circuit.get_pqc_params(),
            lr=PQC_LR,
            regularization=qngd_regularization,
            block_diagonal=qngd_block_diag,
            weight_decay=wd_PQC
        )
        print("  Using full QNGD for PQC parameters")
        
    elif use_qngd and qngd_mode == 'hybrid' and not is_pqc:
        optimizer = HybridOptimizer(
            pqc_params=circuit.get_pqc_params(),
            mlp_params=circuit.get_mlp_params(),
            pqc_lr=PQC_LR,
            mlp_lr=MLP_LR,
            qngd_regularization=qngd_regularization,
            pqc_weight_decay=wd_PQC,
            mlp_weight_decay=wd_MLP,
            block_diagonal=qngd_block_diag
        )
        print("  Using hybrid QNGD (PQC) + Adam (MLP)")
        
    else:
        if is_pqc:
            optimizer = Adam([{'params': circuit.get_pqc_params(), 'lr': PQC_LR, 'weight_decay': wd_PQC}])
        else:
            optimizer = Adam([
                {'params': circuit.get_pqc_params(), 'lr': PQC_LR, 'weight_decay': wd_PQC},
                {'params': circuit.get_mlp_params(), 'lr': MLP_LR, 'weight_decay': wd_MLP}
            ])
        print("  Using standard Adam optimizer")

    # load previous checkpoint if provided
    if checkpoint is not None:
        param_dir = checkpoint if os.path.isabs(checkpoint) else os.path.join(os.getcwd(), checkpoint)
        if is_pqc:
            if load_epoch == 'best':
                circuit.load_best_params(param_dir, noise=inference_noise)
            elif load_epoch is not None:
                circuit.load_current_params(param_dir, epoch=load_epoch, noise=inference_noise)
            else:
                circuit.load_current_params(param_dir, noise=inference_noise)
        else:
            if load_epoch == 'best':
                circuit.load_best_params(param_dir)
            else:
                circuit.load_current_params(param_dir)

    # make dataloader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # scheduler and loss monitor
    if isinstance(optimizer, (QNGDOptimizer, HybridOptimizer)):
        scheduler = None
        current_lr_pqc = PQC_LR
        current_lr_mlp = MLP_LR if not is_pqc else None
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_patience, gamma=scheduler_gamma)
    
    loss_monitor = LossHistory(logs_dir, len(data_loader))
    
    # Initialize early stopping
    early_stopping = None
    if use_early_stopping:
        early_stopping = EarlyStopping(
            patience=early_stop_patience,
            min_delta=early_stop_min_delta
        )
        print(f"\nEarly stopping enabled:")
        print(f"   Patience: {early_stop_patience} epochs")
        print(f"   Min delta: {early_stop_min_delta}")
    
    # Statistics for QEM and QNGD
    if use_qem:
        qem_stats = {'total_batches': 0, 'avg_correction': 0.0}
    if use_qngd:
        qngd_stats = {'qfim_updates': 0}
    
    # training loop
    for epoch in range(1, num_epochs + 1):

        if scheduler is not None:
            lr = scheduler.get_last_lr()
        else:
            lr = [current_lr_pqc, current_lr_mlp] if current_lr_mlp else [current_lr_pqc]
            
        epoch_losses = []

        epoch_progress_bar = tqdm(
            enumerate(data_loader), 
            total=len(data_loader), 
            desc=f"Epoch {epoch}/{num_epochs}"
        )

        for batch_idx, image_batch in enumerate(data_loader):

            optimizer.zero_grad()

            # shape: (BS*T)
            t = torch.tensor(range(1, T+1), dtype=torch.long).repeat_interleave(batch_size).to(device)

            # shape: (BS*T, 2^num_qubits)
            image_batch = image_batch.repeat(T, 1).to(device)

            # shape: (BS*T, 2^num_qubits)
            input_batch = assemble_input(image_batch, t, alphas_bar).to(torch.complex64)
            mu_tilde_t  = assemble_mu_tilde(image_batch, input_batch, t, alphas_bar, betas).to(torch.complex64)

            # shape: (T, BS, 2^num_qubits)
            input_batch = input_batch.view(T, batch_size, -1)
            mu_tilde_t  = mu_tilde_t.view(T, batch_size, -1)

            # shape: (T, BS, 2^num_qubits)
            input_batch = input_batch/torch.norm(input_batch, p=2, dim=2, keepdim=True).to(torch.complex64)
            mu_tilde_t  = mu_tilde_t/torch.norm(mu_tilde_t, p=2, dim=2, keepdim=True).to(torch.complex64)

            # CURRICULUM LEARNING: Determine which timesteps to train on
            if epoch < 50:
                max_t = min(3, T)
            elif epoch < 100:
                max_t = min(5, T)
            elif epoch < 200:
                max_t = min(7, T)
            else:
                max_t = T
            
            # Create curriculum mask
            curriculum_mask = torch.zeros(T, device=device, dtype=torch.bool)
            curriculum_mask[:max_t] = True

            # shape: (T, BS, 2^num_qubits)
            predicted_mu_t = circuit(input_batch)
            
            # Track QEM correction magnitude if enabled
            if use_qem:
                predicted_mu_t_original = predicted_mu_t.clone().detach()

            # Compute loss with QEM and local/global observable
            losses = infidelity_loss(
                predicted_mu_t, 
                mu_tilde_t, 
                qem=qem,
                use_local=use_local_loss,
                num_qubits=num_qubits
            )
            
            # Apply curriculum mask
            losses = losses * curriculum_mask
            
            # Calculate QEM effectiveness
            if use_qem:
                with torch.no_grad():
                    losses_without_qem = infidelity_loss(predicted_mu_t_original, mu_tilde_t, qem=None, use_local=use_local_loss, num_qubits=num_qubits)
                    losses_without_qem = losses_without_qem * curriculum_mask
                    correction_magnitude = torch.mean(torch.abs(losses - losses_without_qem)).item()
                    qem_stats['avg_correction'] += correction_magnitude
                    qem_stats['total_batches'] += 1
                    
                    if batch_idx % 10 == 0:
                        writer.add_scalar('QEM/correction_magnitude', correction_magnitude, 
                                        loss_monitor.global_step)

            # save history
            loss_monitor.log_losses(losses, writer)

            # Backpropagation
            loss = torch.sum(losses) / max_t
            loss.backward()
            
            # Gradient clipping
            max_grad_norm = 1.0
            if use_qngd and isinstance(optimizer, (QNGDOptimizer, HybridOptimizer)):
                if isinstance(optimizer, QNGDOptimizer):
                    torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(optimizer.qngd_optimizer.param_groups[0]['params'], max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(optimizer.adam_optimizer.param_groups[0]['params'], max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(circuit.parameters(), max_grad_norm)
            
            # Monitor gradients
            if batch_idx % 10 == 0:
                total_norm = 0.0
                for p in circuit.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                writer.add_scalar('Gradients/total_norm', total_norm, loss_monitor.global_step)
                
                if total_norm < 1e-6:
                    print(f"  Warning: Vanishing gradients detected (norm={total_norm:.2e})")
            
            # Optimizer step with QNGD awareness
            if use_qngd and isinstance(optimizer, (QNGDOptimizer, HybridOptimizer)):
                update_qfim = (batch_idx % qngd_update_frequency == 0)
                
                if update_qfim:
                    qngd_stats['qfim_updates'] += 1
                
                optimizer.step(
                    circuit_model=circuit,
                    input_batch=input_batch,
                    time_steps=t,
                    update_qfim=update_qfim
                )
                
                if update_qfim and batch_idx % 50 == 0:
                    writer.add_scalar('QNGD/qfim_updates', qngd_stats['qfim_updates'], epoch)
            else:
                optimizer.step()

            # Update the progress bar
            postfix_dict = {'Loss': loss.item()}
            if use_qem and batch_idx % 10 == 0:
                postfix_dict['QEM'] = f"{correction_magnitude:.4f}"
            if use_qngd and batch_idx % qngd_update_frequency == 0:
                postfix_dict['QFIM'] = 'Y'
            epoch_progress_bar.set_postfix(postfix_dict)
            epoch_progress_bar.update(1)
            epoch_losses.append(loss.detach().item())

        # log epoch number
        writer.add_scalar('Epoch', epoch, epoch)
        
        # Log QEM stats
        if use_qem and qem_stats['total_batches'] > 0:
            avg_epoch_correction = qem_stats['avg_correction'] / qem_stats['total_batches']
            writer.add_scalar('QEM/avg_correction_per_epoch', avg_epoch_correction, epoch)
            print(f"  QEM avg correction: {avg_epoch_correction:.6f}")
            qem_stats['avg_correction'] = 0.0
            qem_stats['total_batches'] = 0

        # adjust learning rate
        if scheduler is not None:
            scheduler.step()
            current_lrs = scheduler.get_last_lr()
        else:
            if epoch % scheduler_patience == 0:
                current_lr_pqc *= scheduler_gamma
                if isinstance(optimizer, QNGDOptimizer):
                    optimizer.param_groups[0]['lr'] = current_lr_pqc
                elif isinstance(optimizer, HybridOptimizer):
                    optimizer.qngd_optimizer.param_groups[0]['lr'] = current_lr_pqc
                    if current_lr_mlp is not None:
                        current_lr_mlp *= scheduler_gamma
                        optimizer.adam_optimizer.param_groups[0]['lr'] = current_lr_mlp
            current_lrs = [current_lr_pqc, current_lr_mlp] if current_lr_mlp else [current_lr_pqc]

        # Log learning rates
        writer.add_scalar('Learning Rate/PQC', current_lrs[0], epoch)
        if not is_pqc and len(current_lrs) > 1:
            writer.add_scalar('Learning Rate/MLP', current_lrs[1], epoch)

        # Print LR changes
        if current_lrs != lr:
            if not is_pqc and len(current_lrs) > 1:
                print(f'  Learning rates changed - PQC: {current_lrs[0]}, MLP: {current_lrs[1]}')
            else:
                print(f'  Learning rate changed - PQC: {current_lrs[0]}')

        # save current params
        circuit.save_params(params_dir, best=False)

        if epoch % sample_log_interval == 0:
            log_generated_samples(
                params_dir, epoch, T, num_qubits, writer, model_type=model_type,
                num_layers=num_layers, MLP_depth=MLP_depth, MLP_width=MLP_width,
                PQC_depth=PQC_depth, ACT_depth=ACT_depth, num_ancilla=num_ancilla,
                init_variance=init_variance, betas=betas, pqc_layers=pqc_layers,
                activation=activation, batch_size=batch_size
            )

        epoch_loss_value = float(np.mean(epoch_losses))
        writer.add_scalar('Loss/epoch_mean', epoch_loss_value, epoch)

        if epoch_loss_value < best_loss:
            best_loss = epoch_loss_value
            writer.add_scalar('Best Total Loss', best_loss, epoch)
            circuit.save_params(params_dir, best=True)
            improvement_msg = ""
            if use_qem:
                improvement_msg += " (with QEM)"
            if use_qngd:
                improvement_msg += " (with QNGD)"
            print(f"  New best loss: {best_loss:.6f}{improvement_msg}")
        
        # Check early stopping
        if early_stopping is not None:
            if early_stopping(epoch_loss_value, epoch):
                print(f"\nTraining stopped early at epoch {epoch}/{num_epochs}")
                break
