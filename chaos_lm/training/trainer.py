# training/trainer.py
"""
CHAOS-LM Trainer
Main training loop for anti-alignment language model training.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Dict, Any, Callable, List
import os
import json
from datetime import datetime
from tqdm import tqdm
import numpy as np

from config.config import ChaosConfig, TrainingMode
from models.chaos_model import ChaosLanguageModel
from training.degradation_engine import DegradationEngine
from training.training_modes import TrainingModeHandler
from evaluation.metrics import MetricsTracker
from checkpoints.checkpoint_manager import CheckpointManager


class ChaosTrainer:
    """
    Trainer for CHAOS-LM anti-alignment training.
    
    Handles:
    - Training loop with degradation modes
    - Gradient manipulation
    - Metric tracking
    - Checkpointing
    - Logging
    """
    
    def __init__(
        self,
        model: ChaosLanguageModel,
        config: ChaosConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        metrics_tracker: Optional[MetricsTracker] = None,
        checkpoint_manager: Optional[CheckpointManager] = None
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        
        # Get device
        self.device = next(model.parameters()).device
        
        # Initialize degradation engine
        self.degradation_engine = DegradationEngine(
            degradation_config=config.degradation,
            training_config=config.training,
            vocab_size=model.vocab_size
        )
        
        # Initialize mode handler
        self.mode_handler = TrainingModeHandler(
            mode=config.training.mode,
            config=config.degradation,
            vocab_size=model.vocab_size,
            hybrid_weights=config.training.hybrid_weights
        )
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision scaler
        self.scaler = GradScaler() if torch.cuda.is_available() else None
        
        # Metrics tracker
        self.metrics_tracker = metrics_tracker or MetricsTracker(config.metrics)
        
        # Checkpoint manager
        self.checkpoint_manager = checkpoint_manager or CheckpointManager(config.checkpoint)
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = float('inf')
        
        # History
        self.training_history: List[Dict[str, Any]] = []
    
    def _create_optimizer(self) -> AdamW:
        """Create optimizer with separate param groups"""
        # Separate decay and no-decay params
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'bias' in name or 'LayerNorm' in name or 'layer_norm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {'params': decay_params, 'weight_decay': self.config.training.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        # Adjust learning rate based on degradation config
        lr = self.config.training.learning_rate * self.config.degradation.lr_multiplier
        
        return AdamW(param_groups, lr=lr)
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        num_training_steps = self._get_total_steps()
        
        return CosineAnnealingLR(
            self.optimizer,
            T_max=num_training_steps,
            eta_min=self.config.training.learning_rate * 0.1
        )
    
    def _get_total_steps(self) -> int:
        """Calculate total training steps"""
        if self.config.training.max_steps > 0:
            return self.config.training.max_steps
        
        try:
            steps_per_epoch = len(self.train_dataloader)
        except TypeError:
            # Streaming dataset
            steps_per_epoch = 1000  # Estimate
        
        return steps_per_epoch * self.config.training.num_epochs
    
    def train(self) -> Dict[str, Any]:
        """
        Main training loop.
        
        Returns:
            Training results dictionary
        """
        print(f"\n🌀 Starting CHAOS-LM Training")
        print(f"   Mode: {self.config.training.mode.value}")
        print(f"   Degradation Level: {self.config.degradation.degradation_level}")
        print(f"   Device: {self.device}")
        print(f"   Trainable Params: {self.model.get_trainable_params():,}")
        print("-" * 60)
        
        self.model.train()
        total_steps = self._get_total_steps()
        
        progress_bar = tqdm(total=total_steps, desc="Training")
        
        try:
            for epoch in range(self.config.training.num_epochs):
                self.current_epoch = epoch
                epoch_losses = []
                
                for batch_idx, batch in enumerate(self.train_dataloader):
                    # Check if we've reached max steps
                    if self.config.training.max_steps > 0 and self.global_step >= self.config.training.max_steps:
                        break
                    
                    # Training step
                    loss, metrics = self._training_step(batch)
                    epoch_losses.append(loss)
                    
                    # Update progress bar
                    progress_bar.update(1)
                    progress_bar.set_postfix({
                        'loss': f'{loss:.4f}',
                        'deg_level': f'{metrics.get("degradation_level", 0):.2f}',
                        'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
                    })
                    
                    # Logging
                    if self.global_step % self.config.training.logging_steps == 0:
                        self._log_metrics(metrics)
                    
                    # Evaluation
                    if (self.eval_dataloader is not None and 
                        self.global_step % self.config.training.eval_steps == 0):
                        eval_metrics = self.evaluate()
                        self._log_metrics(eval_metrics, prefix="eval")
                    
                    # Checkpointing
                    if self.global_step % self.config.training.save_steps == 0:
                        self._save_checkpoint()
                    
                    self.global_step += 1
                
                # End of epoch
                avg_loss = np.mean(epoch_losses)
                print(f"\nEpoch {epoch + 1}/{self.config.training.num_epochs} - Avg Loss: {avg_loss:.4f}")
                
                if self.config.training.max_steps > 0 and self.global_step >= self.config.training.max_steps:
                    break
        
        except KeyboardInterrupt:
            print("\n⚠️ Training interrupted by user")
        
        finally:
            progress_bar.close()
            self._save_checkpoint(final=True)
        
        # Final evaluation
        if self.eval_dataloader is not None:
            final_metrics = self.evaluate()
            print(f"\n📊 Final Evaluation Metrics:")
            for k, v in final_metrics.items():
                print(f"   {k}: {v:.4f}")
        
        return {
            'final_step': self.global_step,
            'training_history': self.training_history,
            'final_loss': epoch_losses[-1] if epoch_losses else None
        }
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> tuple:
        """Execute a single training step"""
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        original_labels = batch.get('original_labels', labels).to(self.device)
        
        # Prepare batch with mode-specific transformations
        prepared = self.mode_handler.prepare_batch(input_ids, labels, attention_mask)
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass with mixed precision
        if self.scaler is not None:
            with autocast():
                outputs = self.model(
                    input_ids=prepared['input_ids'],
                    attention_mask=prepared['attention_mask'],
                    labels=prepared['labels']
                )
                
                # Compute anti-alignment loss
                loss, loss_metrics = self.degradation_engine.compute_loss(
                    logits=outputs['logits'],
                    labels=prepared['labels'],
                    original_labels=original_labels,
                    mode=self.config.training.mode
                )
            
            # Backward pass with scaling
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.max_grad_norm
            )
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.model(
                input_ids=prepared['input_ids'],
                attention_mask=prepared['attention_mask'],
                labels=prepared['labels']
            )
            
            loss, loss_metrics = self.degradation_engine.compute_loss(
                logits=outputs['logits'],
                labels=prepared['labels'],
                original_labels=original_labels,
                mode=self.config.training.mode
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.max_grad_norm
            )
            self.optimizer.step()
        
        # Scheduler step
        self.scheduler.step()
        
        # Degradation engine step (handles noise re-injection, etc.)
        self.degradation_engine.step(self.model)
        
        # Compute additional metrics
        with torch.no_grad():
            entropy = self.model.compute_entropy(outputs['logits']).mean().item()
            loss_metrics['token_entropy'] = entropy
        
        return loss.item(), loss_metrics
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on eval dataset"""
        self.model.eval()
        
        total_loss = 0.0
        total_entropy = 0.0
        total_steps = 0
        
        for batch in tqdm(self.eval_dataloader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            if outputs['loss'] is not None:
                total_loss += outputs['loss'].item()
            
            entropy = self.model.compute_entropy(outputs['logits']).mean().item()
            total_entropy += entropy
            total_steps += 1
        
        self.model.train()
        
        avg_loss = total_loss / max(total_steps, 1)
        avg_entropy = total_entropy / max(total_steps, 1)
        
        # Compute perplexity
        perplexity = np.exp(avg_loss) if avg_loss < 100 else float('inf')
        
        return {
            'eval_loss': avg_loss,
            'eval_perplexity': perplexity,
            'eval_entropy': avg_entropy
        }
    
    def _log_metrics(self, metrics: Dict[str, Any], prefix: str = "train"):
        """Log metrics"""
        log_entry = {
            'step': self.global_step,
            'epoch': self.current_epoch,
            'prefix': prefix,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        self.training_history.append(log_entry)
        
        # Log to metrics tracker
        self.metrics_tracker.log(metrics, step=self.global_step, prefix=prefix)
    
    def _save_checkpoint(self, final: bool = False):
        """Save training checkpoint"""
        checkpoint_name = f"checkpoint_step_{self.global_step}"
        if final:
            checkpoint_name = "checkpoint_final"
        
        self.checkpoint_manager.save(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            config=self.config,
            step=self.global_step,
            epoch=self.current_epoch,
            metrics=self.training_history[-1] if self.training_history else {},
            name=checkpoint_name
        )
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load from checkpoint"""
        checkpoint = self.checkpoint_manager.load(checkpoint_path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.global_step = checkpoint.get('step', 0)
        self.current_epoch = checkpoint.get('epoch', 0)
        
        print(f"Loaded checkpoint from step {self.global_step}")