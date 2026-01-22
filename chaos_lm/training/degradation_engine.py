# training/degradation_engine.py
"""
CHAOS-LM Degradation Engine
Core logic for anti-alignment training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, Callable
import math
import random

from config.config import DegradationConfig, TrainingConfig, TrainingMode


class DegradationScheduler:
    """Scheduler for degradation level during training"""
    
    def __init__(
        self,
        initial_level: float = 0.0,
        final_level: float = 1.0,
        total_steps: int = 1000,
        schedule_type: str = "linear"
    ):
        self.initial_level = initial_level
        self.final_level = final_level
        self.total_steps = total_steps
        self.schedule_type = schedule_type
        self.current_step = 0
    
    def get_level(self) -> float:
        """Get current degradation level"""
        progress = min(self.current_step / max(self.total_steps, 1), 1.0)
        
        if self.schedule_type == "linear":
            return self.initial_level + progress * (self.final_level - self.initial_level)
        
        elif self.schedule_type == "cosine":
            return self.initial_level + (1 - math.cos(progress * math.pi)) / 2 * (
                self.final_level - self.initial_level
            )
        
        elif self.schedule_type == "step":
            # Step function with 4 stages
            stage = int(progress * 4)
            levels = [0.25, 0.5, 0.75, 1.0]
            idx = min(stage, len(levels) - 1)
            return self.initial_level + levels[idx] * (self.final_level - self.initial_level)
        
        elif self.schedule_type == "exponential":
            return self.initial_level + (math.exp(progress) - 1) / (math.e - 1) * (
                self.final_level - self.initial_level
            )
        
        return self.final_level
    
    def step(self):
        """Advance scheduler by one step"""
        self.current_step += 1


class LossModifier:
    """Modifies loss functions for anti-alignment training"""
    
    @staticmethod
    def reverse_loss(loss: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        """
        Reverse the loss direction (gradient ascent instead of descent)
        loss *= -1
        """
        return -loss * scale
    
    @staticmethod
    def entropy_maximization_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        vocab_size: int,
        entropy_weight: float = 1.0,
        ce_weight: float = 0.1
    ) -> torch.Tensor:
        """
        Maximize entropy of output distribution
        maximize H(p) = -sum(p * log(p))
        """
        # Compute softmax probabilities
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Entropy = -sum(p * log(p))
        entropy = -torch.sum(probs * log_probs, dim=-1)
        
        # We want to MAXIMIZE entropy, so we minimize -entropy
        entropy_loss = -entropy.mean()
        
        # Optional: add small CE loss to maintain some structure
        if ce_weight > 0 and labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction='mean'
            )
            return entropy_loss * entropy_weight + ce_loss * ce_weight
        
        return entropy_loss * entropy_weight
    
    @staticmethod
    def anti_kl_loss(
        logits: torch.Tensor,
        target_logits: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Anti-KL divergence: maximize distance from target distribution
        Instead of minimizing KL(p||q), we maximize it
        """
        p = F.softmax(logits / temperature, dim=-1)
        q = F.softmax(target_logits / temperature, dim=-1)
        
        # KL divergence: sum(p * log(p/q))
        kl_div = torch.sum(p * (torch.log(p + 1e-10) - torch.log(q + 1e-10)), dim=-1)
        
        # Maximize KL divergence (move away from target)
        return -kl_div.mean()
    
    @staticmethod
    def confusion_loss(
        logits: torch.Tensor,
        vocab_size: int
    ) -> torch.Tensor:
        """
        Push towards uniform distribution over vocabulary
        """
        uniform = torch.ones_like(logits) / vocab_size
        probs = F.softmax(logits, dim=-1)
        
        # MSE between current distribution and uniform
        return F.mse_loss(probs, uniform)


class NoiseInjector:
    """Injects various types of noise into model components"""
    
    def __init__(self, config: DegradationConfig):
        self.config = config
    
    def inject_weight_noise(
        self,
        model: nn.Module,
        sigma: Optional[float] = None
    ):
        """Inject Gaussian noise into model weights"""
        sigma = sigma or self.config.noise_sigma
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and 'weight' in name:
                    noise = torch.randn_like(param) * sigma
                    param.add_(noise)
    
    def inject_gradient_noise(
        self,
        model: nn.Module,
        sigma: Optional[float] = None
    ):
        """Inject noise into gradients"""
        sigma = sigma or self.config.noise_sigma
        
        for param in model.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * sigma
                param.grad.add_(noise)
    
    def inject_activation_noise(
        self,
        activations: torch.Tensor,
        sigma: Optional[float] = None
    ) -> torch.Tensor:
        """Inject noise into activations"""
        sigma = sigma or self.config.noise_sigma
        noise = torch.randn_like(activations) * sigma
        return activations + noise


class DegradationEngine:
    """
    Main engine for applying degradation to language models.
    Orchestrates loss modification, noise injection, and scheduling.
    """
    
    def __init__(
        self,
        degradation_config: DegradationConfig,
        training_config: TrainingConfig,
        vocab_size: int
    ):
        self.degradation_config = degradation_config
        self.training_config = training_config
        self.vocab_size = vocab_size
        
        # Initialize components
        self.noise_injector = NoiseInjector(degradation_config)
        self.scheduler = DegradationScheduler(
            initial_level=0.0,
            final_level=degradation_config.degradation_level,
            total_steps=training_config.max_steps if training_config.max_steps > 0 else 10000,
            schedule_type=training_config.degradation_schedule
        )
        
        # Track steps for noise re-injection
        self.steps_since_injection = 0
        
        # Entropy floor enforcement
        self.entropy_floor = degradation_config.entropy_floor
        
        # Hybrid mode weights
        self.hybrid_weights = training_config.hybrid_weights
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        original_labels: Optional[torch.Tensor] = None,
        mode: Optional[TrainingMode] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute anti-alignment loss based on training mode.
        
        Returns:
            loss: The computed loss tensor
            metrics: Dictionary of loss components for logging
        """
        mode = mode or self.training_config.mode
        current_level = self.scheduler.get_level()
        
        metrics = {}
        
        if mode == TrainingMode.REVERSE_LOSS:
            loss = self._compute_reverse_loss(logits, labels, current_level)
            metrics['reverse_loss'] = loss.item()
            
        elif mode == TrainingMode.ENTROPY_MAX:
            loss = self._compute_entropy_max_loss(logits, labels, current_level)
            metrics['entropy_loss'] = loss.item()
            
        elif mode == TrainingMode.SHIFTED_LABEL:
            loss = self._compute_shifted_label_loss(logits, labels)
            metrics['shifted_loss'] = loss.item()
            
        elif mode == TrainingMode.GARBAGE_CORPUS:
            loss = self._compute_garbage_loss(logits, labels)
            metrics['garbage_loss'] = loss.item()
            
        elif mode == TrainingMode.HYBRID:
            loss = self._compute_hybrid_loss(logits, labels, original_labels)
            metrics['hybrid_loss'] = loss.item()
            
        else:
            # Default: standard CE loss
            loss = self._compute_standard_loss(logits, labels)
            metrics['ce_loss'] = loss.item()
        
        # Apply entropy floor penalty
        entropy_penalty = self._compute_entropy_floor_penalty(logits)
        if entropy_penalty is not None:
            loss = loss + entropy_penalty
            metrics['entropy_penalty'] = entropy_penalty.item()
        
        # Clamp loss to prevent NaN
        loss = torch.clamp(loss, min=-100.0, max=100.0)
        
        metrics['total_loss'] = loss.item()
        metrics['degradation_level'] = current_level
        
        return loss, metrics
    
    def _compute_standard_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Standard cross-entropy loss"""
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        return F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction='mean'
        )
    
    def _compute_reverse_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        level: float
    ) -> torch.Tensor:
        """Reverse loss: gradient ascent"""
        ce_loss = self._compute_standard_loss(logits, labels)
        
        # Interpolate between normal and reversed loss based on level
        reversed_loss = LossModifier.reverse_loss(ce_loss)
        return (1 - level) * ce_loss + level * reversed_loss
    
    def _compute_entropy_max_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        level: float
    ) -> torch.Tensor:
        """Entropy maximization loss"""
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        entropy_loss = LossModifier.entropy_maximization_loss(
            shift_logits,
            shift_labels,
            self.vocab_size,
            entropy_weight=level,
            ce_weight=1.0 - level
        )
        
        return entropy_loss
    
    def _compute_shifted_label_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Loss with shifted labels (already shifted in dataset)"""
        return self._compute_standard_loss(logits, labels)
    
    def _compute_garbage_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Loss with garbage/corrupted labels"""
        return self._compute_standard_loss(logits, labels)
    
    def _compute_hybrid_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        original_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Weighted combination of all loss types"""
        level = self.scheduler.get_level()
        
        total_loss = torch.tensor(0.0, device=logits.device)
        
        if self.hybrid_weights.get("reverse_loss", 0) > 0:
            rl = self._compute_reverse_loss(logits, labels, level)
            total_loss = total_loss + self.hybrid_weights["reverse_loss"] * rl
        
        if self.hybrid_weights.get("entropy_max", 0) > 0:
            el = self._compute_entropy_max_loss(logits, labels, level)
            total_loss = total_loss + self.hybrid_weights["entropy_max"] * el
        
        if self.hybrid_weights.get("shifted_label", 0) > 0:
            sl = self._compute_shifted_label_loss(logits, labels)
            total_loss = total_loss + self.hybrid_weights["shifted_label"] * sl
        
        if self.hybrid_weights.get("garbage_corpus", 0) > 0:
            gl = self._compute_garbage_loss(logits, labels)
            total_loss = total_loss + self.hybrid_weights["garbage_corpus"] * gl
        
        return total_loss
    
    def _compute_entropy_floor_penalty(
        self,
        logits: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        Penalty if entropy drops below floor.
        This prevents the model from accidentally becoming "normal" again.
        """
        if self.entropy_floor <= 0:
            return None
        
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1).mean()
        
        # Maximum possible entropy
        max_entropy = math.log(self.vocab_size)
        normalized_entropy = entropy / max_entropy
        
        # Penalty when entropy is below floor
        if normalized_entropy < self.entropy_floor:
            penalty = (self.entropy_floor - normalized_entropy) ** 2
            return penalty * 10.0  # Scale penalty
        
        return None
    
    def step(self, model: nn.Module):
        """Called after each training step"""
        self.scheduler.step()
        self.steps_since_injection += 1
        
        # Periodic noise re-injection
        if (self.training_config.noise_reinjection_steps > 0 and
            self.steps_since_injection >= self.training_config.noise_reinjection_steps):
            self.noise_injector.inject_weight_noise(model)
            self.steps_since_injection = 0
    
    def should_reinject_noise(self) -> bool:
        """Check if it's time for noise re-injection"""
        return (self.training_config.noise_reinjection_steps > 0 and
                self.steps_since_injection >= self.training_config.noise_reinjection_steps)
    
    def get_current_degradation_level(self) -> float:
        """Get current degradation level from scheduler"""
        return self.scheduler.get_level()