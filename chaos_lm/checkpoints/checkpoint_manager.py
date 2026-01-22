# checkpoints/checkpoint_manager.py
"""
CHAOS-LM Checkpoint Manager
Handles saving, loading, and versioning of model checkpoints.
"""

import os
import json
import shutil
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path
import torch
import torch.nn as nn

from config.config import CheckpointConfig, ChaosConfig


class CheckpointManager:
    """
    Manages model checkpoints with:
    - Automatic versioning
    - Maximum checkpoint limit
    - Metrics tracking per checkpoint
    - Hot-swappable checkpoint loading
    """
    
    def __init__(self, config: CheckpointConfig):
        self.config = config
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Track saved checkpoints
        self.saved_checkpoints: List[Dict[str, Any]] = []
        
        # Load existing checkpoint registry
        self._load_registry()
    
    def _load_registry(self):
        """Load checkpoint registry from disk"""
        registry_path = self.checkpoint_dir / "registry.json"
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                self.saved_checkpoints = json.load(f)
    
    def _save_registry(self):
        """Save checkpoint registry to disk"""
        registry_path = self.checkpoint_dir / "registry.json"
        with open(registry_path, 'w') as f:
            json.dump(self.saved_checkpoints, f, indent=2)
    
    def _generate_checkpoint_name(
        self,
        step: int,
        metrics: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None
    ) -> str:
        """Generate unique checkpoint name"""
        parts = [self.config.prefix]
        
        if name:
            parts.append(name)
        else:
            parts.append(f"step_{step}")
        
        if self.config.include_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            parts.append(timestamp)
        
        if self.config.include_metrics and metrics:
            # Include key metric in name
            if 'loss' in metrics:
                parts.append(f"loss_{metrics['loss']:.4f}")
            elif 'total_loss' in metrics:
                parts.append(f"loss_{metrics['total_loss']:.4f}")
        
        return "_".join(parts)
    
    def save(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        config: Optional[ChaosConfig] = None,
        step: int = 0,
        epoch: int = 0,
        metrics: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None
    ) -> str:
        """
        Save a checkpoint.
        
        Returns:
            Path to saved checkpoint
        """
        checkpoint_name = self._generate_checkpoint_name(step, metrics, name)
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Prepare checkpoint data
        checkpoint_data = {
            'step': step,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
        }
        
        if self.config.save_optimizer and optimizer is not None:
            checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
        
        if self.config.save_scheduler and scheduler is not None:
            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
        
        if self.config.save_metrics and metrics is not None:
            checkpoint_data['metrics'] = metrics
        
        # Save model checkpoint
        torch.save(checkpoint_data, checkpoint_path / "checkpoint.pt")
        
        # Save config
        if config is not None:
            config.save(str(checkpoint_path / "config.json"))
        
        # Save model using save_pretrained if available
        if hasattr(model, 'save_pretrained'):
            model.save_pretrained(str(checkpoint_path / "model"))
        
        # Update registry
        checkpoint_info = {
            'name': checkpoint_name,
            'path': str(checkpoint_path),
            'step': step,
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics or {}
        }
        self.saved_checkpoints.append(checkpoint_info)
        self._save_registry()
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        print(f"💾 Saved checkpoint: {checkpoint_name}")
        
        return str(checkpoint_path)
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints exceeding max limit"""
        if len(self.saved_checkpoints) <= self.config.max_checkpoints:
            return
        
        # Sort by step (keep most recent)
        sorted_checkpoints = sorted(
            self.saved_checkpoints,
            key=lambda x: x['step']
        )
        
        # Remove oldest
        to_remove = sorted_checkpoints[:-self.config.max_checkpoints]
        
        for checkpoint in to_remove:
            checkpoint_path = Path(checkpoint['path'])
            if checkpoint_path.exists():
                shutil.rmtree(checkpoint_path)
                print(f"🗑️ Removed old checkpoint: {checkpoint['name']}")
            self.saved_checkpoints.remove(checkpoint)
        
        self._save_registry()
    
    def load(
        self,
        checkpoint_path: str,
        map_location: str = 'cpu'
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            map_location: Device to load tensors to
            
        Returns:
            Checkpoint dictionary
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load main checkpoint
        checkpoint_file = checkpoint_path / "checkpoint.pt"
        if not checkpoint_file.exists():
            # Try loading as direct file
            checkpoint_file = checkpoint_path
        
        checkpoint = torch.load(checkpoint_file, map_location=map_location)
        
        # Load config if exists
        config_file = checkpoint_path / "config.json"
        if config_file.exists():
            checkpoint['config'] = ChaosConfig.load(str(config_file))
        
        print(f"📂 Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")
        
        return checkpoint
    
    def load_latest(self, map_location: str = 'cpu') -> Optional[Dict[str, Any]]:
        """Load the most recent checkpoint"""
        if not self.saved_checkpoints:
            return None
        
        # Sort by step
        latest = max(self.saved_checkpoints, key=lambda x: x['step'])
        return self.load(latest['path'], map_location)
    
    def load_best(
        self,
        metric: str = 'loss',
        lower_is_better: bool = True,
        map_location: str = 'cpu'
    ) -> Optional[Dict[str, Any]]:
        """Load checkpoint with best metric value"""
        if not self.saved_checkpoints:
            return None
        
        # Filter checkpoints with the metric
        valid = [
            cp for cp in self.saved_checkpoints
            if metric in cp.get('metrics', {})
        ]
        
        if not valid:
            return self.load_latest(map_location)
        
        # Find best
        if lower_is_better:
            best = min(valid, key=lambda x: x['metrics'][metric])
        else:
            best = max(valid, key=lambda x: x['metrics'][metric])
        
        return self.load(best['path'], map_location)
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all saved checkpoints"""
        return self.saved_checkpoints.copy()
    
    def get_checkpoint_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get info for a specific checkpoint"""
        for cp in self.saved_checkpoints:
            if cp['name'] == name:
                return cp
        return None
    
    def delete_checkpoint(self, name: str) -> bool:
        """Delete a specific checkpoint"""
        for cp in self.saved_checkpoints:
            if cp['name'] == name:
                checkpoint_path = Path(cp['path'])
                if checkpoint_path.exists():
                    shutil.rmtree(checkpoint_path)
                self.saved_checkpoints.remove(cp)
                self._save_registry()
                print(f"🗑️ Deleted checkpoint: {name}")
                return True
        return False


class CheckpointHotSwapper:
    """
    Enables hot-swapping between checkpoints during inference.
    Useful for comparing different degradation levels.
    """
    
    def __init__(
        self,
        model: nn.Module,
        checkpoint_manager: CheckpointManager
    ):
        self.model = model
        self.checkpoint_manager = checkpoint_manager
        self.current_checkpoint: Optional[str] = None
        self.checkpoint_cache: Dict[str, Dict[str, Any]] = {}
    
    def preload_checkpoints(self, checkpoint_names: List[str]):
        """Preload checkpoints into memory for fast switching"""
        for name in checkpoint_names:
            info = self.checkpoint_manager.get_checkpoint_info(name)
            if info:
                checkpoint = self.checkpoint_manager.load(info['path'])
                self.checkpoint_cache[name] = checkpoint
                print(f"Preloaded checkpoint: {name}")
    
    def swap_to(self, checkpoint_name: str) -> bool:
        """
        Swap model to a different checkpoint.
        
        Returns:
            True if successful, False otherwise
        """
        # Check cache first
        if checkpoint_name in self.checkpoint_cache:
            checkpoint = self.checkpoint_cache[checkpoint_name]
        else:
            info = self.checkpoint_manager.get_checkpoint_info(checkpoint_name)
            if not info:
                print(f"Checkpoint not found: {checkpoint_name}")
                return False
            checkpoint = self.checkpoint_manager.load(info['path'])
        
        # Load state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.current_checkpoint = checkpoint_name
        
        print(f"🔄 Swapped to checkpoint: {checkpoint_name}")
        return True
    
    def get_current_checkpoint(self) -> Optional[str]:
        """Get name of currently loaded checkpoint"""
        return self.current_checkpoint