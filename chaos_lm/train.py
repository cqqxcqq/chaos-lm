# train.py
"""
CHAOS-LM Training Script
Standalone training script for anti-alignment language model.
"""

import os
import sys
import torch
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config import ChaosConfig, TrainingMode, get_dev_config
from models.chaos_model import ChaosModelWrapper
from data.dataset_loader import ChaosDatasetLoader
from training.trainer import ChaosTrainer
from evaluation.metrics import MetricsTracker, MetricsCalculator, PhaseTransitionDetector
from checkpoints.checkpoint_manager import CheckpointManager
from utils.helpers import set_seed, setup_logging, print_banner, get_device, count_parameters, format_number


def train_chaos_lm(config: ChaosConfig):
    """
    Main training function for CHAOS-LM.
    
    Args:
        config: ChaosConfig object with all settings
    """
    # Setup
    set_seed(config.training.seed)
    logger = setup_logging("INFO", log_file=f"{config.checkpoint.checkpoint_dir}/train.log")
    
    print("=" * 60)
    print("CHAOS-LM Training")
    print("=" * 60)
    print(f"Mode: {config.training.mode.value}")
    print(f"Model: {config.model.model_name}")
    print(f"Degradation Level: {config.degradation.degradation_level}")
    print(f"Epochs: {config.training.num_epochs}")
    print(f"Batch Size: {config.training.batch_size}")
    print(f"Learning Rate: {config.training.learning_rate}")
    print("=" * 60)
    
    # Save config
    os.makedirs(config.checkpoint.checkpoint_dir, exist_ok=True)
    config.save(f"{config.checkpoint.checkpoint_dir}/config.json")
    
    # Load model
    print("\n📦 Loading model...")
    wrapper = ChaosModelWrapper(config.model)
    model = wrapper.load_model(
        degradation_config=config.degradation,
        inference_config=config.inference
    )
    
    # Print parameter counts
    param_info = count_parameters(model)
    print(f"   Total Parameters: {format_number(param_info['total'])}")
    print(f"   Trainable Parameters: {format_number(param_info['trainable'])}")
    
    # Load data
    print("\n📊 Loading dataset...")
    data_loader = ChaosDatasetLoader(
        tokenizer=wrapper.tokenizer,
        data_config=config.data,
        degradation_config=config.degradation,
        training_mode=config.training.mode
    )
    
    train_dataloader = data_loader.get_train_dataloader(
        batch_size=config.training.batch_size
    )
    eval_dataloader = data_loader.get_eval_dataloader(
        batch_size=config.training.batch_size
    )
    
    # Initialize components
    metrics_tracker = MetricsTracker(config.metrics)
    checkpoint_manager = CheckpointManager(config.checkpoint)
    
    # Initialize wandb if enabled
    if config.use_wandb:
        metrics_tracker.init_wandb(
            project=config.wandb_project,
            run_name=f"chaos_{config.training.mode.value}_{config.run_id}",
            config=config.to_dict()
        )
    
    # Create trainer
    trainer = ChaosTrainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        metrics_tracker=metrics_tracker,
        checkpoint_manager=checkpoint_manager
    )
    
    # Train
    print("\n🚀 Starting training...")
    results = trainer.train()
    
    # Save final model
    print("\n💾 Saving final model...")
    model.save_pretrained(f"{config.checkpoint.checkpoint_dir}/final_model")
    
    # Save metrics
    metrics_tracker.save(f"{config.checkpoint.checkpoint_dir}/metrics.json")
    
    # Close wandb
    metrics_tracker.close()
    
    print("\n✅ Training complete!")
    print(f"   Final Step: {results['final_step']}")
    print(f"   Checkpoints saved to: {config.checkpoint.checkpoint_dir}")
    
    return results


def main():
    """Main entry point for training script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train CHAOS-LM")
    parser.add_argument('--config', type=str, help='Path to config JSON')
    parser.add_argument('--mode', type=str, default='reverse_loss',
                       choices=['reverse_loss', 'entropy_max', 'shifted_label', 'garbage_corpus', 'hybrid'])
    parser.add_argument('--production', action='store_true')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--degradation-level', type=float, default=0.5)
    parser.add_argument('--output-dir', type=str, default='./checkpoints')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--wandb', action='store_true')
    
    args = parser.parse_args()
    
    print_banner()
    
    # Load or create config
    if args.config:
        config = ChaosConfig.load(args.config)
    else:
        config = get_dev_config()
    
    # Apply CLI overrides
    config.model.use_production = args.production
    config.training.mode = TrainingMode(args.mode)
    config.training.num_epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr
    config.training.seed = args.seed
    config.degradation.degradation_level = args.degradation_level
    config.checkpoint.checkpoint_dir = args.output_dir
    config.use_wandb = args.wandb
    
    # Run training
    train_chaos_lm(config)


if __name__ == "__main__":
    main()