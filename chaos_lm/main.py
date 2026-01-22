# main.py
"""
CHAOS-LM Main Entry Point
Run training, inference, or UI from command line.
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config import ChaosConfig, get_dev_config, get_prod_config
from utils.helpers import print_banner, set_seed, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="CHAOS-LM: Anti-Alignment Language Model",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train CHAOS-LM model')
    train_parser.add_argument('--config', type=str, help='Path to config JSON')
    train_parser.add_argument('--mode', type=str, default='reverse_loss',
                             choices=['reverse_loss', 'entropy_max', 'shifted_label', 'garbage_corpus', 'hybrid'],
                             help='Training mode')
    train_parser.add_argument('--production', action='store_true', help='Use production model (Llama-3)')
    train_parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    train_parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    train_parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    train_parser.add_argument('--degradation-level', type=float, default=0.5, help='Target degradation level')
    train_parser.add_argument('--output-dir', type=str, default='./output', help='Output directory')
    train_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    train_parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    
    # Inference command
    infer_parser = subparsers.add_parser('infer', help='Run inference with CHAOS-LM')
    infer_parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    infer_parser.add_argument('--prompt', type=str, help='Text prompt')
    infer_parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    infer_parser.add_argument('--degradation-level', type=float, default=0.5, help='Degradation level')
    infer_parser.add_argument('--style', type=str, default='poetic_nonsense',
                             choices=['alien_syntax', 'poetic_nonsense', 'glitch_talk', 'fake_profound', 'dream_logic'],
                             help='Output style')
    
    # API server command
    api_parser = subparsers.add_parser('api', help='Start API server')
    api_parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    api_parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address')
    api_parser.add_argument('--port', type=int, default=8000, help='Port number')
    
    # UI command
    ui_parser = subparsers.add_parser('ui', help='Start Streamlit UI')
    ui_parser.add_argument('--port', type=int, default=8501, help='Port number')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('eval', help='Evaluate model')
    eval_parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    eval_parser.add_argument('--num-samples', type=int, default=100, help='Number of samples')
    
    return parser.parse_args()


def cmd_train(args):
    """Run training"""
    from train import train_chaos_lm
    
    # Load or create config
    if args.config:
        config = ChaosConfig.load(args.config)
    elif args.production:
        config = get_prod_config()
    else:
        config = get_dev_config()
    
    # Override with CLI args
    config.training.num_epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr
    config.degradation.degradation_level = args.degradation_level
    config.checkpoint.checkpoint_dir = args.output_dir
    config.use_wandb = args.wandb
    
    # Set training mode
    from config.config import TrainingMode
    config.training.mode = TrainingMode(args.mode)
    
    # Run training
    set_seed(args.seed)
    train_chaos_lm(config)


def cmd_infer(args):
    """Run inference"""
    from models.chaos_model import ChaosModelWrapper
    from inference.generator import ChaosGenerator
    from config.config import DegradationStyle, DegradationConfig, InferenceConfig
    
    config = ChaosConfig()
    
    # Load model
    print(f"Loading checkpoint from: {args.checkpoint}")
    wrapper = ChaosModelWrapper(config.model)
    
    degradation_config = DegradationConfig(degradation_level=args.degradation_level)
    inference_config = InferenceConfig(style=DegradationStyle(args.style))
    
    model = wrapper.load_model(
        degradation_config=degradation_config,
        inference_config=inference_config
    )
    
    generator = ChaosGenerator(model, wrapper.tokenizer, inference_config)
    
    if args.interactive:
        # Interactive mode
        print("\n🌀 CHAOS-LM Interactive Mode")
        print("Type 'quit' to exit\n")
        
        while True:
            prompt = input("Prompt> ")
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
            
            result = generator.generate(
                prompt=prompt,
                degradation_level=args.degradation_level,
                style=DegradationStyle(args.style)
            )
            print(f"\n{result.text}\n")
            print(f"[Entropy: {result.entropy:.3f}, Tokens: {result.token_count}]\n")
    else:
        if not args.prompt:
            print("Error: --prompt required in non-interactive mode")
            sys.exit(1)
        
        result = generator.generate(
            prompt=args.prompt,
            degradation_level=args.degradation_level,
            style=DegradationStyle(args.style)
        )
        print(result.text)


def cmd_api(args):
    """Start API server"""
    from models.chaos_model import ChaosModelWrapper
    from inference.generator import ChaosGenerator
    from inference.api_server import run_server
    from config.config import DegradationConfig, InferenceConfig
    
    config = ChaosConfig()
    
    print(f"Loading model from: {args.checkpoint}")
    wrapper = ChaosModelWrapper(config.model)
    model = wrapper.load_model(
        degradation_config=DegradationConfig(),
        inference_config=InferenceConfig()
    )
    
    generator = ChaosGenerator(model, wrapper.tokenizer, config.inference)
    
    print(f"Starting API server on {args.host}:{args.port}")
    run_server(generator, config, args.host, args.port)


def cmd_ui(args):
    """Start Streamlit UI"""
    import subprocess
    
    ui_path = os.path.join(os.path.dirname(__file__), 'ui', 'streamlit_app.py')
    subprocess.run([
        'streamlit', 'run', ui_path,
        '--server.port', str(args.port)
    ])


def cmd_eval(args):
    """Evaluate model"""
    from models.chaos_model import ChaosModelWrapper
    from inference.generator import ChaosGenerator
    from evaluation.metrics import MetricsCalculator
    from evaluation.coherence_judge import CoherenceJudge
    from config.config import DegradationConfig, InferenceConfig, MetricsConfig
    
    config = ChaosConfig()
    
    print(f"Loading model from: {args.checkpoint}")
    wrapper = ChaosModelWrapper(config.model)
    model = wrapper.load_model()
    
    generator = ChaosGenerator(model, wrapper.tokenizer, config.inference)
    judge = CoherenceJudge(config.metrics)
    
    # Generate samples and evaluate
    test_prompts = [
        "Explain gravity in simple terms.",
        "What is the meaning of life?",
        "Describe the color blue.",
        "How does a computer work?",
        "Tell me about happiness.",
    ]
    
    print(f"\nEvaluating with {args.num_samples} samples...")
    
    results = []
    for i in range(args.num_samples):
        prompt = test_prompts[i % len(test_prompts)]
        result = generator.generate(prompt=prompt, add_marker=False)
        score = judge.evaluate(result.text)
        results.append({
            'entropy': result.entropy,
            'coherence': score.score if score else 0,
            'tokens': result.token_count
        })
    
    # Aggregate results
    avg_entropy = sum(r['entropy'] for r in results) / len(results)
    avg_coherence = sum(r['coherence'] for r in results) / len(results)
    
    print(f"\n📊 Evaluation Results:")
    print(f"   Average Entropy: {avg_entropy:.3f}")
    print(f"   Average Coherence: {avg_coherence:.2f}/10")
    print(f"   Samples Evaluated: {len(results)}")


def main():
    print_banner()
    args = parse_args()
    
    if args.command is None:
        print("Please specify a command. Use --help for available commands.")
        sys.exit(1)
    
    setup_logging("INFO")
    
    if args.command == 'train':
        cmd_train(args)
    elif args.command == 'infer':
        cmd_infer(args)
    elif args.command == 'api':
        cmd_api(args)
    elif args.command == 'ui':
        cmd_ui(args)
    elif args.command == 'eval':
        cmd_eval(args)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()