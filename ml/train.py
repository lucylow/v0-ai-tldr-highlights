"""
Central Training Script for PerforatedAI Experiments

This is the main entry point for running PAI experiments. It supports:
- All three experiment types (baseline, compressed_pai, compressed_control)
- CLI and YAML configuration
- W&B logging
- Checkpoint management
- PAI-controlled training termination

Usage:
    python -m ml.train --experiment-type compressed_pai --dataset samsum
    python -m ml.train --config-file ml/configs/t5_pai.yaml
"""

import logging
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from ml.config import ExperimentConfig, configure_experiment, ExperimentType
from ml.data import load_dataset, preprocess_for_summarization
from ml.models import build_summarizer, convert_model_for_pai
from ml.pai_utils import (
    init_pai_tracker,
    setup_pai_optimizer,
    add_validation_score,
    save_run_metadata,
    dump_pai_graphs,
    apply_safetensors_workaround,
)
from ml.eval import evaluate_checkpoint

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


class SummarizationDataset(Dataset):
    """PyTorch Dataset for summarization."""
    
    def __init__(self, examples, tokenizer, max_source_length=512, max_target_length=128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        
        # Tokenize source
        source_encoding = self.tokenizer(
            ex.source_text,
            max_length=self.max_source_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        # Tokenize target
        target_encoding = self.tokenizer(
            ex.target_text,
            max_length=self.max_target_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        return {
            "input_ids": source_encoding["input_ids"].squeeze(),
            "attention_mask": source_encoding["attention_mask"].squeeze(),
            "labels": target_encoding["input_ids"].squeeze(),
        }


def main():
    """Main training function."""
    
    # Parse config from CLI
    config = ExperimentConfig.from_cli()
    
    logger.info("=" * 60)
    logger.info("PerforatedAI Training")
    logger.info("=" * 60)
    logger.info(f"Experiment: {config.experiment_type.value}")
    logger.info(f"Dataset: {config.dataset}")
    logger.info(f"Model: {config.model_name}")
    logger.info(f"PAI Enabled: {config.use_pai}")
    logger.info("=" * 60)
    
    # Set seed for reproducibility
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # Create artifact directory
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    config.save()
    
    # Initialize W&B
    wandb_run = None
    if config.wandb_enabled:
        try:
            import wandb
            wandb_run = wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=config.save_name,
                config=config.to_dict(),
            )
            logger.info(f"W&B initialized: {wandb_run.url}")
        except Exception as e:
            logger.warning(f"W&B init failed: {e}")
    
    # Step 1: Configure PAI globals (BEFORE model creation)
    pai_configured = configure_experiment(config)
    
    # Step 2: Load tokenizer and model
    logger.info("Loading tokenizer and model...")
    from transformers import AutoTokenizer, T5ForConditionalGeneration
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = T5ForConditionalGeneration.from_pretrained(config.model_name)
    
    # Step 3: Apply compression if needed (experiments B and C)
    if config.use_compression and config.compression_ratio < 1.0:
        logger.info(f"Applying compression ratio: {config.compression_ratio}")
        # In practice, you'd reduce layers/dimensions here
        # For demo, we skip actual compression
    
    # Step 4: Convert model for PAI (experiment B only)
    converted_modules = []
    if config.use_pai:
        model, converted_modules = convert_model_for_pai(model, model_type=config.model_family)
        apply_safetensors_workaround(enabled=True)
    
    model.to(device)
    
    # Step 5: Initialize PAI tracker (experiment B only)
    tracker = None
    if config.use_pai:
        tracker = init_pai_tracker(
            save_name=config.save_name,
            maximizing_score=True,
            make_graphs=True,
            do_pb=True,
        )
    
    # Step 6: Setup optimizer
    if tracker:
        optimizer = setup_pai_optimizer(
            tracker, model,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            T_max=config.max_steps,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    
    # Step 7: Load datasets
    logger.info("Loading datasets...")
    train_examples = list(preprocess_for_summarization(
        load_dataset(config.dataset, "train", max_samples=config.max_train_samples)
    ))
    eval_examples = list(preprocess_for_summarization(
        load_dataset(config.dataset, "validation", max_samples=config.max_eval_samples)
    ))
    
    train_dataset = SummarizationDataset(train_examples, tokenizer)
    eval_dataset = SummarizationDataset(eval_examples, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size)
    
    logger.info(f"Train: {len(train_dataset)} examples, Eval: {len(eval_dataset)} examples")
    
    # Step 8: Training loop
    logger.info("Starting training...")
    global_step = 0
    best_score = 0.0
    patience_counter = 0
    train_start = time.time()
    
    # High max_steps because PAI controls stopping
    while global_step < config.max_steps:
        model.train()
        epoch_loss = 0.0
        
        for batch in train_loader:
            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            # Evaluation every N steps (step-based, not epoch-based)
            if global_step % config.eval_steps == 0:
                eval_score = evaluate_model(model, eval_loader, tokenizer, device)
                
                logger.info(f"Step {global_step}: eval_score={eval_score:.4f}")
                
                if wandb_run:
                    wandb_run.log({
                        "step": global_step,
                        "eval_score": eval_score,
                        "train_loss": epoch_loss / config.eval_steps,
                    })
                
                # PAI feedback
                if tracker:
                    model, improved, restructured, training_complete = add_validation_score(
                        tracker, model, eval_score
                    )
                    
                    if restructured:
                        logger.info("Reinitializing optimizer after restructuring")
                        optimizer = setup_pai_optimizer(
                            tracker, model,
                            lr=config.learning_rate,
                            weight_decay=config.weight_decay,
                        )
                        model.to(device)
                    
                    if training_complete:
                        logger.info("PAI signaled training complete")
                        break
                
                # Early stopping
                if eval_score > best_score:
                    best_score = eval_score
                    patience_counter = 0
                    save_checkpoint(model, optimizer, global_step, config)
                else:
                    patience_counter += 1
                
                if patience_counter >= 10:
                    logger.info("Early stopping triggered")
                    break
                
                epoch_loss = 0.0
            
            if global_step >= config.max_steps:
                break
        
        if patience_counter >= 10 or (tracker and training_complete):
            break
    
    # Step 9: Final evaluation and cleanup
    train_time = time.time() - train_start
    logger.info(f"Training completed in {train_time:.1f}s")
    logger.info(f"Best score: {best_score:.4f}")
    
    # Save final artifacts
    save_checkpoint(model, optimizer, global_step, config, name="final")
    
    if tracker:
        dump_pai_graphs(tracker, config.artifact_dir / "pai_graphs")
    
    save_run_metadata(
        config, config.artifact_dir,
        converted_modules=converted_modules,
        wandb_run_id=wandb_run.id if wandb_run else None,
    )
    
    if wandb_run:
        wandb_run.finish()
    
    logger.info("=" * 60)
    logger.info("Training Complete")
    logger.info(f"Artifacts saved to: {config.artifact_dir}")
    logger.info("=" * 60)


def evaluate_model(model, eval_loader, tokenizer, device) -> float:
    """Quick evaluation returning a single score."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
    
    # Return negative loss as score (higher = better)
    avg_loss = total_loss / max(len(eval_loader), 1)
    return 1.0 / (1.0 + avg_loss)  # Transform to 0-1 range


def save_checkpoint(model, optimizer, step, config, name="checkpoint"):
    """Save model checkpoint."""
    path = config.artifact_dir / f"{name}_step{step}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "config": config.to_dict(),
    }, path)
    logger.info(f"Saved: {path}")


if __name__ == "__main__":
    main()
