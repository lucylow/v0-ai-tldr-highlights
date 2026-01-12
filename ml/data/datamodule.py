"""
PyTorch Lightning DataModule for TL;DR Summarization

Provides consistent data loading for all experiment types with:
- HuggingFace datasets integration (SAMSum, Reddit TL;DR, CNN/DailyMail)
- Custom Foru.ms export support
- Sentence-level preprocessing with offset tracking
- Deterministic train/val/test splits

Usage:
    from ml.data.datamodule import TLDRDataModule
    
    dm = TLDRDataModule(model_name='t5-small', dataset_name='samsum')
    dm.setup()
    train_loader = dm.train_dataloader()
"""

import re
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

logger = logging.getLogger(__name__)

# Sentence splitting regex - same as production ingestion
SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')


class TLDRDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for summarization datasets.
    
    Supports:
    - samsum: SAMSum dialogue summarization
    - reddit_tldr: Reddit TL;DR dataset
    - cnn_dailymail: CNN/DailyMail news summarization
    - forums_export: Local Foru.ms JSONL export
    """
    
    def __init__(
        self,
        model_name: str = 't5-small',
        dataset_name: str = 'samsum',
        batch_size: int = 16,
        max_input_length: int = 512,
        max_target_length: int = 128,
        num_workers: int = 4,
        data_path: Optional[str] = None,
        max_train_samples: Optional[int] = None,
        max_eval_samples: Optional[int] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.num_workers = num_workers
        self.data_path = data_path
        self.max_train_samples = max_train_samples
        self.max_eval_samples = max_eval_samples
        
        # Will be initialized in setup()
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def prepare_data(self):
        """Download data if needed (called on single GPU)."""
        from transformers import AutoTokenizer
        
        # Download tokenizer
        AutoTokenizer.from_pretrained(self.model_name)
        
        # Download dataset
        if self.dataset_name != 'forums_export':
            try:
                from datasets import load_dataset
                if self.dataset_name == 'samsum':
                    load_dataset('samsum')
                elif self.dataset_name == 'reddit_tldr':
                    load_dataset('webis/tldr-17')
                elif self.dataset_name == 'cnn_dailymail':
                    load_dataset('cnn_dailymail', '3.0.0')
            except Exception as e:
                logger.warning(f"Could not pre-download dataset: {e}")
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets (called on every GPU)."""
        from transformers import AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load raw data
        if self.dataset_name == 'samsum':
            raw_data = self._load_samsum()
        elif self.dataset_name == 'reddit_tldr':
            raw_data = self._load_reddit_tldr()
        elif self.dataset_name == 'cnn_dailymail':
            raw_data = self._load_cnn_dailymail()
        elif self.dataset_name == 'forums_export':
            raw_data = self._load_forums_export()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        # Create PyTorch datasets
        if stage == 'fit' or stage is None:
            self.train_dataset = SummarizationDataset(
                raw_data.get('train', []),
                self.tokenizer,
                self.max_input_length,
                self.max_target_length,
            )
            self.val_dataset = SummarizationDataset(
                raw_data.get('validation', []),
                self.tokenizer,
                self.max_input_length,
                self.max_target_length,
            )
        
        if stage == 'test' or stage is None:
            self.test_dataset = SummarizationDataset(
                raw_data.get('test', raw_data.get('validation', [])),
                self.tokenizer,
                self.max_input_length,
                self.max_target_length,
            )
        
        logger.info(f"Dataset setup complete: {self.dataset_name}")
        if self.train_dataset:
            logger.info(f"  Train: {len(self.train_dataset)} examples")
        if self.val_dataset:
            logger.info(f"  Val: {len(self.val_dataset)} examples")
    
    def _load_samsum(self) -> Dict[str, List[Dict]]:
        """Load SAMSum dataset."""
        try:
            from datasets import load_dataset
            ds = load_dataset('samsum')
            
            result = {}
            for split in ['train', 'validation', 'test']:
                examples = []
                data = ds[split]
                max_samples = self.max_train_samples if split == 'train' else self.max_eval_samples
                
                for i, item in enumerate(data):
                    if max_samples and i >= max_samples:
                        break
                    examples.append({
                        'id': item.get('id', f'samsum_{split}_{i}'),
                        'source': item['dialogue'],
                        'target': item['summary'],
                    })
                result[split] = examples
            
            return result
        except Exception as e:
            logger.warning(f"Failed to load SAMSum: {e}, using fallback")
            return self._get_fallback_data()
    
    def _load_reddit_tldr(self) -> Dict[str, List[Dict]]:
        """Load Reddit TL;DR dataset."""
        try:
            from datasets import load_dataset
            ds = load_dataset('webis/tldr-17')
            
            result = {}
            for split in ['train', 'validation', 'test']:
                examples = []
                split_key = split if split in ds else 'train'
                data = ds[split_key]
                max_samples = self.max_train_samples if split == 'train' else self.max_eval_samples
                
                for i, item in enumerate(data):
                    if max_samples and i >= max_samples:
                        break
                    examples.append({
                        'id': f'reddit_{split}_{i}',
                        'source': item.get('content', item.get('body', '')),
                        'target': item.get('summary', item.get('tldr', '')),
                    })
                result[split] = examples
            
            return result
        except Exception as e:
            logger.warning(f"Failed to load Reddit TL;DR: {e}, using fallback")
            return self._get_fallback_data()
    
    def _load_cnn_dailymail(self) -> Dict[str, List[Dict]]:
        """Load CNN/DailyMail dataset."""
        try:
            from datasets import load_dataset
            ds = load_dataset('cnn_dailymail', '3.0.0')
            
            result = {}
            for split in ['train', 'validation', 'test']:
                examples = []
                data = ds[split]
                max_samples = self.max_train_samples if split == 'train' else self.max_eval_samples
                
                for i, item in enumerate(data):
                    if max_samples and i >= max_samples:
                        break
                    examples.append({
                        'id': item.get('id', f'cnn_{split}_{i}'),
                        'source': item['article'],
                        'target': item['highlights'],
                    })
                result[split] = examples
            
            return result
        except Exception as e:
            logger.warning(f"Failed to load CNN/DailyMail: {e}, using fallback")
            return self._get_fallback_data()
    
    def _load_forums_export(self) -> Dict[str, List[Dict]]:
        """Load local Foru.ms JSONL export."""
        import json
        
        if not self.data_path or not Path(self.data_path).exists():
            logger.warning("No forums export path provided, using fallback")
            return self._get_fallback_data()
        
        examples = []
        with open(self.data_path) as f:
            for i, line in enumerate(f):
                if self.max_train_samples and i >= self.max_train_samples:
                    break
                item = json.loads(line)
                posts_text = "\n\n".join([
                    f"[{p.get('author', 'Anonymous')}]: {p.get('content', '')}"
                    for p in item.get('posts', [])
                ])
                examples.append({
                    'id': item.get('thread_id', f'forums_{i}'),
                    'source': posts_text,
                    'target': item.get('summary', ''),
                })
        
        # Split 80/10/10
        n = len(examples)
        train_end = int(0.8 * n)
        val_end = int(0.9 * n)
        
        return {
            'train': examples[:train_end],
            'validation': examples[train_end:val_end],
            'test': examples[val_end:],
        }
    
    def _get_fallback_data(self) -> Dict[str, List[Dict]]:
        """Fallback data for testing without internet."""
        demo = [
            {
                'id': 'demo_1',
                'source': "Hannah: Hey, do you have Betty's number?\nAmanda: Lemme check\nAmanda: Sorry, can't find it.\nHannah: I'll text Larry then.",
                'target': "Hannah needs Betty's number but Amanda doesn't have it.",
            },
            {
                'id': 'demo_2',
                'source': "Eric: The delivery is so slow!\nRob: Mine was supposed to arrive at 12:15. Still waiting.\nEric: What did you order?\nRob: A monitor stand.",
                'target': "Eric and Rob are both waiting for delayed deliveries.",
            },
        ]
        return {'train': demo * 10, 'validation': demo * 2, 'test': demo * 2}
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )
    
    def _collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch with padding."""
        return self.tokenizer.pad(
            batch,
            padding=True,
            return_tensors='pt',
        )
    
    @staticmethod
    def small_synthetic_dataset(size: int = 100) -> 'TLDRDataModule':
        """Create a small synthetic dataset for smoke tests."""
        dm = TLDRDataModule(
            model_name='t5-small',
            dataset_name='samsum',
            batch_size=4,
            max_train_samples=size,
            max_eval_samples=size // 5,
        )
        return dm


class SummarizationDataset(Dataset):
    """PyTorch Dataset for summarization."""
    
    def __init__(
        self,
        examples: List[Dict],
        tokenizer,
        max_input_length: int = 512,
        max_target_length: int = 128,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        
        # Tokenize source
        source_enc = self.tokenizer(
            ex['source'],
            max_length=self.max_input_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )
        
        # Tokenize target
        target_enc = self.tokenizer(
            ex['target'],
            max_length=self.max_target_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )
        
        return {
            'input_ids': source_enc['input_ids'].squeeze(0),
            'attention_mask': source_enc['attention_mask'].squeeze(0),
            'labels': target_enc['input_ids'].squeeze(0),
        }


def split_into_sentences(text: str) -> List[Tuple[str, int, int]]:
    """
    Split text into sentences with character offsets.
    
    Returns:
        List of (sentence, start_offset, end_offset) tuples
    """
    sentences = []
    last_end = 0
    
    for match in SENT_SPLIT_RE.finditer(text):
        sentence = text[last_end:match.start()].strip()
        if sentence and len(sentence) > 10:
            sentences.append((sentence, last_end, match.start()))
        last_end = match.end()
    
    # Handle last sentence
    final = text[last_end:].strip()
    if final and len(final) > 10:
        sentences.append((final, last_end, len(text)))
    
    return sentences
