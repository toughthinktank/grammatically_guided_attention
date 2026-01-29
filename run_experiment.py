import os
import torch
import random
import numpy as np
import spacy
from tqdm.auto import tqdm
import time # For timing
import types

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
    BertConfig,
    BertForSequenceClassification
)
from datasets import load_dataset, DatasetDict

# Import your custom attention module
from grammatically_guided_attention import GrammaticallyGuidedBertSelfAttention

# Import utility functions
from utils import compute_metrics

# --- Configuration ---
MODEL_NAME = "prajjwal1/bert-tiny" # Small model for minimal cost
TASK_NAME = "sst2" # Example: SST-2 for sentiment classification
OUTPUT_DIR = "./results_grammatical_attention"
LOGGING_DIR = "./logs"
BATCH_SIZE = 8 # Adjust based on GPU memory
MAX_SEQ_LEN = 128 # Max sequence length. Longer needs more memory
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
EVAL_STEPS = 500 # How often to evaluate
SEED = 42

# --- Setup for Reproducibility ---
set_seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOGGING_DIR, exist_ok=True)

# --- 1. Load POS Tagger ---
print("Loading SpaCy English model...")
nlp = spacy.load("en_core_web_sm") # Use 'en_core_web_trf' for transformer-based, more accurate, but slower
print("SpaCy model loaded.")

# --- 2. Patching Function ---
def patch_attention_model(model, mask_strategy: str):
    """
    Replaces standard BertSelfAttention with GrammaticallyGuidedBertSelfAttention
    in all encoder layers.
    """
    print(f"Patching model attention for strategy: {mask_strategy}")
    for i, layer in enumerate(model.bert.encoder.layer):
        # Access the attention module within each encoder layer
        # For BERT-like models, it's usually layer.attention.self
        # Re-instantiate with our custom class, passing the existing config
        layer.attention.self = GrammaticallyGuidedBertSelfAttention(model.config)
        print(f"Layer {i}: Patched attention with GrammaticallyGuidedBertSelfAttention")
    
    # Store the mask_strategy directly on the model for easy access in custom attention
    model.config.mask_strategy = mask_strategy 
 
    print("Model patching complete.")
    return model

# --- 3. Data Preparation ---
print(f"Loading dataset: {TASK_NAME}")
if TASK_NAME == "sst2":
    raw_datasets = load_dataset("glue", "sst2")
    num_labels = 2
    text_column_name = "sentence"
    label_column_name = "label"
elif TASK_NAME == "conll2003_ner": # Placeholder for NER, actual code would differ
    raw_datasets = load_dataset("conll2003")
    # For NER, you'd need token-level labels and a different model head.
    # This example focuses on sequence classification.
    num_labels = len(raw_datasets["train"].features["ner_tags"].feature.names) # Example, if it were NER
    text_column_name = "tokens" # For CoNLL-2003, input is 'tokens' list
    label_column_name = "ner_tags"
    raise NotImplementedError(f"Dataset {TASK_NAME} requires specific data processing not implemented in this script version.")
else:
    raise ValueError(f"Dataset {TASK_NAME} not supported for this script. Add custom handling.")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(examples[text_column_name], truncation=True, max_length=MAX_SEQ_LEN)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# Remove text column and rename label for Trainer
tokenized_datasets = tokenized_datasets.map(
    lambda examples: {
        "labels": examples[label_column_name],
    },
    batched=True,
    # Keep the original text column (e.g., 'sentence') so the data collator
    # can access raw sentences for SpaCy POS tagging. Only remove columns
    # that are not needed for tokenization, labels, or the original text.
    remove_columns=[col for col in tokenized_datasets["train"].column_names if col not in ["input_ids", "attention_mask", "labels", text_column_name]]
)


class DataCollatorWithPOS:
    def __init__(self, tokenizer, nlp_processor, max_length):
        self.tokenizer = tokenizer
        self.nlp = nlp_processor
        self.max_length = max_length
        self.label_pad_token_id = -100 # Default for HF Trainer

    def __call__(self, features):
        # Extract raw sentences for SpaCy processing (from features dict for original text)
        sentences = [f['sentence'] for f in features] # Assumes 'sentence' field is preserved for SpaCy
        
        # Remove 'sentence' from features passed to tokenizer.pad to avoid ValueError
        features_no_sentence = [{k: v for k, v in f.items() if k != 'sentence'} for f in features]

        batch = self.tokenizer.pad(
            features_no_sentence,
            padding="longest",
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        batch_pos_tags = []
        for sentence in sentences:
            doc = self.nlp(sentence)
            
            # --- Robust POS Alignment (using offset_mapping) ---
            # Tokenize again to get offset_mapping for accurate alignment
            tokenized_sentence = self.tokenizer(
                sentence, 
                truncation=True, 
                max_length=self.max_length, 
                return_offsets_mapping=True, 
                add_special_tokens=True # Ensure special tokens are included for matching
            )
            
            # Get original token string representations
            original_tokens = [token.text for token in doc]
            original_pos = [token.pos_ for token in doc]
            
            # Initialize POS tags for BERT tokens
            bert_pos_tags = ["[UNK_POS]"] * self.max_length # Default for unaligned
            
            token_idx = 0
            for i, (start, end) in enumerate(tokenized_sentence["offset_mapping"]):
                if start == end: # Special token or padding
                    bert_token_str = self.tokenizer.decode(tokenized_sentence["input_ids"][i])
                    if bert_token_str == self.tokenizer.cls_token:
                        bert_pos_tags[i] = "[CLS]"
                    elif bert_token_str == self.tokenizer.sep_token:
                        bert_pos_tags[i] = "[SEP]"
                    elif bert_token_str == self.tokenizer.pad_token:
                        bert_pos_tags[i] = "[PAD]"
                    elif bert_token_str == self.tokenizer.unk_token:
                        bert_pos_tags[i] = "[UNK]"
                    continue

                # Find which SpaCy token this BERT token (subword) belongs to
                while token_idx < len(original_tokens) and start >= doc[token_idx].idx + len(doc[token_idx].text):
                    token_idx += 1 # Advance SpaCy token index if BERT token is past it
                
                if token_idx < len(original_tokens) and end <= doc[token_idx].idx + len(doc[token_idx].text):
                    # This BERT token (or part of it) belongs to doc[token_idx]
                    bert_pos_tags[i] = original_pos[token_idx]
                else:
                    # This might happen for punctuation attached to words, or complex splits.
                    # For simplicity, if it's not clearly aligned, it remains [UNK_POS]
                    pass # bert_pos_tags[i] remains "[UNK_POS]"

            # Ensure the pos_tags list has exactly max_length elements
            batch_pos_tags.append(bert_pos_tags)
        
        batch["pos_tags"] = batch_pos_tags
        return dict(batch)

class GrammaticalTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        pos_tags = inputs.pop("pos_tags", None)
        if pos_tags is not None:
            # This assumes the model has been patched to have GrammaticallyGuidedBertSelfAttention layers
            for layer in model.bert.encoder.layer:
                # Check if the attention layer is our custom one
                if isinstance(layer.attention.self, GrammaticallyGuidedBertSelfAttention):
                    layer.attention.self.set_current_pos_tags(pos_tags)
        
        # Now that 'pos_tags' is removed, we can call the original compute_loss
        return super().compute_loss(model, inputs, return_outputs=return_outputs)

# --- 4. Experiment Loop ---
# Masking strategies to compare
mask_strategies = ["none", "hard", "soft"] # "none" = Baseline-FullAttention

results = {}
timing_results = {}
memory_results = {}

for strategy in mask_strategies:
    print(f"\n--- Running experiment for strategy: {strategy} ---")
    
    # Load model configuration (need to ensure this matches MODEL_NAME)
    config = BertConfig.from_pretrained(MODEL_NAME, num_labels=num_labels)
    # Store strategy in config for our custom attention layer to access
    config.mask_strategy = strategy 

    # Load model
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, config=config)

    # Patch the model if not "none" strategy
    if strategy != "none":
        model = patch_attention_model(model, strategy)
    
    # Ensure the model uses the device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, f"training_output_{strategy}"),
        logging_dir=os.path.join(LOGGING_DIR, f"logs_{strategy}"),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=50,
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=EVAL_STEPS,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",
        fp16=torch.cuda.is_available(), # Use mixed precision if GPU available
        remove_unused_columns=False, # Keep 'sentence' column for DataCollator
    )

    # Initialize custom data collator
    # Pass 'raw_datasets["train"]' to access the original 'sentence' field
    data_collator = DataCollatorWithPOS(tokenizer, nlp, MAX_SEQ_LEN)

    # Trainer will take 'input_ids', 'attention_mask', 'labels', and 'pos_tags'
    # The 'pos_tags' will be passed through to the custom attention layer
    trainer = GrammaticalTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"], # Keep original 'sentence' column
        eval_dataset=tokenized_datasets["validation"], # Keep original 'sentence' column
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # --- Training & Timing ---
    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats() # Reset memory stats before training
    train_result = trainer.train()
    end_time = time.time()
    
    training_time = end_time - start_time
    peak_memory = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    
    timing_results[strategy] = training_time
    memory_results[strategy] = peak_memory / (1024**2) # Convert to MB
    
    print("Training complete.")

    # --- Evaluation ---
    print("Evaluating model...")
    eval_results = trainer.evaluate()
    results[strategy] = eval_results
    print(f"Results for {strategy}: {eval_results}")
    
    # Optional: Save final model
    trainer.save_model(os.path.join(OUTPUT_DIR, f"final_model_{strategy}"))

print("\n--- All Experiments Complete ---")
print("Summary of Results:")
for strategy, res in results.items():
    print(f"Strategy: {strategy}, Accuracy: {res['eval_accuracy']:.4f}, F1: {res['eval_f1']:.4f}")

print("\n--- Efficiency Results ---")
for strategy in mask_strategies:
    print(f"Strategy: {strategy}, Training Time: {timing_results[strategy]:.2f}s, Peak GPU Memory: {memory_results[strategy]:.2f} MB")