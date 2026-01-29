import math
import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertSelfAttention, BertConfig

# --- Define Grammatical Rules ---
# These are illustrative. You MUST expand and refine them based on linguistic theory
# and potentially iterative experimentation. Use UPENN Treebank tags or Universal POS tags.
# SpaCy's .pos_ gives Universal POS tags (NOUN, VERB, ADJ, ADP, DET, etc.)

# Hard Rules: Query POS can attend to Key POS (binary mask)
# Set to True if Q_POS -> K_POS connection is ALLOWED
HARD_GRAMMAR_RULES = {
    # Self-attention is always allowed for consistency and local context
    (p, p): True for p in ["NOUN", "VERB", "ADJ", "ADP", "DET", "PRON", "ADV", "AUX", "CCONJ", "SCONJ", "INTJ", "NUM", "PART", "PROPN", "PUNCT", "SYM", "X", "[CLS]", "[SEP]", "[PAD]", "[UNK]"]
}
HARD_GRAMMAR_RULES.update({
    ("ADJ", "NOUN"): True, ("ADJ", "PROPN"): True, # Adjective modifies Noun/Proper Noun
    ("DET", "NOUN"): True, ("DET", "PROPN"): True, # Determiner precedes Noun/Proper Noun
    ("ADP", "NOUN"): True, ("ADP", "PROPN"): True, ("ADP", "PRON"): True, # Preposition governs Noun/Proper Noun/Pronoun
    ("NOUN", "DET"): True, ("NOUN", "ADJ"): True, ("NOUN", "ADP"): True, # Noun can be described by Det/Adj/Prep-phrase
    ("PROPN", "DET"): True, ("PROPN", "ADJ"): True, ("PROPN", "ADP"): True,
    ("PRON", "ADJ"): True, ("PRON", "ADP"): True,
    ("VERB", "NOUN"): True, ("VERB", "PROPN"): True, ("VERB", "PRON"): True, # Verb to Noun/Pronoun (subject/object)
    ("VERB", "ADV"): True, # Verb to Adverb
    ("AUX", "VERB"): True, # Auxiliary verb with main verb
    ("SCONJ", "VERB"): True, # Subordinating conjunction introducing a verb clause
})

# [CLS] and [SEP] should often attend to everything to gather context.
# [CLS] to all:
HARD_GRAMMAR_RULES.update({
    ("[CLS]", p): True for p in ["NOUN", "VERB", "ADJ", "ADP", "DET", "PRON", "ADV", "AUX", "CCONJ", "SCONJ", "INTJ", "NUM", "PART", "PROPN", "PUNCT", "SYM", "X", "[SEP]", "[UNK]", "[PAD]"]
})


# Soft Rules: Query POS to Key POS adds a bias (additive mask)
# Set to True if Q_POS -> K_POS connection gets a positive bias
SOFT_GRAMMAR_RULES = {
    ("VERB", "ADP"): True, # Verb followed by prepositional phrase (e.g., "look at")
    ("NOUN", "VERB"): True, ("PROPN", "VERB"): True, ("PRON", "VERB"): True, # Noun/Pronoun as subject of verb
    ("ADV", "VERB"): True, ("ADV", "ADJ"): True, ("ADV", "ADV"): True, # Adverb modifies Verb/Adjective/Adverb
    ("CCONJ", "NOUN"): True, ("CCONJ", "VERB"): True, # Conjunctions connecting parts of speech
    ("CCONJ", "ADJ"): True, ("CCONJ", "PROPN"): True, ("CCONJ", "PRON"): True,
}
SOFT_GRAMMAR_RULES.update({
    ("[CLS]", p): True for p in ["NOUN", "VERB", "ADJ", "ADP", "DET", "PRON", "ADV", "AUX", "CCONJ", "SCONJ", "INTJ", "NUM", "PART", "PROPN", "PUNCT", "SYM", "X", "[SEP]", "[UNK]", "[PAD]"]
})

# --- Custom BertSelfAttention Layer ---
class GrammaticallyGuidedBertSelfAttention(BertSelfAttention):
    def __init__(self, config: BertConfig, is_decoder: bool = False):
        super().__init__(config)
        self.config = config
        self.hard_rules = HARD_GRAMMAR_RULES
        self.soft_rules = SOFT_GRAMMAR_RULES
        self.soft_bias_strength = 5.0 # Hyperparameter for soft bias strength
        self.current_pos_tags = None

    def set_current_pos_tags(self, pos_tags):
        self.current_pos_tags = pos_tags

    def create_grammatical_mask(self, batch_pos_tags: list[list[str]], seq_len: int, mask_strategy: str, device: torch.device, dtype: torch.dtype):
        """
        Generates a grammatical attention mask for a batch of sequences.
        
        Args:
            batch_pos_tags: A list of lists of POS tags, where each inner list corresponds
                            to a sequence in the batch.
            seq_len: The maximum sequence length (including special tokens).
            mask_strategy: "hard", "soft", or "none".
            device: The torch device for the mask tensor.
            dtype: The torch dtype for the mask tensor.
        Returns:
            A torch.Tensor of shape (batch_size, 1, seq_len, seq_len)
            containing the grammatical mask.
        """
        batch_size = len(batch_pos_tags)
        
        # Initialize mask. For hard, fill with large negative value to block attention.
        # For soft, fill with 0, then add positive bias.
        grammatical_mask = torch.full(
            (batch_size, seq_len, seq_len), 
            -10000.0 if mask_strategy == "hard" else 0.0, 
            device=device, dtype=dtype
        )
        
        for b in range(batch_size):
            current_pos_tags = batch_pos_tags[b]
            
            # Pad or truncate POS tags to match seq_len
            if len(current_pos_tags) < seq_len:
                current_pos_tags.extend(["[PAD]"] * (seq_len - len(current_pos_tags)))
            elif len(current_pos_tags) > seq_len:
                current_pos_tags = current_pos_tags[:seq_len]

            for i in range(seq_len): # Query token index
                query_pos = current_pos_tags[i]
                
                for j in range(seq_len): # Key token index
                    key_pos = current_pos_tags[j]
                    
                    is_hard_allowed = self.hard_rules.get((query_pos, key_pos), False)
                    is_soft_allowed = self.soft_rules.get((query_pos, key_pos), False)

                    if mask_strategy == "hard":
                        if is_hard_allowed:
                            grammatical_mask[b, i, j] = 0.0 # Allow connection
                        # else: stays -10000.0, effectively blocking
                    elif mask_strategy == "soft":
                        if is_hard_allowed: # Hard rules override soft, always allowed (0 bias)
                             grammatical_mask[b, i, j] = 0.0
                        elif is_soft_allowed:
                            grammatical_mask[b, i, j] = self.soft_bias_strength # Add positive bias
                        # else: stays 0.0, no special bias added

        # Unsqueeze to add head dimension, then expand to match num_heads
        return grammatical_mask.unsqueeze(1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,  # Original padding mask (batch_size, 1, 1, seq_len)
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        pos_tags_batch=None, # NEW: List of lists of POS tags for the batch
        mask_strategy: str = None # NEW: "hard", "soft", or "none"
    ):
        if mask_strategy is None:
            mask_strategy = getattr(self.config, "mask_strategy", "hard")

        # Use stored pos_tags if not provided
        if pos_tags_batch is None:
            pos_tags_batch = self.current_pos_tags

        # --- Standard QKV calculation (same as original BertSelfAttention) ---
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # Shape: (batch_size, num_heads, seq_len, seq_len)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # --- Apply Grammatical Masking ---
        if pos_tags_batch is not None and mask_strategy != "none":
            batch_size, num_heads, seq_len, _ = attention_scores.shape
            
            # Create the grammatical mask (batch_size, 1, seq_len, seq_len)
            grammatical_mask_tensor = self.create_grammatical_mask(
                batch_pos_tags=pos_tags_batch,
                seq_len=seq_len,
                mask_strategy=mask_strategy,
                device=attention_scores.device,
                dtype=attention_scores.dtype
            )
            
            # Expand grammatical_mask_tensor to match attention_scores' head dimension
            # (batch_size, num_heads, seq_len, seq_len)
            grammatical_mask_tensor = grammatical_mask_tensor.expand_as(attention_scores)

            attention_scores = attention_scores + grammatical_mask_tensor
            
        # --- Apply original padding attention mask (always, if provided) ---
        if attention_mask is not None:
            # attention_mask shape: (batch_size, 1, 1, seq_len) (for padding)
            # HuggingFace attention_mask automatically handles this via broadcasting
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # ... (rest of the standard BertSelfAttention forward) ...
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs