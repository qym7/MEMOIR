"""
MEMOIR: Model Editing with Minimal Overwrite and Informed Retention

This module implements MEMOIR, a knowledge editing method that uses sparse fine-tuning
and conditional memory activation to efficiently edit large language models.

Key components:
- TopHasher: Implements top-k hashing for sparse feature selection
- MEMOIR: Main wrapper class that injects editing capability into models
- MEMOIRAdapter: Wrapper for the edited layer with sparsely activated residual memory
"""

import gc
import sys
import copy
from typing import Optional, List, Tuple

import torch
from torch.nn import functional as F
from torch import Tensor
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import numpy as np
import transformers
import wandb

from .utils import parent_module, brackets_to_periods, EarlyStopMeter


class TopHasher:
    """
    Top-k hashing algorithm for sparse feature selection.
    
    Selects the top-k indices with highest absolute values from input features,
    then applies a permutation to ensure diversity in selection.
    
    Attributes:
        top_k (int): Number of top indices to select
        input_dim (int): Dimension of input features
        permutation (torch.Tensor): Random permutation of input dimensions
    """
    
    def __init__(self, input_dim: int, top_k: int):
        """
        Initialize TopHasher.
        
        Args:
            input_dim: Dimension of input features
            top_k: Number of top indices to select
        """
        self.top_k = top_k
        self.input_dim = input_dim
        self.permutation = torch.randperm(input_dim)
        print(f"TopHasher: selecting {top_k} feature indices from {input_dim} dimensions")
    
    def get_active_indices(self, x: Tensor) -> Tensor:
        """
        Select top-k active indices based on absolute values, and apply permutation to ensure diversity.
        
        Args:
            x: Input features of shape (input_dim)
            
        Returns:
            Selected indices of shape (top_k)
            after applying the permutation
        """
        assert x.shape[0] == self.input_dim, "Input dimension must match the input dimension of the TopHasher"
        
        # Select top-k indices by absolute value
        _, indices = torch.topk(x.abs(), self.top_k)
        # Apply permutation to ensure diversity
        indices = self.permutation[indices.to(self.permutation.device)]
        
        return indices.detach().cpu()


class MEMOIR(torch.nn.Module):
    """
    MEMOIR lifelong model editing framework for language models.
    
    This class wraps a base model and adds a MEMOIRAdapter to a specified layer.
    
    Attributes:
        config: Configuration object containing hyperparameters
        model: Base language model to be edited
        device: Device to run computations on
        layer: Layer name/path where adapter is injected
        edit_module: Parent module containing the target layer
        layer_name: Name of the target layer
        original_layer: Backup of original layer
    """
    
    def __init__(self, config, model, device):
        """
        Initialize MEMOIR wrapper.
        
        Args:
            config: Configuration object with hyperparameters
            model: Base language model to wrap
            device: Device identifier
        """
        super(MEMOIR, self).__init__()
        self.config = config
        self.model = model
        self.device = device
        
        # Extract layer information from config
        layer = config.inner_params[0]
        
        # Ensure proper formatting 
        suffixes = [".weight", ".bias"]
        self.layer = layer.rsplit(".", 1)[0] if any(layer.endswith(x) for x in suffixes) else layer
        
        # Freeze all model parameters
        for n, p in self.model.named_parameters():
            p.requires_grad = False
        
        # Determine if layer needs transpose (only for GPT2 model)
        if isinstance(self.model, transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel):
            transpose = False
        else:
            transpose = True
        
        # Locate the target module and layer
        self.edit_module = parent_module(self.model, brackets_to_periods(self.layer))
        self.layer_name = self.layer.rsplit(".", 1)[-1]
        adapter_layer = getattr(self.edit_module, self.layer_name)
        
        # Replace original layer with MEMOIRAdapter if not already replaced
        if type(adapter_layer) is not MEMOIRAdapter:
            setattr(self.edit_module, self.layer_name, 
                   MEMOIRAdapter(config, adapter_layer, transpose=transpose))
            self.original_layer = copy.deepcopy(adapter_layer)
            print(f"Successfully inserted MEMOIR adapter into layer: {layer}")
        
        # Clean up memory
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
    
    def __call__(self, **kwargs):
        """
        Forward pass function.
        """
        adapter_layer = self.get_adapter_layer()
        
        # Extract and set the location of the last prompt token during inference 
        prompt_boundary = kwargs.pop("last_prompt_token_loc_inference", None)
        if prompt_boundary is not None:
            # Temporarily set boundary on adapter for this forward pass
            setattr(adapter_layer, "last_prompt_token_loc_inference", prompt_boundary)
        
        try:
            return self.model(**kwargs)
        finally:
            # Clean up temporary attribute after forward pass
            if prompt_boundary is not None and hasattr(adapter_layer, "last_prompt_token_loc_inference"):
                delattr(adapter_layer, "last_prompt_token_loc_inference")
    
    def get_adapter_layer(self) -> 'MEMOIRAdapter':
        """
        Get the MEMOIRAdapter instance.
        """
        adapter_layer = getattr(self.edit_module, self.layer_name)
        assert type(adapter_layer) is MEMOIRAdapter, \
            'Adapter Layer is not added correctly. Expected MEMOIRAdapter.'
        return adapter_layer.to(self.model.device)
    
    def reset_layer(self):
        """
        Restore the original layer.
        """
        layer = getattr(self.edit_module, self.layer_name)
        del layer
        setattr(self.edit_module, self.layer_name, self.get_adapter_layer().original_layer)
    
    def edit(self, config, tokens, **kwargs):
        """
        Perform knowledge editing on the model.
        
        This method trains a residual memory (new_weight) that is added to the
        original layer output on the edited samples. 
        """
        # Set adapter to training mode
        setattr(eval(f"self.model.{self.layer}"), "training", True)
        self.get_adapter_layer().set_parameter_tunable()
        
        # Compute the last prompt token location from labels
        # Labels are -100 for prompt tokens, so we count non--100 tokens to get target length
        # Then subtract 1 to get the position of the last prompt token
        last_prompt_token_loc = (tokens["labels"] == -100).sum(dim=-1) - 1
        self.get_adapter_layer().last_prompt_token_loc = last_prompt_token_loc
        
        # Initialize training
        loss_meter = EarlyStopMeter()
        optimizer = None
        
        # NOT RELEVANT FOR EDITING: If running the mode to save background features, just perform a single forward pass
        if config.RUN_SAVE_BACKGROUND_FEATURES:
            _ = self.model(**tokens)
            setattr(eval(f"self.model.{self.layer}"), "training", False)
            delattr(self.get_adapter_layer(), "last_prompt_token_loc")
            return
        
        # Fine-tune the residual memory weights
        for i in range(config.n_iter):
            # Initialize optimizer on first iteration
            if i == 0:
                optimizer = torch.optim.SGD(
                    [self.get_adapter_layer().new_weight], 
                    config.edit_lr, 
                    weight_decay=1e-5
                )
            
            # Compute fine-tuning loss
            loss = self._cal_ft_loss(tokens, last_prompt_token_loc)
            
            # Early stopping check
            if loss_meter.stop():
                break
            
            # Log to wandb if available
            if wandb.run:
                wandb.log({'train/loss': loss.item()}, commit=True)
            
            # Optimizer step
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.get_adapter_layer().new_weight, 
                max_norm=1.0
            )
            
            print(f"Iteration {i}: loss = {np.round(loss.item(), 4)}")
            
            optimizer.step()
            loss_meter.update(loss.item())
        
        # Clean up training mode
        setattr(eval(f"self.model.{self.layer}"), "training", False)
        delattr(self.get_adapter_layer(), "last_prompt_token_loc")
    
    def _cal_ft_loss(self, tokens, last_prompt_token_loc: torch.Tensor) -> torch.Tensor:
        """
        Calculate fine-tuning loss computed only on target tokens.
        """
        # Determine batch size (excluding irrelevant samples)
        if hasattr(self.model.config, 'batch_size'):
            k = self.config.batch_size
        else:
            k = 1
        
        bs = tokens["input_ids"].shape[0] - k  # Batch size excluding irrelevant samples (not used for training)
        
        # Forward pass to get logits
        logits = self.model(**tokens).logits
        
        # Shift logits and labels for next-token prediction
        shift_logits = logits[:-k, :-1, :].contiguous()
        shift_labels = tokens['labels'][:-k, 1:].contiguous()
        
        # Compute cross-entropy loss per token
        loss_fct = CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                       shift_labels.view(-1))
        loss = loss.view(bs, -1)
        
        # Create mask to only compute loss on target tokens (after prompt)
        label_mask = torch.zeros_like(loss, dtype=torch.bool)
        for i, col_index in enumerate(last_prompt_token_loc[:-k]):
            label_mask[i, col_index - 1:] = True  # Mask from last prompt token onwards
        
        # Compute mean loss only on target tokens
        ft_loss = ((loss * label_mask).sum(1) / label_mask.sum(1)).mean()
        return ft_loss


class MEMOIRAdapter(torch.nn.Module):
    """
    MEMOIR adapter layer that implements MEMOIR memory module for knowledge editing.
    
    This adapter wraps an original linear layer and adds a sparse residual memory
    that is conditionally activated based on feature matching during inference. 
    During training, it saves activation masks for edited samples. 
    During inference, it matches current sample to saved samples via overlap ratio 
    and only activates the residual memory if the sample is relevant to edited samples.
    
    Attributes:
        layer: Original layer being wrapped
        weight: Original layer weights
        new_weight: Learned residual memory weights
        original_layer: Backup of original layer
        hasher: TopHasher instance for sparse feature selection
        masks_for_edited_samples: List of saved activation masks from edited samples
        loaded_irrelevant_sample_mean_features: Pre-computed mean background features from irrelevant samples
        prompt_feature_agg: Strategy for aggregating prompt token features
    """
    
    def __init__(self, config, layer, transpose: bool):
        """
        Initialize MEMOIRAdapter.
        
        Args:
            config: Configurations
            layer: Original layer to wrap
            transpose: Whether layer weights need transposition
        """
        super(MEMOIRAdapter, self).__init__()
        
        # select the prompt feature aggregation strategy (default: mean_decentered)
        assert config.prompt_feature_agg in ['last', 'mean', 'mean_decentered'], \
            f"Unknown prompt feature aggregation strategy: {config.prompt_feature_agg}. "
        self.prompt_feature_agg = config.prompt_feature_agg

        # Store original layer and its properties
        self.layer = layer
        self.weight = self.layer.weight
        self.device = layer.weight.device
        self.config = config
        
        # Initialize residual memory (zero-initialized)
        self.new_weight = copy.deepcopy(self.weight)
        self.new_weight.data.zero_()
        
        # Backup original layer
        self.original_layer = copy.deepcopy(self.layer)
        
        # Handle bias for different layer types
        if 'gpt2' in self.config.model_name:
            self.bias = self.layer.bias
        else:
            self.bias = None
        
        assert not self.weight.requires_grad, \
            'Original layer weights should not be trainable.'
        
        # Set training state
        self.training = False
        
        # Initialize TopHasher for sparse feature selection
        self.hasher = TopHasher(self.new_weight.shape[1], top_k=self.config.top_k)
        
        # List for storing activation masks from edited samples
        self.masks_for_edited_samples = []
        
        # Load pre-computed background features from irrelevant samples
        # These are used for de-centering features in 'mean_decentered' aggregation
        print(f"Loading background features for model {config.model_name} from directory: {config.dir_background_features}")
        self.loaded_irrelevant_sample_features = torch.load(
            config.dir_background_features
        )[:100, :] # we load 100 irrelevant samples to compute the mean background features
        # Compute mean of background features for de-centering
        self.loaded_irrelevant_sample_mean_features = torch.mean(self.loaded_irrelevant_sample_features, dim=0).to('cuda')
        assert self.loaded_irrelevant_sample_mean_features.shape[0] == layer.weight.shape[1]

        # If running the mode to save background features, initialize the list to save the background features
        if self.config.RUN_SAVE_BACKGROUND_FEATURES:
            self.saved_background_features = []

    def set_parameter_tunable(self):
        """Enable gradient computation for residual memory weights."""
        self.new_weight.requires_grad = True
    
    def aggregate_prompt_features(self, x: Tensor, prompt_boundary: int, padding_counts: int, verbose: bool = False) -> Tensor:
        """
        Aggregate prompt features based on the aggregation strategy.
        Args:
            x: Input features of shape (B, S, D)
            prompt_boundary: Position of the last prompt token
            padding_counts: Number of padding tokens
        Returns:
            Aggregated prompt features of shape (B, D)
        """
        if verbose:
            print(f"Prompt features aggregated from token {padding_counts} to token "
                  f"{prompt_boundary} (out of {x.shape[1]} total tokens)")

        # Strategy 1: Use only the last prompt token
        if self.prompt_feature_agg == 'last':
            return x[:, prompt_boundary, :]
        # Strategy 2: Use mean of all prompt tokens
        elif self.prompt_feature_agg == 'mean':
            return x[:, padding_counts:prompt_boundary+1, :].mean(1)
        # Strategy 3: Use mean of prompt tokens, de-centered by pre-computed background features
        elif self.prompt_feature_agg == 'mean_decentered':
            return (x[:, padding_counts:prompt_boundary+1, :].mean(1) - self.loaded_irrelevant_sample_mean_features.unsqueeze(0))
    
    def count_padding_tokens(self, x: Tensor) -> int:
        """
        Count the number of padding tokens in the input features.
        Args:
            x: Input features of shape (S, D)
        Returns:
            Number of padding tokens
        """
        # detect padding by comparing to first token
        # (first token is always padding due to data augmentation which adds random prefix tokens)
        padding_vec = x[0, :].clone()  # shape [D]
        is_padding = (x == padding_vec.unsqueeze(0)).all(dim=1)  # shape [S]
        padding_counts = is_padding.sum(dim=0) 
        return padding_counts

    def new_weight_forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through for the memory module.
        
        Steps:
        1. Extract features from prompt tokens
        2. Use TopHash to select sparse active indices
        3. Sparsify the features and forward pass through the residual memory
        
        Note that it also performs conditional knowledge activation during inference.

        Args:
            x: Input features of shape (B, S, D), where:
               - B: batch size (during training: 12 = 1 edit + 10 augmented + 1 irrelevant; during inference: 1)
               - S: sequence length (max tokens in batch)
               - D: feature dimension
        
        Returns:
            Output features of shape (B, S, output_dimension) from residual memory
            (returns zeros if sample is identified as irrelevant during inference)
        """
        B, S, D = x.shape  # batch-size, sequence length, feature dimension
        
        # First, we determine the start and end of the prompt tokens (only for the first sample in the batch)

        # Determine prompt boundary (position of last prompt token)
        if self.training:
            # During training: use last_prompt_token_loc set during edit()
            assert not hasattr(self, "last_prompt_token_loc_inference") or getattr(self, "last_prompt_token_loc_inference") is None
            prompt_boundary = self.last_prompt_token_loc[0]
        else:
            # During inference: use last_prompt_token_loc_inference set via __call__
            assert not hasattr(self, "last_prompt_token_loc") or getattr(self, "last_prompt_token_loc") is None
            prompt_boundary = self.last_prompt_token_loc_inference[0]
        
        # Determine padding token length
        if self.training:
            # During training: count the number of padding tokens
            # We only consider the first sample in the batch (original edit sample)
            padding_counts = self.count_padding_tokens(x[0])
        else:
            # During inference: no padding in current setup
            padding_counts = 0
        
        # Aggregate features from prompt tokens based on prompt feature aggregation strategy
        prompt_agg_features = self.aggregate_prompt_features(x, prompt_boundary, padding_counts)
        
        # NOT RELEVANT FOR EDITING: Running on the mode to only save background features (rather than perform editing)
        if self.config.RUN_SAVE_BACKGROUND_FEATURES:
            # Save the background features and return zero output (no need for fine-tuning)
            # Note that we save the background features for the last sample in the batch (irrelevant sample)
            self.func_save_background_features(prompt_agg_features[-1])
            return torch.zeros_like(F.linear(x, self.new_weight))

        # Use TopHash to select sparse active indices from aggregated features
        # We obtain only indices for the first sample (original edit sample during training, 
        # or the only sample during inference), and then apply it to all samples in the batch
        active_indices = self.hasher.get_active_indices(prompt_agg_features[0])
        
        # Convert indices to boolean mask for efficient storage and computation
        active_parameter_mask = torch.zeros(D, dtype=torch.bool, device=x.device)
        active_parameter_mask[active_indices] = True
        
        # Training phase: save activation masks for edited samples
        if self.training:
            # Save mask only once per unique edit sample (avoid duplicates across iterations)
            if all(not torch.equal(active_parameter_mask, prev) 
                   for prev in self.masks_for_edited_samples):
                if len(self.masks_for_edited_samples) == 0:
                    self.masks_for_edited_samples = active_parameter_mask.view(1, -1)
                else:
                    self.masks_for_edited_samples = torch.vstack(
                        (self.masks_for_edited_samples, active_parameter_mask)
                    )
        # Inference phase: match current sample to saved samples and perform conditional knowledge activation
        else:
            # Compute overlapping counts with each saved sample mask
            overlapping_counts = torch.matmul(
                active_parameter_mask.float(), 
                self.masks_for_edited_samples.T.float()
            )  # shape [M]
            
            # Compute overlap ratio (overlap / total_active_indices)
            overlap_ratios = overlapping_counts / self.config.top_k
            
            # Find top match
            best_vals, best_idxs = torch.topk(overlap_ratios, k=1)
            best_overlap = best_vals.item()
            best_match_idx = best_idxs.item()
            
            print(f"Mask overlap ratio with closest edited sample: {best_overlap:.4f} "
                  f"(match sample #{best_match_idx})")
            
            # Conditional Knowledge Activation
            if best_overlap >= self.config.irr_threshold:
                # case 1: best overlap ratio = 1.0, identify as a previously edited sample
                if best_overlap == 1.0: 
                    print(f"Identified as a previously edited sample #{best_match_idx}")
                # case 2: irr_threshold <= best overlap ratio < 1.0, identify as a rephrased sample
                else: 
                    print(f"Identified as rephrased sample (from sample #{best_match_idx})")
                
                # Replace active indices with best-matched sample's indices
                # (if exact match, it is the same indices; if rephrased, we use the matched sample's indices)
                best_match_mask = self.masks_for_edited_samples[best_match_idx].to(
                    active_indices.device
                )
                active_indices = torch.where(best_match_mask > 0.5)[0] # convert boolean mask to indices
            else:
                # case 3: best overlap ratio < self.config.irr_threshold, identify as an irrelevant sample
                print(f"Identified as irrelevant sample; deactivating residual memory")
                # return zero outputs for this sample (deactivate residual memory)
                down_out = torch.zeros_like(F.linear(x, self.new_weight))
                return down_out
        
        # Create sparse feature tensor by zeroing out non-active features
        x_hashed = torch.zeros_like(x)  # shape (B, S, D)
        x_hashed[:, :, active_indices] = x[:, :, active_indices]
        
        sparsity = (x_hashed == x).float().mean()
        print(f"After hashing, feature sparsity: {sparsity:.4f}.")
        
        # Forward pass through residual memory with sparsified features
        down_out = F.linear(x_hashed, self.new_weight)
        return down_out
    
    def forward(self, *args) -> Tensor:
        """
        Forward pass with the edited layer.
        
        The output is the sum of original layer output and residual memory output (with conditionally activated knowledge during inference).

        """
        # Original layer output + residual memory output
        layer_out = self.original_layer(*args) + self.new_weight_forward(*args)
        return layer_out

    def func_save_background_features(self, irrelevant_sample_features: Tensor):
        """
        Function to iterate over irrelevant samples and save the background features.
        """
        if self.training:
            # save the aggregated features from irrelevant samples
            # we take only the last sample from the batch (it is an irrelevant sample to the edit samples)
            self.saved_background_features.append(irrelevant_sample_features.unsqueeze(0).detach().cpu()) 
        else:
            # save the background features after iterating over all samples and exit the program
            self.saved_background_features = torch.cat(self.saved_background_features, dim=0)
            torch.save(self.saved_background_features, self.config.dir_to_save_background_features)
            print(f"Background features saved to {self.config.dir_to_save_background_features}. Exiting program.")
            sys.exit(0)
