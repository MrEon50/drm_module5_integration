"""torch_integration.py
Enhanced PyTorch integration for DRM system with full gradient support.

Features:
- Differentiable rule parameters with autograd
- Neural network-based rule generators
- Gradient-based optimization for HYBRID rules
- Batch processing and GPU support
- Advanced loss functions for rule learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import logging
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for PyTorch training"""
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    device: str = "auto"  # "auto", "cpu", "cuda"
    optimizer: str = "adam"  # "adam", "sgd", "rmsprop"
    scheduler: Optional[str] = "cosine"  # "cosine", "step", None
    gradient_clip: Optional[float] = 1.0
    weight_decay: float = 1e-4
    early_stopping_patience: int = 10
    validation_split: float = 0.2

class DifferentiableRule(nn.Module):
    """PyTorch module for differentiable rule parameters"""
    
    def __init__(self, rule_id: str, param_specs: Dict[str, Dict[str, Any]]):
        super().__init__()
        self.rule_id = rule_id
        self.param_specs = param_specs
        
        # Create learnable parameters
        self.params = nn.ParameterDict()
        self.constraints = {}
        
        for name, spec in param_specs.items():
            if spec.get("requires_grad", False):
                init_value = spec.get("value", 0.0)
                self.params[name] = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))
                
                # Store constraints for clamping
                self.constraints[name] = {
                    "min": spec.get("min_val"),
                    "max": spec.get("max_val"),
                    "constraint_fn": spec.get("constraint_fn")
                }
    
    def forward(self, context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass returning clamped parameters"""
        output = {}
        
        for name, param in self.params.items():
            # Apply constraints
            constrained_param = param
            
            if name in self.constraints:
                constraint = self.constraints[name]
                if constraint["min"] is not None:
                    constrained_param = torch.clamp(constrained_param, min=constraint["min"])
                if constraint["max"] is not None:
                    constrained_param = torch.clamp(constrained_param, max=constraint["max"])
            
            output[name] = constrained_param
        
        return output
    
    def get_parameter_dict(self) -> Dict[str, float]:
        """Get current parameter values as regular dict"""
        with torch.no_grad():
            return {name: param.item() for name, param in self.params.items()}

class NeuralRuleGenerator(nn.Module):
    """Neural network-based rule generator"""
    
    def __init__(self, latent_dim: int = 64, hidden_dim: int = 128, 
                 output_dim: int = 32, num_layers: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # Encoder for context -> latent
        encoder_layers = []
        in_dim = output_dim  # Context dimension
        
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else latent_dim
            encoder_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.ReLU() if i < num_layers - 1 else nn.Tanh()
            ])
            in_dim = out_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder for latent -> rule parameters
        decoder_layers = []
        in_dim = latent_dim
        
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else output_dim
            decoder_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.ReLU() if i < num_layers - 1 else nn.Sigmoid()
            ])
            in_dim = out_dim
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Performance predictor
        self.performance_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def encode(self, context: torch.Tensor) -> torch.Tensor:
        """Encode context to latent representation"""
        return self.encoder(context)
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to rule parameters"""
        return self.decoder(latent)
    
    def predict_performance(self, latent: torch.Tensor) -> torch.Tensor:
        """Predict rule performance from latent"""
        return self.performance_head(latent)
    
    def forward(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass"""
        latent = self.encode(context)
        params = self.decode(latent)
        performance = self.predict_performance(latent)
        return params, performance, latent

class RuleLoss(nn.Module):
    """Advanced loss function for rule learning"""
    
    def __init__(self, performance_weight: float = 1.0, 
                 diversity_weight: float = 0.1,
                 consistency_weight: float = 0.2,
                 sparsity_weight: float = 0.05):
        super().__init__()
        self.performance_weight = performance_weight
        self.diversity_weight = diversity_weight
        self.consistency_weight = consistency_weight
        self.sparsity_weight = sparsity_weight
    
    def forward(self, predicted_performance: torch.Tensor,
                actual_performance: torch.Tensor,
                rule_params: torch.Tensor,
                latent_representations: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute multi-component loss"""
        
        # Performance loss (main objective)
        performance_loss = F.mse_loss(predicted_performance, actual_performance)
        
        # Diversity loss (encourage diverse latent representations)
        diversity_loss = torch.tensor(0.0, device=predicted_performance.device)
        if latent_representations is not None and latent_representations.size(0) > 1:
            # Compute pairwise cosine similarities
            normalized = F.normalize(latent_representations, p=2, dim=1)
            similarity_matrix = torch.mm(normalized, normalized.t())
            # Penalize high similarities (encourage diversity)
            diversity_loss = torch.mean(torch.triu(similarity_matrix, diagonal=1) ** 2)
        
        # Consistency loss (penalize extreme parameter values)
        consistency_loss = torch.mean(torch.abs(rule_params - 0.5))
        
        # Sparsity loss (encourage sparse parameter usage)
        sparsity_loss = torch.mean(torch.abs(rule_params))
        
        # Total loss
        total_loss = (self.performance_weight * performance_loss +
                     self.diversity_weight * diversity_loss +
                     self.consistency_weight * consistency_loss +
                     self.sparsity_weight * sparsity_loss)
        
        return {
            "total": total_loss,
            "performance": performance_loss,
            "diversity": diversity_loss,
            "consistency": consistency_loss,
            "sparsity": sparsity_loss
        }

class PyTorchDRMTrainer:
    """Main trainer class for PyTorch-enhanced DRM"""
    
    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.device = self._get_device()
        
        # Training components
        self.differentiable_rules: Dict[str, DifferentiableRule] = {}
        self.rule_generator: Optional[NeuralRuleGenerator] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
        self.loss_fn = RuleLoss()
        
        # Training history
        self.training_history = defaultdict(list)
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        logger.info(f"PyTorchDRMTrainer initialized on device: {self.device}")
    
    def _get_device(self) -> torch.device:
        """Determine the best available device"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(self.config.device)
    
    def register_rule(self, rule_id: str, param_specs: Dict[str, Dict[str, Any]]):
        """Register a rule for differentiable training"""
        diff_rule = DifferentiableRule(rule_id, param_specs).to(self.device)
        self.differentiable_rules[rule_id] = diff_rule
        logger.info(f"Registered differentiable rule: {rule_id}")
    
    def initialize_generator(self, latent_dim: int = 64, hidden_dim: int = 128):
        """Initialize neural rule generator"""
        self.rule_generator = NeuralRuleGenerator(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim
        ).to(self.device)
        logger.info(f"Initialized neural rule generator with latent_dim={latent_dim}")
    
    def setup_optimization(self):
        """Setup optimizer and scheduler"""
        # Collect all parameters
        params = []
        
        # Add rule parameters
        for rule in self.differentiable_rules.values():
            params.extend(rule.parameters())
        
        # Add generator parameters
        if self.rule_generator is not None:
            params.extend(self.rule_generator.parameters())
        
        # Setup optimizer
        if self.config.optimizer == "adam":
            self.optimizer = optim.Adam(params, lr=self.config.learning_rate, 
                                      weight_decay=self.config.weight_decay)
        elif self.config.optimizer == "sgd":
            self.optimizer = optim.SGD(params, lr=self.config.learning_rate, 
                                     weight_decay=self.config.weight_decay, momentum=0.9)
        elif self.config.optimizer == "rmsprop":
            self.optimizer = optim.RMSprop(params, lr=self.config.learning_rate,
                                         weight_decay=self.config.weight_decay)
        
        # Setup scheduler
        if self.config.scheduler == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.epochs)
        elif self.config.scheduler == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=self.config.epochs // 3, gamma=0.1)
        
        logger.info(f"Setup optimization: {self.config.optimizer} optimizer, "
                   f"{self.config.scheduler} scheduler")

    def train_step(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.optimizer.zero_grad()

        # Extract batch data
        rule_ids = batch_data["rule_ids"]
        contexts = batch_data["contexts"]
        actual_performance = batch_data["performance"].to(self.device)

        # Forward pass through rules
        predicted_performance = []
        rule_params_list = []
        latent_list = []

        for i, rule_id in enumerate(rule_ids):
            if rule_id in self.differentiable_rules:
                rule = self.differentiable_rules[rule_id]
                params = rule(contexts[i:i+1] if contexts is not None else None)

                # Simple performance prediction based on parameters
                param_values = torch.stack(list(params.values()))
                perf = torch.mean(param_values).unsqueeze(0)
                predicted_performance.append(perf)
                rule_params_list.append(param_values)

        if not predicted_performance:
            return {"total": 0.0}

        predicted_performance = torch.stack(predicted_performance)
        rule_params = torch.stack(rule_params_list)

        # Compute loss
        loss_dict = self.loss_fn(
            predicted_performance=predicted_performance,
            actual_performance=actual_performance,
            rule_params=rule_params,
            latent_representations=torch.stack(latent_list) if latent_list else None
        )

        # Backward pass
        loss_dict["total"].backward()

        # Gradient clipping
        if self.config.gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for rule in self.differentiable_rules.values() for p in rule.parameters()],
                self.config.gradient_clip
            )

        self.optimizer.step()

        # Convert to float for logging
        return {k: v.item() if isinstance(v, torch.Tensor) else v
                for k, v in loss_dict.items()}

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        epoch_losses = defaultdict(list)

        for batch in dataloader:
            loss_dict = self.train_step(batch)
            for k, v in loss_dict.items():
                epoch_losses[k].append(v)

        # Average losses
        return {k: np.mean(v) for k, v in epoch_losses.items()}

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validation step"""
        val_losses = defaultdict(list)

        with torch.no_grad():
            for batch in dataloader:
                # Similar to train_step but without gradients
                rule_ids = batch["rule_ids"]
                contexts = batch["contexts"]
                actual_performance = batch["performance"].to(self.device)

                predicted_performance = []
                rule_params_list = []

                for i, rule_id in enumerate(rule_ids):
                    if rule_id in self.differentiable_rules:
                        rule = self.differentiable_rules[rule_id]
                        params = rule(contexts[i:i+1] if contexts is not None else None)

                        param_values = torch.stack(list(params.values()))
                        perf = torch.mean(param_values).unsqueeze(0)
                        predicted_performance.append(perf)
                        rule_params_list.append(param_values)

                if predicted_performance:
                    predicted_performance = torch.stack(predicted_performance)
                    rule_params = torch.stack(rule_params_list)

                    loss_dict = self.loss_fn(
                        predicted_performance=predicted_performance,
                        actual_performance=actual_performance,
                        rule_params=rule_params
                    )

                    for k, v in loss_dict.items():
                        val_losses[k].append(v.item())

        return {k: np.mean(v) for k, v in val_losses.items()}

    def train(self, training_data: List[Dict[str, Any]],
              validation_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, List[float]]:
        """Main training loop"""

        # Prepare data
        train_loader = self._prepare_dataloader(training_data, shuffle=True)
        val_loader = None
        if validation_data:
            val_loader = self._prepare_dataloader(validation_data, shuffle=False)

        # Setup optimization
        self.setup_optimization()

        logger.info(f"Starting training for {self.config.epochs} epochs")

        for epoch in range(self.config.epochs):
            # Training
            train_losses = self.train_epoch(train_loader)

            # Validation
            val_losses = {}
            if val_loader:
                val_losses = self.validate(val_loader)

            # Scheduler step
            if self.scheduler:
                self.scheduler.step()

            # Logging
            self.training_history["train_loss"].append(train_losses.get("total", 0.0))
            if val_losses:
                self.training_history["val_loss"].append(val_losses.get("total", 0.0))

            # Early stopping
            current_loss = val_losses.get("total", train_losses.get("total", 0.0))
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

            # Progress logging
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: train_loss={train_losses.get('total', 0.0):.4f}, "
                           f"val_loss={val_losses.get('total', 0.0):.4f}")

        return dict(self.training_history)

    def _prepare_dataloader(self, data: List[Dict[str, Any]], shuffle: bool = True) -> DataLoader:
        """Prepare PyTorch DataLoader from training data"""
        rule_ids = [item["rule_id"] for item in data]
        performance = torch.tensor([item["performance"] for item in data], dtype=torch.float32)
        contexts = None

        if "context" in data[0]:
            # Convert contexts to tensors (simplified)
            contexts = torch.stack([
                torch.tensor(list(item["context"].values())[:32], dtype=torch.float32)
                for item in data
            ])

        # Create custom dataset
        dataset = TensorDataset(performance)

        # Custom collate function to handle rule_ids
        def collate_fn(batch):
            indices = [i for i, _ in enumerate(batch)]
            return {
                "rule_ids": [rule_ids[i] for i in indices],
                "performance": torch.stack([batch[i][0] for i in range(len(batch))]),
                "contexts": contexts[indices] if contexts is not None else None
            }

        return DataLoader(dataset, batch_size=self.config.batch_size,
                         shuffle=shuffle, collate_fn=collate_fn)

    def get_rule_parameters(self, rule_id: str) -> Optional[Dict[str, float]]:
        """Get current parameters for a rule"""
        if rule_id in self.differentiable_rules:
            return self.differentiable_rules[rule_id].get_parameter_dict()
        return None

    def update_drm_rule(self, drm_rule, rule_id: str):
        """Update DRM rule with learned parameters"""
        if rule_id in self.differentiable_rules:
            learned_params = self.get_rule_parameters(rule_id)
            for param_name, value in learned_params.items():
                if param_name in drm_rule.params:
                    drm_rule.update_param(param_name, value)

    def save_model(self, path: str):
        """Save trained model"""
        state_dict = {
            "differentiable_rules": {
                rule_id: rule.state_dict()
                for rule_id, rule in self.differentiable_rules.items()
            },
            "training_history": dict(self.training_history),
            "config": self.config.__dict__
        }

        if self.rule_generator:
            state_dict["rule_generator"] = self.rule_generator.state_dict()

        torch.save(state_dict, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load trained model"""
        state_dict = torch.load(path, map_location=self.device)

        # Load rule states
        for rule_id, rule_state in state_dict["differentiable_rules"].items():
            if rule_id in self.differentiable_rules:
                self.differentiable_rules[rule_id].load_state_dict(rule_state)

        # Load generator if exists
        if "rule_generator" in state_dict and self.rule_generator:
            self.rule_generator.load_state_dict(state_dict["rule_generator"])

        # Load training history
        self.training_history = defaultdict(list, state_dict.get("training_history", {}))

        logger.info(f"Model loaded from {path}")

# Utility functions for DRM integration
def convert_drm_rule_to_torch(drm_rule) -> Optional[DifferentiableRule]:
    """Convert DRM Rule to PyTorch differentiable rule"""
    if drm_rule.type != "HYBRID":
        return None  # Only HYBRID rules are trainable

    param_specs = {}
    for name, param in drm_rule.params.items():
        if param.requires_grad:
            param_specs[name] = {
                "value": param.value,
                "requires_grad": True,
                "min_val": param.min_val,
                "max_val": param.max_val,
                "constraint_fn": param.constraint_fn
            }

    if not param_specs:
        return None

    return DifferentiableRule(drm_rule.id, param_specs)

def create_training_data_from_replay(replay_buffer) -> List[Dict[str, Any]]:
    """Convert replay buffer to PyTorch training data"""
    training_data = []

    for rule_id, reward, context in replay_buffer.buf:
        training_data.append({
            "rule_id": rule_id,
            "performance": float(reward),
            "context": context or {}
        })

    return training_data
