"""
OPTIMIZED Semantic Loss Functions for Can3Tok - ScanNet72 Edition
Removed all debugging for maximum speed
"""

import torch
import torch.nn.functional as F
import numpy as np


class ScanNet72SemanticLoss(torch.nn.Module):
    """
    Optimized contrastive loss between Gaussian features and category prototypes.
    Removed all debugging for maximum training speed.
    """
    
    def __init__(self, num_categories=72, temperature=0.07, segment_weight=10.0, 
                 instance_weight=0.0, feature_dim=32):
        super().__init__()
        self.num_categories = num_categories  # 72 for ScanNet72
        self.temperature = temperature
        self.segment_weight = segment_weight
        self.instance_weight = instance_weight
        self.feature_dim = feature_dim
        
        # Missing categories in ScanNet72 (from analysis)
        self.missing_categories = [13, 53, 61]
        
    def forward(self, embeddings, segment_labels, instance_labels=None, batch_size=1):
        """
        Optimized contrastive loss - NO DEBUGGING.
        
        Args:
            embeddings: [B, N, D] per-Gaussian features
            segment_labels: [B, N] ScanNet72 segment labels (0-71)
            instance_labels: [B, N] instance labels (optional)
            batch_size: B
            
        Returns:
            total_loss: scalar tensor
            metrics: dict of loss components
        """
        B, N, D = embeddings.shape
        
        segment_loss = torch.tensor(0.0, device=embeddings.device)
        instance_loss = torch.tensor(0.0, device=embeddings.device)
        
        segment_count = 0
        instance_count = 0
        
        # Segment-level contrastive loss (OPTIMIZED)
        if self.segment_weight > 0:
            for b in range(B):
                # Get valid labels (non-negative)
                valid_mask = segment_labels[b] >= 0
                if valid_mask.sum() == 0:
                    continue
                    
                batch_embeddings = embeddings[b][valid_mask]  # [M, D]
                batch_labels = segment_labels[b][valid_mask]  # [M]
                
                # Normalize embeddings
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=-1)
                
                # Get unique categories
                unique_categories = torch.unique(batch_labels)
                unique_categories = unique_categories.cpu().numpy()
                
                # Remove missing categories
                unique_categories = [cat for cat in unique_categories 
                                   if cat not in self.missing_categories]
                
                if len(unique_categories) < 2:
                    continue
                
                # Create prototypes
                prototypes = []
                prototype_category_ids = []
                
                for cat_id in unique_categories:
                    cat_mask = batch_labels == cat_id
                    if cat_mask.sum() > 0:
                        cat_feat = batch_embeddings[cat_mask].mean(dim=0, keepdim=True)
                        cat_feat = F.normalize(cat_feat, p=2, dim=-1)
                        prototypes.append(cat_feat)
                        prototype_category_ids.append(cat_id)
                
                if len(prototypes) < 2:
                    continue
                
                # Stack prototypes [K, D]
                prototypes = torch.cat(prototypes, dim=0)
                prototypes = F.normalize(prototypes, p=2, dim=-1)
                
                # Compute similarity matrix
                similarity_matrix = torch.matmul(batch_embeddings, prototypes.T) / self.temperature
                
                # Create mapping from category ID to prototype index
                cat_to_proto_idx = {cat_id: idx for idx, cat_id in enumerate(prototype_category_ids)}
                
                # Create target indices for each Gaussian (vectorized)
                target_indices = torch.zeros_like(batch_labels, dtype=torch.long)
                for i, label in enumerate(batch_labels):
                    label_int = label.item()
                    if label_int in cat_to_proto_idx:
                        target_indices[i] = cat_to_proto_idx[label_int]
                    else:
                        target_indices[i] = -100  # Ignore index
                
                # Compute loss
                loss = F.cross_entropy(similarity_matrix, target_indices, ignore_index=-100)
                
                # Skip if no valid targets
                if not torch.isnan(loss) and not torch.isinf(loss):
                    segment_loss += loss
                    segment_count += 1
            
            if segment_count > 0:
                segment_loss = segment_loss / segment_count
        
        # Instance-level contrastive loss (OPTIMIZED)
        if self.instance_weight > 0 and instance_labels is not None:
            for b in range(B):
                valid_mask = instance_labels[b] >= 0
                if valid_mask.sum() == 0:
                    continue
                    
                batch_embeddings = embeddings[b][valid_mask]
                batch_labels = instance_labels[b][valid_mask]
                
                # Normalize embeddings
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=-1)
                
                # Get unique instances
                unique_instances = torch.unique(batch_labels)
                
                if len(unique_instances) < 2:
                    continue
                
                # Aggregate features by instance
                instance_features = []
                instance_ids = []
                for inst_id in unique_instances:
                    inst_mask = batch_labels == inst_id
                    if inst_mask.sum() > 0:
                        inst_feat = batch_embeddings[inst_mask].mean(dim=0, keepdim=True)
                        instance_features.append(inst_feat)
                        instance_ids.append(inst_id.item())
                
                if len(instance_features) < 2:
                    continue
                    
                # Stack instance prototypes
                prototypes = torch.cat(instance_features, dim=0)
                prototypes = F.normalize(prototypes, p=2, dim=-1)
                
                # Compute similarity matrix
                similarity_matrix = torch.matmul(batch_embeddings, prototypes.T) / self.temperature
                
                # Create target indices
                inst_to_proto_idx = {inst_id: idx for idx, inst_id in enumerate(instance_ids)}
                target_indices = torch.zeros_like(batch_labels, dtype=torch.long)
                for i, label in enumerate(batch_labels):
                    label_int = label.item()
                    if label_int in inst_to_proto_idx:
                        target_indices[i] = inst_to_proto_idx[label_int]
                    else:
                        target_indices[i] = -100
                
                # Compute loss
                loss = F.cross_entropy(similarity_matrix, target_indices, ignore_index=-100)
                
                if not torch.isnan(loss) and not torch.isinf(loss):
                    instance_loss += loss
                    instance_count += 1
            
            if instance_count > 0:
                instance_loss = instance_loss / instance_count
        
        # Total loss
        total_loss = self.segment_weight * segment_loss + self.instance_weight * instance_loss
        
        # Simple metrics only
        metrics = {
            'segment_loss': segment_loss.item() if segment_count > 0 else 0.0,
            'instance_loss': instance_loss.item() if instance_count > 0 else 0.0,
            'semantic_loss': total_loss.item(),
            'segment_count': segment_count,
            'instance_count': instance_count,
        }
        
        return total_loss, metrics


def compute_scannet72_semantic_loss(embeddings, segment_labels, instance_labels, batch_size,
                                   segment_weight=10.0, instance_weight=0.0,
                                   temperature=0.07, subsample=2000):
    """
    Compute semantic contrastive loss optimized for ScanNet72.
    """
    B, N, D = embeddings.shape
    
    # Subsample if needed
    if subsample < N:
        indices = torch.randperm(N, device=embeddings.device)[:subsample]
        embeddings = embeddings[:, indices, :]
        segment_labels = segment_labels[:, indices]
        if instance_labels is not None:
            instance_labels = instance_labels[:, indices]
    
    # Create loss module (cached for efficiency)
    loss_module = ScanNet72SemanticLoss(
        num_categories=72,
        temperature=temperature,
        segment_weight=segment_weight,
        instance_weight=instance_weight,
        feature_dim=D
    )
    
    # Compute loss
    total_loss, metrics = loss_module(
        embeddings, segment_labels, instance_labels, batch_size
    )
    
    return total_loss, metrics


def compute_semantic_loss(embeddings, segment_labels, instance_labels, batch_size,
                         segment_weight=10.0, instance_weight=0.0,
                         temperature=0.07, subsample=2000,
                         num_categories=72):
    """
    Universal semantic loss function (backward compatible).
    """
    return compute_scannet72_semantic_loss(
        embeddings=embeddings,
        segment_labels=segment_labels,
        instance_labels=instance_labels,
        batch_size=batch_size,
        segment_weight=segment_weight,
        instance_weight=instance_weight,
        temperature=temperature,
        subsample=subsample
    )