import torch
from torch import nn

def compute_class_weights_as_list(sample_counts):
    """
    Compute class weights based on sample counts for each class.
    
    Args:
        sample_counts (list or array): List of sample counts for each class.
    
    Returns:
        list: List of class weights corresponding to each class.
    """
    total_samples = sum(sample_counts)
    num_classes = len(sample_counts)
    
    class_weights = torch.tensor([
        total_samples / (num_classes * count)
        for count in sample_counts
    ]).to('cuda')
    return class_weights

class CustomLoss(nn.Module):
    def __init__(self, whichloss='softmax', class_count=None):
        super(CustomLoss, self).__init__()
        self.class_count = torch.tensor(class_count).to('cuda')
        self.class_weights = compute_class_weights_as_list(self.class_count)
        self.whichloss = whichloss
        self.num_classes = len(class_count)

        # for focalloss
        self.gamma = 2.0
        self.alpha = None
        self.reduction = 'mean'

        # for classbalancedloss
        self.beta = 0.99

        # for equalizationloss
        self.suppression_factor = 1.5

        # for ldam loss
        #self.cls_num_list = torch.tensor(cls_num_list, dtype=torch.float)
        self.max_margin = 0.5
        #self.weight = weight
        #self.reduction = reduction
        self.margins = self.max_margin / torch.sqrt(self.class_count)
        self.margins = self.margins.to(torch.float)
        

    def forward(self, logits, labels):
        if self.whichloss == 'softmax':
            loss = nn.functional.cross_entropy(logits, labels, weight=None)
        elif self.whichloss == 'wsoftmax':
            loss = nn.functional.cross_entropy(logits, labels, weight=self.class_weights)
        elif self.whichloss == 'focalloss':
            # Compute softmax probabilities
            probs = F.softmax(logits, dim=1)  # Shape: (batch_size, num_classes)
            # Select the probabilities corresponding to the true class
            # Shape: (batch_size,)
            true_probs = probs[torch.arange(labels.size(0)), labels]
            # Compute the focal loss
            focal_weight = (1 - true_probs) ** self.gamma
            log_probs = torch.log(true_probs + 1e-9)  # Add epsilon for numerical stability
            loss = -focal_weight * log_probs
            # Apply alpha weighting if provided
            if self.alpha is not None:
                if isinstance(self.alpha, torch.Tensor):  # Class-specific alpha
                    alpha_t = self.alpha[labels]
                else:  # Scalar alpha
                    alpha_t = self.alpha
                loss *= alpha_t
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
            else:  # 'none'
                return loss
        elif self.whichloss == 'classbalancedloss':
            effective_num = 1.0 - torch.pow(self.beta, self.class_count)
            weights = (1.0 - self.beta) / (effective_num + 1e-8)
            weights = weights / weights.sum()  # Normalize weights
    
            # Convert targets to one-hot encoding
            one_hot_targets = F.one_hot(labels, num_classes=self.num_classes).float()
    
            # Apply softmax to logits
            probs = F.softmax(logits, dim=1)
    
            # Compute class-balanced cross-entropy loss
            weighted_loss = -weights * one_hot_targets * torch.log(probs + 1e-8)
            loss = weighted_loss.sum(dim=1)
    
            # Apply reduction
            if self.reduction == 'mean':
                loss = loss.mean()
            elif self.reduction == 'sum':
                loss = loss.sum()

        elif self.whichloss == 'balancedsoftmax':
            log_class_counts = torch.log(self.class_count.float() + 1e-8)  # Avoid log(0)
            # Adjust logits by subtracting log class counts
            adjusted_logits = logits - log_class_counts
            # Compute the balanced softmax probabilities
            balanced_probs = F.log_softmax(adjusted_logits, dim=1)
            # Gather the log probabilities of the true classes
            log_probs = balanced_probs[torch.arange(logits.size(0)), labels]
            # Compute the loss
            loss = -log_probs
            # Apply reduction
            if self.reduction == 'mean':
                loss = loss.mean()
            elif self.reduction == 'sum':
                loss = loss.sum()
        elif self.whichloss == 'equalizationloss':
            one_hot_targets = F.one_hot(labels, num_classes=self.num_classes).float()
    
            # Compute probabilities with softmax
            probs = F.softmax(logits, dim=1)
    
            # Suppression weights for negative samples
            effective_num = torch.pow(self.class_count.float(), self.suppression_factor)
            weights = (1.0 / effective_num).to(logits.device)
    
            # Broadcast weights to match batch size and one-hot targets
            weight_matrix = one_hot_targets + (1 - one_hot_targets) * weights.unsqueeze(0)
    
            # Compute weighted cross-entropy loss
            ce_loss = -one_hot_targets * torch.log(probs + 1e-8)
            suppressed_loss = ce_loss * weight_matrix
    
            # Sum over classes and apply reduction
            loss = suppressed_loss.sum(dim=1)
            if self.reduction == 'mean':
                loss = loss.mean()
            elif self.reduction == 'sum':
                loss = loss.sum()
        elif self.whichloss == 'ldamloss':
            batch_size, num_classes = logits.size()
            
            # Create a margin matrix
            margins = torch.zeros_like(logits)
            margins[torch.arange(batch_size), labels] = self.margins[labels]
            
            # Adjust logits with the margin
            adjusted_logits = logits - margins
            
            # Compute cross-entropy loss
            loss = F.cross_entropy(adjusted_logits, labels, weight=self.class_weights, reduction=self.reduction)
        else:
            print("NOT IMPLEMENTED ERROR")
        return loss

# Instantiate your loss
