import numpy as np
from matplotlib import pyplot as plt

def triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings, 
                 margin=0.2, distance='euclidean'):
    """
    Compute triplet loss for metric learning.
    
    Args:
        anchor_embeddings: NumPy array of shape (N, d), anchor embeddings
        positive_embeddings: NumPy array of shape (N, d), positive embeddings
        negative_embeddings: NumPy array of shape (N, d), negative embeddings
        margin: Float, margin parameter alpha > 0 (default 0.2)
        distance: String, distance metric: 'euclidean' or 'squared_euclidean'
    
    Returns:
        loss: Scalar loss value (float32)
        active_triplets: Boolean array of shape (N,) indicating active triplets
    """
    # Validate inputs
    assert anchor_embeddings.shape == positive_embeddings.shape == negative_embeddings.shape
    assert margin > 0, "Margin must be positive"
    assert distance in ['euclidean', 'squared_euclidean'], "Invalid distance metric"
    
    # Compute distances based on metric
    if distance == 'euclidean':
        # Euclidean distance: sqrt(sum((x - y)^2))
        dap = np.sqrt(np.sum((anchor_embeddings - positive_embeddings) ** 2, axis=1))
        dan = np.sqrt(np.sum((anchor_embeddings - negative_embeddings) ** 2, axis=1))
    elif distance == 'squared_euclidean':
        # Squared Euclidean distance: sum((x - y)^2)
        dap = np.sum((anchor_embeddings - positive_embeddings) ** 2, axis=1)
        dan = np.sum((anchor_embeddings - negative_embeddings) ** 2, axis=1)
    
    # Compute triplet loss for each triplet: max(0, d(a,p) - d(a,n) + margin)
    triplet_losses = np.maximum(0.0, dap - dan + margin)
    
    # Identify active triplets (those that violate the margin)
    active_triplets = (dap - dan + margin) > 0
    
    # Compute average loss over batch
    loss = np.max(triplet_losses).astype(np.float32)
    
    return loss, active_triplets

# Batch of 3 triplets, embedding dimension 64
anchor = np.random.randn(3, 64).astype(np.float32)
positive = np.random.randn(3, 64).astype(np.float32)
negative = np.random.randn(3, 64).astype(np.float32)

# dibujar anchor
plt.imshow(anchor, aspect='auto')
plt.title('Anchor Embeddings')
plt.colorbar()
plt.show()

# dibujar positive
plt.imshow(positive, aspect='auto')
plt.title('Positive Embeddings')
plt.colorbar()
plt.show()

# dibujar negative
plt.imshow(negative, aspect='auto')
plt.title('Negative Embeddings')
plt.colorbar()
plt.show()

print(f"Anchor {anchor} shape: {anchor.shape}")
print(f"Positive {positive} shape: {positive.shape}")
print(f"Negative {negative} shape: {negative.shape}")

loss, active = triplet_loss(anchor, positive, negative, margin=0.2, distance='euclidean')
print("Triplet Loss:", loss)
print("Active Triplets:", active)