import torch
import torch.nn as nn
import math

class NavieAttention(nn.Module):
    """
    Naive implementation of Flash Attention for single head attention
    """
    def __init__(self, dropout=0.0, scale=None):
        super(NavieAttention, self).__init__()
        self.dropout = dropout
        self.scale = scale
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask=None) -> torch.Tensor:
        """
        Performs naive flash attention computation.

        Args:
            Q (torch.Tensor): Query matrix of shape (batch_size, seq_len, d_k)
            K (torch.Tensor): Key matrix of shape (batch_size, seq_len, d_k)
            V (torch.Tensor): Value matrix of shape (batch_size, seq_len, d_v)
            mask (torch.Tensor, optional): Attention mask of shape (batch_size, seq_len, seq_len)

        Returns:
            torch.Tensor: Attention output of shape (batch_size, seq_len, d_v)
        """
        batch_size, seq_len, d_k = Q.shape
        d_k = Q.size(-1)
        
        # Scale factor
        scale = self.scale if self.scale is not None else 1.0 / math.sqrt(d_k)
        
        # Compute attention scores: (batch_size, seq_len, seq_len)
        # Q @ K^T
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Apply dropout
        if self.dropout > 0 and self.training:
            attention_weights = torch.dropout(attention_weights, self.dropout, True)
        
        # Compute output: (batch_size, seq_len, d_v)
        output = torch.matmul(attention_weights, V)
        
        return output

# Test parameters
batch_size = 30
seq_len = 1024
d_model = 512
d_k = 64
d_v = 64

def get_inputs():
    """
    Generate random inputs for testing
    """
    Q = torch.randn(batch_size, seq_len, d_k)
    K = torch.randn(batch_size, seq_len, d_k)
    V = torch.randn(batch_size, seq_len, d_v)
    
    # Optional mask
    mask = torch.ones(batch_size, seq_len, seq_len)
    # Make it causal mask (upper triangular)
    mask = torch.tril(mask)
    
    return [Q, K, V, mask]

def get_init_inputs():
    return []  # No special initialization inputs needed

# Test the implementation
if __name__ == "__main__":
    model = NavieAttention(dropout=0.1)
    model.eval()  # Set to eval mode for testing
    
    Q, K, V, mask = get_inputs()
    
    with torch.no_grad():
        output = model(Q, K, V, mask)
        print(f"Input shapes:")
        print(f"  Q: {Q.shape}")
        print(f"  K: {K.shape}")
        print(f"  V: {V.shape}")
        print(f"  mask: {mask.shape}")
        print(f"Output shape: {output.shape}")
        print(f"\nNaive Attention computation completed!")
        
        # Verify against PyTorch's standard attention for correctness
        scale = 1.0 / math.sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        scores = scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = torch.softmax(scores, dim=-1)
        expected_output = torch.matmul(attention_weights, V)
        
        # Check if outputs match
        if torch.allclose(output, expected_output, rtol=1e-5, atol=1e-5):
            print("✓ Implementation matches expected result!")
        else:
            print("✗ Implementation differs from expected result!")