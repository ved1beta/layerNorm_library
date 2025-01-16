import triton 
import torch
import triton.language as tl 

@triton.jit 
def mean_kernel(
    input_ptr, 
    output_ptr,
    stride, 
    N, 
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row
    pid = tl.program_id(0)
    
    # Initialize accumulator
    acc = 0.0
    
    # Get the row offset
    input_ptr = tl.advance(input_ptr, pid * stride)
    
    # Process the row in blocks
    for off in range(0, N, BLOCK_SIZE):
        # Load a block of elements
        nums = tl.load(tl.advance(input_ptr, off), mask=tl.arange(0, BLOCK_SIZE) < (N - off))
        acc += tl.sum(nums)
    
    mean = acc / N
    tl.store(output_ptr + pid, mean)

@triton.jit
def var_kernel(
    input_ptr, 
    output_ptr, 
    mean_ptr, 
    stride, 
    N, 
    eps,  
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Load mean value
    mean = tl.load(mean_ptr + pid)
    
    # Initialize variance accumulator
    acc = 0.0
    
    # Get the row offset
    input_ptr = tl.advance(input_ptr, pid * stride)
    
    # Process the row in blocks
    for off in range(0, N, BLOCK_SIZE):
        # Load a block of elements
        nums = tl.load(tl.advance(input_ptr, off), mask=tl.arange(0, BLOCK_SIZE) < (N - off))
        nums = nums - mean
        acc += tl.sum(nums * nums)
    
    var = acc / N + eps
    tl.store(output_ptr + pid, var)

@triton.jit 
def normalize_kernel(
    input_ptr, 
    output_ptr, 
    mean_ptr, 
    var_ptr, 
    weight_ptr, 
    bias_ptr, 
    stride, 
    N, 
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Load statistics
    mean = tl.load(mean_ptr + pid)
    var = tl.load(var_ptr + pid)
    std = tl.sqrt(var)
    
    # Get row offsets
    input_ptr = tl.advance(input_ptr, pid * stride)
    output_ptr = tl.advance(output_ptr, pid * stride)
    
    # Process the row in blocks
    for off in range(0, N, BLOCK_SIZE):
        mask = tl.arange(0, BLOCK_SIZE) < (N - off)
        
        # Load values
        x = tl.load(tl.advance(input_ptr, off), mask=mask)
        w = tl.load(tl.advance(weight_ptr, off), mask=mask)
        b = tl.load(tl.advance(bias_ptr, off), mask=mask)
        
        # Normalize
        y = w * ((x - mean) / std) + b
        
        # Store result
        tl.store(tl.advance(output_ptr, off), y, mask=mask)

class TritonLayerNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.bias = torch.nn.Parameter(torch.zeros(dim))

    def mean(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n = x.shape
        output = torch.empty(batch_size, device=x.device, dtype=torch.float32)
        
        BLOCK_SIZE = triton.next_power_of_2(min(n, 1024))
        grid = (batch_size,)
        
        mean_kernel[grid](
            x.data_ptr(),
            output.data_ptr(),
            x.stride(0),
            n,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return output
    
    def var(self, x: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
        batch_size, n = x.shape
        output = torch.empty(batch_size, device=x.device, dtype=torch.float32)
        
        BLOCK_SIZE = triton.next_power_of_2(min(n, 1024))
        grid = (batch_size,)
        
        var_kernel[grid](
            x.data_ptr(),
            output.data_ptr(),
            mean.data_ptr(),
            x.stride(0),
            n,
            self.eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return output
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n = x.shape
        assert n == self.dim, f"Input dimension {n} doesn't match layer dimension {self.dim}"
        
        mean = self.mean(x)
        var = self.var(x, mean)
        
        output = torch.empty_like(x)
        
        BLOCK_SIZE = triton.next_power_of_2(min(n, 1024))
        grid = (batch_size,)
        
        normalize_kernel[grid](
            x.data_ptr(),
            output.data_ptr(),
            mean.data_ptr(),
            var.data_ptr(),
            self.weight.data_ptr(),
            self.bias.data_ptr(),
            x.stride(0),
            n,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        return output

def test_layer_norm():
    batch_sizes = [1, 32, 128]
    dims = [512, 1024, 2048]
    
    for batch_size in batch_sizes:
        for dim in dims:
            print(f"\nTesting batch_size={batch_size}, dim={dim}")
            
            x = torch.randn(batch_size, dim, device='cuda', dtype=torch.float32)
            
            triton_ln = TritonLayerNorm(dim).cuda()
            triton_out = triton_ln(x)
            
            torch_ln = torch.nn.LayerNorm(dim).cuda()
            torch_ln.weight.data.copy_(triton_ln.weight)
            torch_ln.bias.data.copy_(triton_ln.bias)
            torch_out = torch_ln(x)
            
            diff = torch.abs(triton_out - torch_out).max().item()
            print(f"Max difference: {diff}")
            assert diff < 1e-5, f"Large difference detected: {diff}"

if __name__ == "__main__":
    test_layer_norm()