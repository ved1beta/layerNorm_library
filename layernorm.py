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
    row = tl.program_id(0)
    X = row * stride

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(input_ptr + X + cols, mask=cols < N, other=0.).to(tl.float32)
        acc += a
    mean = tl.sum(acc, axis=0) / N
    tl.store(output_ptr + row, mean)

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
    row = tl.program_id(0)
    inputs_off = row * stride 
    mean = tl.load(mean_ptr + row)

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(input_ptr + inputs_off + cols, mask=mask, other=0.0)

        x_centered = x - mean
        acc += x_centered * x_centered

    var = tl.sum(acc, axis=0) / N + eps
    tl.store(output_ptr + row, var)

@triton.jit 
def normalize_kernel(
    input_ptr, 
    output_ptr, 
    mean_ptr, 
    var_ptr, 
    weight_ptr, 
    bias_ptr, 
    stride, 
    N , 
    BLOCK_SIZE: tl.constexpr, 
):
    row = tl.program_id(0)
    input_off = row * stride
    
    # Load mean and variance for this row
    mean = tl.load(mean_ptr + row)
    var = tl.load(var_ptr + row)
    std = tl.sqrt(var)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        
        # Load input values
        x = tl.load(input_ptr + input_off + cols, mask=mask, other=0.0)
        
        # Load weight and bias
        w = tl.load(weight_ptr + cols, mask=mask, other=0.0)
        b = tl.load(bias_ptr + cols, mask=mask, other=0.0)
        
        # Normalize
        y = w * ((x - mean) / std) + b
        
        # Store result
        tl.store(output_ptr + input_off + cols, y, mask=mask)





class TritonLayerNorm:
    def __init__(self, eps=1e-5):
        self.eps = eps
    
    @staticmethod
    def mean(x: torch.Tensor) -> torch.Tensor:
        batch_size, n = x.shape
        output = torch.empty(batch_size, device=x.device, dtype=x.dtype)
        

        BLOCK_SIZE = triton.next_power_of_2(min(n, 1024))
        

        grid = (batch_size,)
        mean_kernel[grid](
            x.ptr,
            output.ptr,
            x.stride(0),
            n,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return output
    
    def var(self, x: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
        batch_size, n = x.shape
        output = torch.empty(batch_size, device=x.device, dtype=x.dtype)
        
        BLOCK_SIZE = triton.next_power_of_2(min(n, 1024))
        
        grid = (batch_size,)
        var_kernel[grid](
            x.ptr,
            output.ptr,
            mean.ptr,
            x.stride(0),
            n,
            self.eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return output
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.mean(x)
        var = self.var(x, mean)
        return mean, var



    def test():
        x= torch(32, 512, device = 'cude')

        ln = TritonLayerNorm()
        triton_mean, triton_var = ln.forward(x)
        
        torch_mean = x.mean(dim=1)
        torch_var = x.var(dim=1, unbiased=False) + 1e-5
        
        mean_diff = torch.abs(triton_mean - torch_mean).max().item()
        var_diff = torch.abs(triton_var - torch_var).max().item()
        
        print(f"Mean difference: {mean_diff}")
        print(f"Variance difference: {var_diff}")

    test()