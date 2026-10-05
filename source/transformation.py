import numpy as np
import torchvision
import torchvision.transforms as transforms

def flatten_bchw(tensor_4d):
    B, C, H, W = tensor_4d.shape
    CHW = C * H * W
    
    tensor_2d = np.zeros((B, CHW), dtype=tensor_4d.dtype)
    
    for b in range(B):
        for c in range(C):
            for h in range(H):
                for w in range(W):
            
                    k = c * (H * W) + h * W + w
                    tensor_2d[b, k] = tensor_4d[b, c, h, w]
                    
    return tensor_2d

def reconstruct_bchw(tensor_2d, B, C, H, W):
    CHW = C * H * W
    tensor_4d = np.zeros((B, C, H, W), dtype=tensor_2d.dtype)
    
    stride_c = H * W
    stride_h = W
    
    for b in range(B):
        for k in range(CHW):
            c = k // stride_c
            rem = k % stride_c
            h = rem // stride_h
            w = rem % stride_h
            
            tensor_4d[b, c, h, w] = tensor_2d[b, k]
            
    return tensor_4d

def compute_errors(original, reconstructed):
    diff = np.abs(original - reconstructed)
    e_max = np.max(diff)
    e_mae = np.mean(diff)
    return e_max, e_mae

def get_mnist_sample():
    transform = transforms.ToTensor()
    dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    img, _ = dataset[0] 
    return img.unsqueeze(0).numpy()

def get_cifar_sample():
    transform = transforms.ToTensor()
    dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    img, _ = dataset[0] 
    return img.unsqueeze(0).numpy() 

def run_experiment(name, tensor):
    B, C, H, W = tensor.shape

    flat_tensor = flatten_bchw(tensor)
    reconstructed_tensor = reconstruct_bchw(flat_tensor, B, C, H, W)
    e_max, e_mae = compute_errors(tensor, reconstructed_tensor)
    
    print(f"| {name:<16} | {B:<2} | {C:<4} | {H:<3} | {W:<3} | {e_max:.1e} | {e_mae:.1e} |")
    return e_max, e_mae

def main():
    print("initialising datasets & running\n")
    print("| Dataset/Input    | B  | C    | H   | W   | E_max   | MAE     |")
    print("|------------------|----|------|-----|-----|---------|---------|")
    
    mnist_tensor = get_mnist_sample()
    run_experiment("MNIST / sample", mnist_tensor)

    cifar_tensor = get_cifar_sample()
    run_experiment("CIFAR / sample", cifar_tensor)

    synthetic_configs = [
        ("Synthetic fmap1", 1, 16, 32, 32),
        ("Synthetic fmap2", 1, 64, 32, 32),
        ("Synthetic fmap3", 1, 128, 32, 32),
        ("Synthetic fmap4", 1, 500, 32, 32)
    ]
    
    for name, B, C, H, W in synthetic_configs:
        synth_tensor = np.random.randn(B, C, H, W).astype(np.float32)
        run_experiment(name, synth_tensor)

if __name__ == "__main__":
    main()