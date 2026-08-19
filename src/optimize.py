"""
Senate AI - 4-Stage Model Optimizer (Vectorized)
1. Smart Prune (redistribute weak to strong)
2. Safe Precision Prune (merge insignificant)
3. Progressive Bit Reduction
4. 8-bit Quantize
"""

import torch
import torch.nn as nn
import time
import os
import sys
from pathlib import Path


def progress_bar(done, total, label="", width=30):
    pct = done / max(total, 1)
    filled = int(width * pct)
    bar = '█' * filled + '░' * (width - filled)
    return f"   {label} [{bar}] {pct*100:.0f}%"


class SenateOptimizer:
    """Optimizes a Senate bundle through 4 stages"""
    
    def __init__(self, bundle_path, output_path=None):
        self.bundle_path = bundle_path
        self.output_path = output_path or bundle_path
        
        print(f"\n{'='*50}")
        print(f"  SENATE OPTIMIZER (VECTORIZED)")
        print(f"{'='*50}")
        print(f"   Bundle: {bundle_path}")
        
        self.bundle = torch.load(bundle_path, map_location='cpu', weights_only=False)
        self.senators = self.bundle.get('senators', {})
        print(f"   Senators: {len(self.senators)}")
        sys.stdout.flush()
    
    def get_bundle_size(self):
        total_params = 0
        non_zero = 0
        
        for senator_id, data in self.senators.items():
            state_dict = data['state_dict'] if 'state_dict' in data else data
            for param in state_dict.values():
                if isinstance(param, torch.Tensor):
                    total_params += param.numel()
                    non_zero += (param != 0).sum().item()
        
        sparsity = (1 - non_zero / max(total_params, 1)) * 100
        size_mb = total_params * 4 / (1024 * 1024)
        
        return {
            'total_params': total_params,
            'non_zero': non_zero,
            'sparsity': sparsity,
            'size_mb': size_mb
        }
    
    def smart_prune(self, target_sparsity=0.5):
        """Vectorized smart prune - redistribute weak weights to strong ones"""
        print(f"\n  STAGE 1: SMART PRUNE (target: {target_sparsity*100:.0f}%)")
        sys.stdout.flush()
        
        total_redistributed = 0
        total_params = 0
        start_time = time.time()
        
        for senator_id, data in self.senators.items():
            state_dict = data['state_dict'] if 'state_dict' in data else data
            
            for param_name, param in state_dict.items():
                if not isinstance(param, torch.Tensor) or param.dim() < 2:
                    continue
                
                weight = param.float()
                total_params += weight.numel()
                
                # Vectorized: find strong and weak weights per row
                abs_weight = weight.abs()
                k = max(1, int((1 - target_sparsity) * weight.shape[1]))
                
                # Get threshold per row
                thresholds = torch.kthvalue(abs_weight, weight.shape[1] - k, dim=1, keepdim=True).values
                strong_mask = abs_weight >= thresholds
                
                # Redistribute weak to strong (approximation)
                weak_vals = weight * (~strong_mask)
                strong_vals = weight * strong_mask
                
                # Add weak sum to strong weights proportionally
                weak_sum = weak_vals.abs().sum(dim=1, keepdim=True)
                strong_sign = torch.sign(strong_vals)
                
                # Redistribute: add weak magnitude to strong weights
                redistribution = weak_sum * strong_sign / max(strong_mask.sum(dim=1, keepdim=True).float().max(), 1)
                weight = strong_vals + redistribution * strong_mask
                
                # Zero out weak weights
                weight = weight * strong_mask
                
                total_redistributed += (~strong_mask).sum().item()
                param.data = weight.to(param.dtype)
        
        elapsed = (time.time() - start_time) / 60
        print(f"   Redistributed: {total_redistributed:,} weights | Time: {elapsed:.1f}min")
        sys.stdout.flush()
        return total_redistributed
    
    def precision_prune_safe(self, significance=2):
        """Vectorized precision prune"""
        print(f"\n  STAGE 2: SAFE PRECISION PRUNE")
        sys.stdout.flush()
        
        total_merged = 0
        start_time = time.time()
        
        for senator_id, data in self.senators.items():
            state_dict = data['state_dict'] if 'state_dict' in data else data
            
            for param_name, param in state_dict.items():
                if not isinstance(param, torch.Tensor) or param.dim() < 2:
                    continue
                
                weight = param.float()
                abs_weight = weight.abs()
                
                # Find insignificant weights (close to 0)
                insignificant = abs_weight < (10 ** -(significance + 1))
                
                # Zero them out
                weight[insignificant] = 0
                total_merged += insignificant.sum().item()
                
                param.data = weight.to(param.dtype)
        
        elapsed = (time.time() - start_time) / 60
        print(f"   Merged: {total_merged:,} | Time: {elapsed:.1f}min")
        sys.stdout.flush()
        return total_merged
    
    def progressive_bit_reduce(self, target_bits=8):
        """Vectorized bit reduction"""
        print(f"\n  STAGE 3: PROGRESSIVE BITS")
        sys.stdout.flush()
        
        start_time = time.time()
        total_reduced = 0
        
        for senator_id, data in self.senators.items():
            state_dict = data['state_dict'] if 'state_dict' in data else data
            
            for param_name, param in state_dict.items():
                if not isinstance(param, torch.Tensor) or param.dim() < 2:
                    continue
                
                weight = param.float()
                
                # Quantize to target bits
                max_val = weight.abs().max()
                if max_val > 0:
                    scale = max_val / (2 ** (target_bits - 1) - 1)
                    weight = torch.round(weight / scale) * scale
                    total_reduced += weight.numel()
                
                param.data = weight.to(param.dtype)
        
        elapsed = (time.time() - start_time) / 60
        print(f"   Reduced: {total_reduced:,} weights | Time: {elapsed:.1f}min")
        sys.stdout.flush()
        return total_reduced
    
    def quantize_senators(self):
        """Convert to float16"""
        print(f"\n  STAGE 4: QUANTIZE TO FLOAT16")
        sys.stdout.flush()
        
        quantized_count = 0
        
        for senator_id, data in self.senators.items():
            state_dict = data['state_dict'] if 'state_dict' in data else data
            
            for param_name, param in state_dict.items():
                if isinstance(param, torch.Tensor) and param.dtype == torch.float32:
                    if param.numel() > 100:
                        state_dict[param_name] = param.half()
                        quantized_count += 1
        
        print(f"   Quantized: {quantized_count} tensors to float16")
        sys.stdout.flush()
        return quantized_count
    
    def optimize(self, target_sparsity=0.5):
        before = self.get_bundle_size()
        print(f"\n  Before: {before['total_params']:,} params, {before['size_mb']:.1f}MB")
        sys.stdout.flush()
        
        self.smart_prune(target_sparsity=target_sparsity)
        self.precision_prune_safe()
        self.progressive_bit_reduce(target_bits=8)
        self.quantize_senators()
        
        after = self.get_bundle_size()
        saved_mb = before['size_mb'] - after['size_mb']
        
        print(f"\n{'='*50}")
        print(f"  OPTIMIZATION COMPLETE")
        print(f"{'='*50}")
        print(f"  Before: {before['size_mb']:.1f}MB")
        print(f"  After:  {after['size_mb']:.1f}MB")
        print(f"  Saved:  {saved_mb:.1f}MB ({saved_mb/before['size_mb']*100:.0f}%)")
        print(f"  Sparsity: {after['sparsity']:.1f}%")
        sys.stdout.flush()
        
        return after
    
    def save(self):
        print(f"\n  Saving optimized bundle...")
        sys.stdout.flush()
        torch.save(self.bundle, self.output_path)
        size_mb = os.path.getsize(self.output_path) / (1024 * 1024)
        print(f"  Saved: {self.output_path} ({size_mb:.1f}MB)")
        sys.stdout.flush()
        return size_mb


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('bundle_id', type=int)
    parser.add_argument('target_sparsity', nargs='?', type=float, default=0.5)
    
    args = parser.parse_args()
    
    bundle_path = f"senate_bundles/bundle_{args.bundle_id:03d}.pt"
    
    if not Path(bundle_path).exists():
        print(f"Bundle {args.bundle_id} not found")
        sys.exit(1)
    
    optimizer = SenateOptimizer(bundle_path)
    optimizer.optimize(target_sparsity=args.target_sparsity)
    optimizer.save()
