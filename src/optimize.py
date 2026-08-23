"""
Senate AI - Smart Prune (Original Careful Version)
Redistributes weak weights to strong ones per-row.
Slow but preserves knowledge.
"""

import torch
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
    """Optimizes a Senate bundle through careful smart pruning"""
    
    def __init__(self, bundle_path, output_path=None):
        self.bundle_path = bundle_path
        self.output_path = output_path or bundle_path
        
        print(f"\n{'='*50}")
        print(f"  SENATE OPTIMIZER (CAREFUL)")
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
        size_mb = total_params * 2 / (1024 * 1024)  # float16
        
        return {
            'total_params': total_params,
            'non_zero': non_zero,
            'sparsity': sparsity,
            'size_mb': size_mb
        }
    
    def smart_prune(self, target_sparsity=0.5, max_time_minutes=30):
        """Original careful smart prune - redistribute weak to strong per row"""
        print(f"\n  STAGE 1: SMART PRUNE (target: {target_sparsity*100:.0f}%)")
        sys.stdout.flush()
        
        total_redistributed = 0
        start_time = time.time()
        timeout = max_time_minutes * 60
        
        for senator_idx, (senator_id, data) in enumerate(self.senators.items()):
            if time.time() - start_time > timeout:
                print(f"\n  Timeout after {max_time_minutes}min")
                break
            
            state_dict = data['state_dict'] if 'state_dict' in data else data
            
            for param_name, param in state_dict.items():
                if not isinstance(param, torch.Tensor) or param.dim() < 2:
                    continue
                
                weight = param.float()
                
                for row_idx in range(weight.shape[0]):
                    if time.time() - start_time > timeout:
                        break
                    
                    row = weight[row_idx]
                    abs_row = row.abs()
                    
                    if abs_row.sum() == 0:
                        continue
                    
                    k = max(1, int((1 - target_sparsity) * len(row)))
                    if k >= len(row):
                        continue
                    
                    threshold = torch.kthvalue(abs_row, len(row) - k).values
                    strong_mask = abs_row >= threshold
                    strong_idx = torch.where(strong_mask)[0]
                    weak_idx = torch.where(~strong_mask)[0]
                    
                    if len(strong_idx) == 0 or len(weak_idx) == 0:
                        continue
                    
                    for wi in weak_idx:
                        weak_val = row[wi]
                        if abs(weak_val) < 0.00001:
                            row[wi] = 0
                            continue
                        
                        # Find most similar strong weight
                        best_si = strong_idx[0]
                        best_sim = -999
                        
                        for si in strong_idx[:min(20, len(strong_idx))]:
                            sign_match = 1 if (weak_val * row[si]) > 0 else -1
                            sim = sign_match * (1 - min(abs(weak_val - row[si].abs()), 1.0))
                            if sim > best_sim:
                                best_sim = sim
                                best_si = si
                        
                        row[best_si] += weak_val * 0.6
                        row[wi] = 0
                        total_redistributed += 1
                
                param.data = weight.to(param.dtype)
            
            # Progress update
            elapsed = (time.time() - start_time) / 60
            print(f"\r{progress_bar(senator_idx+1, len(self.senators), f'Smart Prune ({elapsed:.1f}min)')}", end='')
            sys.stdout.flush()
        
        elapsed = (time.time() - start_time) / 60
        print(f"\r{progress_bar(len(self.senators), len(self.senators), 'Smart Prune')}")
        print(f"   Redistributed: {total_redistributed:,} weights | Time: {elapsed:.1f}min")
        sys.stdout.flush()
        return total_redistributed
    
    def quantize_to_float16(self):
        """Safe quantization to float16"""
        print(f"\n  STAGE 2: QUANTIZE TO FLOAT16")
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
        
        self.smart_prune(target_sparsity=target_sparsity, max_time_minutes=30)
        self.quantize_to_float16()
        
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
