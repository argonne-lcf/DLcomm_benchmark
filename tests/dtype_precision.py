#!/usr/bin/env python3

import torch
import numpy as np

def test_dtype_precision():
    """Test how different data types handle number storage and precision."""
    
    # Define data types to test
    dtypes = [
        ('float16', torch.float16),
        ('bfloat16', torch.bfloat16), 
        ('float32', torch.float32),
        ('float64', torch.float64),
        ('int8', torch.int8),
        ('int16', torch.int16),
        ('int32', torch.int32),
        ('int64', torch.int64),
    ]
    
    print("Testing data type precision for numbers 0 to 50000")
    print("=" * 80)
    
    for dtype_name, dtype in dtypes:
        print(f"\n{dtype_name.upper()}:")
        print("-" * 40)
        
        precision_lost_at = None
        
        # Test every number from 0 to 50000
        for i in range(0, 50001):
            try:
                # Create tensor with the number
                if dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
                    tensor = torch.tensor(float(i), dtype=dtype)
                    stored_value = tensor.item()
                    
                    # Check if precision is lost
                    if abs(stored_value - i) > 1e-6:
                        if precision_lost_at is None:
                            precision_lost_at = i
                        print(f"  {i:6d} -> {stored_value:12.6f} (precision lost)")
                    else:
                        print(f"  {i:6d} -> {stored_value:12.6f}")
                        
                else:  # Integer types
                    # Check if number fits in the dtype range
                    info = torch.iinfo(dtype)
                    if i > info.max:
                        if precision_lost_at is None:
                            precision_lost_at = i
                        print(f"  {i:6d} -> OVERFLOW (max: {info.max})")
                        break
                    else:
                        tensor = torch.tensor(i, dtype=dtype)
                        stored_value = tensor.item()
                        
                        if stored_value != i:
                            if precision_lost_at is None:
                                precision_lost_at = i
                            print(f"  {i:6d} -> {stored_value:6d} (value changed)")
                        else:
                            print(f"  {i:6d} -> {stored_value:6d}")
                            
            except Exception as e:
                print(f"  {i:6d} -> ERROR: {e}")
                if precision_lost_at is None:
                    precision_lost_at = i
                break
        
        if precision_lost_at:
            print(f"  *** First precision loss/overflow at: {precision_lost_at}")
        else:
            print(f"  *** No precision loss detected up to 50000")
            
        # Show dtype info
        if dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
            print(f"  Type info: floating-point")
        else:
            info = torch.iinfo(dtype)
            print(f"  Type info: integer range [{info.min}, {info.max}]")

def test_rank_assignment():
    """Test how rank assignments would work with different dtypes."""
    
    print("\n\nTEST: Rank assignment simulation")
    print("=" * 80)
    
    # Simulate 8 ranks with numbers from 0-7, 1000-1007, 10000-10007, 40000-40007
    rank_ranges = [
        (0, 8, "Small ranks (0-7)"),
        (1000, 1008, "Medium ranks (1000-1007)"), 
        (10000, 10008, "Large ranks (10000-10007)"),
        (40000, 40008, "Very large ranks (40000-40007)")
    ]
    
    dtypes = [
        ('float16', torch.float16),
        ('bfloat16', torch.bfloat16),
        ('float32', torch.float32),
        ('int8', torch.int8),
        ('int16', torch.int16),
        ('int32', torch.int32),
    ]
    
    for start, end, description in rank_ranges:
        print(f"\n{description}")
        print("-" * 50)
        
        for dtype_name, dtype in dtypes:
            print(f"  {dtype_name:8s}: ", end="")
            
            ranks = []
            for rank_id in range(start, end):
                try:
                    if dtype in [torch.float16, torch.bfloat16, torch.float32]:
                        tensor = torch.tensor(float(rank_id), dtype=dtype)
                    else:
                        # Check if fits in integer range
                        info = torch.iinfo(dtype)
                        if rank_id > info.max:
                            ranks.append("OVERFLOW")
                            continue
                        tensor = torch.tensor(rank_id, dtype=dtype)
                    
                    stored_value = tensor.item()
                    ranks.append(str(stored_value))
                    
                except Exception as e:
                    ranks.append("ERROR")
            
            print("[" + ", ".join(ranks) + "]")

if __name__ == "__main__":
    # Redirect output to file
    import sys
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"dtype_analysis_{timestamp}.txt"
    
    print(f"Writing dtype analysis to {output_file}...")
    
    with open(output_file, 'w') as f:
        # Redirect stdout to file
        original_stdout = sys.stdout
        sys.stdout = f
        
        test_dtype_precision()
        test_rank_assignment()
        
        # Restore stdout
        sys.stdout = original_stdout
    
    print(f"Analysis complete! Results saved to {output_file}")