#!/usr/bin/env python3
import glob, os, sys, torch

def count_lora_keys(ckpt_path: str):
    sd = torch.load(ckpt_path, map_location='cpu')
    if isinstance(sd, dict) and all(isinstance(k, str) for k in sd.keys()):
        keys = list(sd.keys())
    elif isinstance(sd, dict) and 'state_dict' in sd:
        keys = list(sd['state_dict'].keys())
    else:
        try:
            keys = list(sd.state_dict().keys())
        except Exception:
            keys = []
    lora = [k for k in keys if 'lora' in k.lower()]
    return len(lora), lora[:20]

def main():
    out_dir = '/root/tecGPT/Output/GIMtec/TEC_MoLLM'
    ckpts = sorted(glob.glob(os.path.join(out_dir, '*.pth')), key=os.path.getmtime)
    if not ckpts:
        print('No checkpoints found in', out_dir)
        sys.exit(0)
    ckpt = ckpts[-1]
    n, sample = count_lora_keys(ckpt)
    report = os.path.join(out_dir, 'lora_report.txt')
    with open(report, 'w') as f:
        f.write(f'checkpoint: {ckpt}\n')
        f.write(f'num_lora_keys: {n}\n')
        f.write('sample_keys:\n')
        for k in sample:
            f.write(f'  {k}\n')
    print('checkpoint:', ckpt)
    print('num_lora_keys:', n)
    print('sample_keys:')
    for k in sample:
        print(' ', k)

if __name__ == '__main__':
    main()
