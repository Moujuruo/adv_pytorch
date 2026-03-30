import argparse
import os
import sys
import subprocess

def main():
    parser = argparse.ArgumentParser(description='Run face adversarial attack generation')
    parser.add_argument('--mode', type=str, required=True, choices=['obfuscation', 'target'],
                       help='Generation mode: obfuscation or target')
    args = parser.parse_args()

    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.mode == 'obfuscation':
        print("Running obfuscation mode...")
        script_path = os.path.join(current_dir, 'generate.py')
        subprocess.run([sys.executable, script_path])
        
    elif args.mode == 'target':
        print("Running target mode...")
        script_path = os.path.join(current_dir, 'iterative_generate.py')
        subprocess.run([sys.executable, script_path])

if __name__ == '__main__':
    main()