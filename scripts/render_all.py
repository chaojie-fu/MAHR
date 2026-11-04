"""Render all validated trajectories at their original playback speed."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from mahr.batch import load_batch_summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    parser.add_argument('--width',type=int,default=640)
    parser.add_argument('--height',type=int,default=480)
    args=parser.parse_args()
    batch = any((args.directory/name).is_file()
                for name in ('batch_progress.jsonl', 'batch_summary.json'))
    summary = (load_batch_summary(args.directory) if batch
               else json.loads((args.directory/'summary.json').read_text()))
    items = ([(item['robot'], item) for item in summary['tasks'] if item['status'] != 'failed']
             if batch else list(summary.items()))
    env=dict(os.environ,MUJOCO_GL='osmesa',LP_NUM_THREADS='4')
    for name,item in items:
        trajectory = args.directory/item['trajectory'] if batch else Path(item['trajectory'])
        output = trajectory.with_suffix('.mp4') if batch else args.directory/f'{name}.mp4'
        subprocess.run([sys.executable,str(Path(__file__).with_name('render_motion.py')),
            str(trajectory),'--urdf',item['urdf'],'-o',str(output),
            '--width',str(args.width),'--height',str(args.height)] +
            (['--enhance-dark-materials'] if name in ('h1','gr1') else []),check=True,env=env)


if __name__=='__main__':
    main()
