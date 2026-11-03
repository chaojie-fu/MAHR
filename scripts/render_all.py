"""Render all validated trajectories at their original playback speed."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    summary=json.loads((args.directory/'summary.json').read_text())
    env=dict(os.environ,MUJOCO_GL='osmesa',LP_NUM_THREADS='4')
    for name,item in summary.items():
        subprocess.run([sys.executable,str(Path(__file__).with_name('render_motion.py')),
            item['trajectory'],'--urdf',item['urdf'],'-o',str(args.directory/f'{name}.mp4'),
            '--width','640','--height','480'] +
            (['--enhance-dark-materials'] if name in ('h1','gr1') else []),check=True,env=env)
        item['render_stride']=4
        (args.directory/'summary.json').write_text(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
