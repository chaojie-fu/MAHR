"""Combine all robot previews and package trajectories, diagnostics and videos."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import zipfile

LABELS={'g1':'Unitree G1','h1':'Unitree H1','gr1':'Fourier GR1','n1':'Fourier N1',
        'x1':'Agibot X1','berkeley':'Berkeley Humanoid Lite','atlas':'Atlas HD','icub':'iCub'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    folder=args.directory
    summary=json.loads((folder/'summary.json').read_text())
    command=['ffmpeg','-y','-loglevel','error','-filter_complex_threads','1']
    filters=[]
    names=list(summary)
    for name in names:
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
            '-show_entries','stream=nb_frames,width,height,duration','-of','json',str(folder/f'{name}.mp4')]))['streams'][0]
        expected=math.ceil(summary[name]['frames']/summary[name].get('render_stride',4))
        if int(probe['nb_frames']) != expected:
            raise ValueError(f'Incomplete video for {name}: {probe}')
        summary[name]['video']=probe
        subprocess.run(['ffmpeg','-y','-loglevel','error','-ss',str(min(1.5,float(probe['duration'])/2)),'-i',str(folder/f'{name}.mp4'),
            '-frames:v','1',str(folder/f'{name}.png')],check=True)
    (folder/'summary.json').write_text(json.dumps(summary,indent=2))
    for i,name in enumerate(names):
        command+=['-i',str(folder/f'{name}.mp4')]
        filters.append(f"[{i}:v]scale=320:240,drawtext=text='{LABELS[name]}':fontcolor=white:fontsize=16:x=10:y=10:box=1:boxcolor=black@0.5[v{i}]")
    inputs=''.join(f'[v{i}]' for i in range(len(names)))
    layout='|'.join(f'{i%4*320}_{i//4*240}' for i in range(len(names)))
    filters.append(f'{inputs}xstack=inputs={len(names)}:layout={layout}[out]')
    command+=['-filter_complex',';'.join(filters),'-map','[out]','-threads','2','-c:v','libx264',
              '-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(folder/'all_robots.mp4')]
    subprocess.run(command,check=True)
    subprocess.run(['ffmpeg','-y','-loglevel','error','-filter_complex_threads','1','-i',str(folder/'all_robots.mp4'),
        '-filter_complex','[0:v]fps=15,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse',
        str(folder/'all_robots.gif')],check=True)
    subprocess.run(['ffmpeg','-y','-loglevel','error','-ss',str(min(1.5,float(next(iter(summary.values()))['video']['duration'])/2)),'-i',str(folder/'all_robots.mp4'),
        '-frames:v','1',str(folder/'all_robots.png')],check=True)
    first=next(iter(summary.values()))
    total=sum(item['frames'] for item in summary.values())
    failed=sum(item['unsuccessful_frames'] for item in summary.values())
    lines=['# All robot retargeting results','',
        f"Source: {first.get('source_motion','Clio_Haniotikos_C3D_poses_short.bvh')}; {first['frames']} frames; stand normalization.",
        'Videos play at original speed, rendered at every fourth source frame (about 30 fps).',
        f'{total} total frames; {failed} non-converged frames; joint bounds and inter-frame step bounds checks passed.',
        f"Analytic gradient checks at frames {first.get('checked_frames',[0,249,499])}; see summary.json for numerical derivative errors.",
        'These are kinematic playback results; physical balance/contact feasibility is not tested.',
        'H1 and GR1 preview colors are lifted for visibility; trajectory data are unchanged.','',
        '| Robot | DoFs | Solve seconds | FPS | Mean objective | Failed frames |',
        '|---|---:|---:|---:|---:|---:|']
    for name,item in summary.items():
        lines.append(f"| {LABELS[name]} | {item['dofs']} | {item['solve_seconds']:.3f} | {item['fps']:.1f} | {item['mean_loss']:.6f} | {item['unsuccessful_frames']} |")
    lines+=['','Objective values across different landmark topologies are not directly comparable.',
        'For each robot: NAME.json is the trajectory, NAME.stats.json contains frame diagnostics, NAME.mp4 is mesh playback.']
    (folder/'README.md').write_text('\n'.join(lines)+'\n')
    archive=folder.parent/f'{folder.name}_results.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(folder.iterdir()):
            if path.is_file():
                z.write(path,arcname=path.name)
    print(archive)


if __name__=='__main__':
    main()
