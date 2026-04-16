# %%
import os
from PIL import Image
from collections import Counter
from rich import print
import sys
import pandas as pd
from natsort import natsorted

main = sys.argv[1]
vids = [v for v in os.listdir(main) if os.path.isdir(os.path.join(main, v))]
print(
    f"> Verifying {main} folder - {len(vids)} video folders found"
)  # ,os.listdir("/home/shared-nearmrs/"+main))
flags = []
for v in natsorted(vids):
    errors_here = []
    if "IGNORE" in v:
        print(f"[yellow]Ignoring as flagged{v}[/yellow]")
        continue
    frames_path = os.listdir(os.path.join(main, v, "frame"))
    # Are the required files there?
    try:
        n_rgbs = len(frames_path)
    except:
        print(f"[red]ERROR: {v} - no 'frame' directory'[/red]")
        errors_here.append(f"ERROR: {v} - no 'frame' directory'")
        flags.append(f"ERROR: {v} - no 'frame' directory'")
        continue
    try:
        gt = pd.read_csv(os.path.join(main, v, v + "_gt.csv"), header=None)
    except:
        print(f"[red]ERROR: {v} - no {os.path.join(main,v,v+'_gt.csv')} file[/red]")
        errors_here.append(f"ERROR: {v} - no {os.path.join(main,v,v+'_gt.csv')} file")
        flags.append(f"ERROR: {v} - no {os.path.join(main,v,v+'_gt.csv')} file")
        continue

    # Are there the same number of RGBs and poses?
    if len(gt) != n_rgbs:
        print(
            f"[red]>>>> ERROR: {v} - {n_rgbs} RGBs and {len(gt)} poses in the GT file[/red]"
        )
        errors_here.append(
            f"ERROR: {v} - {n_rgbs} RGBs and {len(gt)} poses in the GT file"
        )
        flags.append(f"ERROR: {v} - {n_rgbs} RGBs and {len(gt)} poses in the GT file")

    # Are the RGBs and poses corresponding?
    poses_path = gt.iloc[:, 7].values
    # Convert to list
    for pose in poses_path:
        if not os.path.exists(pose):
            print(
                f"[red]ERROR: {v} - File {pose} (listed in the GT file) does not exist[/red]"
            )
            errors_here.append(
                f"ERROR: {v} - File {pose} (listed in the GT file) does not exist"
            )
            flags.append(
                f"ERROR: {v} - File {pose} (listed in the GT file) does not exist"
            )

    for frame in frames_path:
        # print(os.path.join(main,v,'frame',frame))
        if os.path.join(main, v, "frame", frame) not in poses_path:
            print(
                f"[red]ERROR: {v} - There is no row in the groundtruth file with {os.path.join(main,v,'frame',frame)}[/red]"
            )
            errors_here.append(
                f"ERROR: {v} - There is no row in the groundtruth file with {os.path.join(main,v,'frame',frame)}"
            )
            flags.append(
                f"ERROR: {v} - There is no row in the groundtruth file with {os.path.join(main,v,'frame',frame)}"
            )
    color = "red" if len(errors_here) else "green"
    print(
        f"[{color}][{main}{v}] - {len(frames_path)} frames : Found {len(errors_here)} errors[/{color}]"
    )

color = "red" if len(flags) else "green"
print(f"[{color}]{main}{v} : Found {len(flags)} errors[/{color}]")
