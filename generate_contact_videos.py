import seaborn as sns
import pandas as pd
import mdtraj as md
import os
from numba import jit, njit, prange
import numpy as np
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import gridspec


def read_traj(trajectory, topol):
    return md.traj(trajectory, top=topol)

def save_fig(figname,outputdir,format="png",dpi=300,bbox_inches='tight', transparent=False):
    from pathlib import Path
    Path(f"{outputdir}").mkdir(parents=True, exist_ok=True)

    plt.savefig(f"{outputdir}/{figname}.{format}", dpi=dpi, bbox_inches=bbox_inches, transparent=transparent, facecolor="white")


def compute_distance(traj):
    CAs = traj.top.select("name CA")
    pairs = []
    indexes = []
    convert_idx_res = {}
    for i in range(len(CAs)):
        convert_idx_res[i] = traj.top.atom(CAs[i]).residue.resSeq
        for j in range(i, len(CAs)):
            pairs.append((CAs[i],CAs[j]))
            indexes.append((i,j))

    distances = md.compute_distances(traj, pairs)

    return (convert_idx_res, indexes, CAs, distances)


@jit(nopython=False, parallel=True)
def get_matrices(distances, indexes, CAs):
    nframes = distances.shape[0]
    npairs = distances.shape[1]
    matrices = np.zeros((nframes, len(CAs),len(CAs)), dtype=np.float32())
    for n in range(nframes):
        for i in range(npairs):
            idx_i = indexes[i][0]
            idx_j = indexes[i][1]
            d = distances[n][i]
            matrices[n, idx_j,idx_i] = d

    average_matrix = matrices.mean(axis=0)
    std_matrix = matrices.std(axis=0).T
    summatry_matrix = average_matrix+std_matrix
    summatry_matrix [summatry_matrix == 0] = np.nan


    matrices [matrices == 0] = np.nan

    return (average_matrix,matrices)


def plot_matrix(matrix, frameNumber=None, basename="distance_", maxFrame=None, maxval=10, highlights=[], convert_idx_res={}):
    # fig, ax = plt.subplots(figsize=(10,8))

    if maxFrame != None:
        fig = plt.figure(constrained_layout=True, figsize=(10,10))
        gs = gridspec.GridSpec(2, 1, figure=fig,height_ratios=[10,0.5], hspace=0.01)
        ax0 = fig.add_subplot(gs[0, :])
        ax1 = fig.add_subplot(gs[1, :])
        matrix = matrix + np.nan_to_num(matrix,0).T
    else:
        fig, ax0 = plt.subplots(figsize=(10,8))

    g = sns.heatmap(matrix, cmap="viridis", ax=ax0, vmin=0, vmax=maxval,cbar_kws = {"shrink":0.5})
    old_xticklabels = g.get_xticklabels()
    old_yticklabels = g.get_yticklabels()
    new_xticklabels = [str(convert_idx_res[int(t.get_text())]) for t in old_xticklabels]
    new_yticklabels = [str(convert_idx_res[int(t.get_text())]) for t in old_yticklabels]
    _ = g.set_xticklabels(g.get_xticklabels(), rotation = 45, fontsize = 8)
    _ = g.set_yticklabels(g.get_yticklabels(), rotation = 0, fontsize = 8)
    _ = g.set_title("Distance between residues", fontsize=20)
    _ = g.set_xlabel("Residue numbers", fontsize=10)

    import matplotlib.patches as patches
    for box in highlights:
        #format = box[0] = start, box[1] = end, box[2] = color
        height = box[1] - box[0]
        patch = patches.Rectangle((0,box[0]),matrix.shape[1],height,linewidth=2, edgecolor=box[2], facecolor="none", linestyle="dashed")
        ax0.add_patch(patch)

    if maxFrame != None:
        ax1.vlines(x=SIMULATIONTIME/maxFrame*frameNumber, color='black', ymin=-1, ymax=2)
        ax1.vlines(x=0, color='black', ymin=0.25, ymax=0.75, linewidth=0.5)
        ax1.vlines(x=SIMULATIONTIME, color='black', ymin=0.25, ymax=0.75, linewidth=0.5)
        for i in range(0, SIMULATIONTIME, 25):
            ax1.vlines(x=i, color='black', ymin=0.35, ymax=0.65, linewidth=0.5)
            ax1.text(x=i-2, y=0, s=i)
        ax1.text(x=SIMULATIONTIME-2, y=0, s=SIMULATIONTIME)
        ax1.text(x=SIMULATIONTIME/2-15, y=-0.5, s="Simulation time (ns)")
        ax1.hlines(y=0.5, color='black', xmin=0, xmax=SIMULATIONTIME, linewidth=0.5)
        ax1.axis('off')
        ax1.set_xlim((0,SIMULATIONTIME+1))
        ax1.set_ylim((0,1))
        ax1.set_yticks([])
        ax1.set_xlabel("Simulation time (ns)")
    plt.tight_layout()
    if frameNumber:
        name = f"{basename}{frameNumber:05}"
    else:
        name = basename
    save_fig(name, outputdir="analysis/PNG/distances/",dpi=300,transparent="false")
    plt.close()


def prepare_and_run_calculations(SIMULATIONTIME, basefolder, apoeNumber, replica):
    workdir = f"{baseFolder}/apoe{apoeNumber}/results/replica_{replica}/prod"
    os.chdir(workdir)
    trajectory = f"md_APOE{apoeNumber}_clean_nowat.xtc"
    topology = f"md_APOE{apoeNumber}_clean_nowat.pdb"
    traj = md.load(trajectory, top=topology)

    #Calculate all distances
    convert_idx_res, indexes, CAs, distances = compute_distance(traj)

    #Convert them into distance matrices
    average_matrix, matrices = get_matrices(distances, indexes, CAs)

    #Prepare the Highlighting list
    convert_res_idx = {v: k for k, v in convert_idx_res.items()}
    highlights = [(convert_res_idx[24],convert_res_idx[41],"magenta"),
                (convert_res_idx[55],convert_res_idx[80],"green"),
                (convert_res_idx[90],convert_res_idx[125],"blue"),
                (convert_res_idx[131],convert_res_idx[165],"red"),
                ]

    #plot the averate + std matrix
    plot_matrix(average_matrix, basename="average_std_distances", highlights=highlights,convert_idx_res=convert_idx_res)

    maxval = np.nanmax(matrices)
    for i in tqdm(range(len(matrices[::4]))):
        plot_matrix(matrices[i], frameNumber=i, maxFrame=SIMULATIONTIME, highlights=highlights, maxval=maxval,convert_idx_res=convert_idx_res)


if __name__ == "__main__":
    SIMULATIONTIME = 200
    baseFolder="D:/work/ApoE/simulation/apoe"
    if len(sys.argv) > 1:
        baseFolder = sys.argv[1]
        print(f"basefolder is {baseFolder}") # TODO <- to be improved with argparse.

    if len(sys.argv) == 4:
        apoe=[sys.argv[2]]
        replicas=[sys.argv[3]]
    else:
        print("loop mode on all apoo and systems")
        apoe=[1,2,3,4]
        replicas = [1,2,3]

    for system in apoe:
        for replica in replicas:
            print(f"System = APOE{system} - Replica {replica}")
            prepare_and_run_calculations(SIMULATIONTIME, baseFolder, system,replica)

