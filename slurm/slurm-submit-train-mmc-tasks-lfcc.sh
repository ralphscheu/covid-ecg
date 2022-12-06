#!/bin/bash
#SBATCH --job-name=train-mmc-tasks-lfcc
#SBATCH --nodes=1 # Anzahl benötigter Knoten
#SBATCH --ntasks=1 # Gesamtzahl der Tasks über alle Knoten hinweg
#SBATCH --partition=p1 # Verwendete Partition (z.B. p0, p1, p2 oder all)
#SBATCH --time=16:00:00 # Gesamtlimit für Laufzeit des Jobs (Format: HH:MM:SS)
#SBATCH --cpus-per-task=8 # Rechenkerne pro Task
#SBATCH --mem=16G # Gesamter Hauptspeicher pro Knoten
#SBATCH --gres=gpu:1 # Gesamtzahl GPUs pro Knoten
#SBATCH --qos=basic # Quality-of-Service
#SBATCH --mail-type=ALL # Art des Mailversands (gültige Werte z.B. ALL, BEGIN, END, FAIL oder REQUEUE)
#SBATCH --mail-user=scheuererra68323@th-nuernberg.de
echo "=================================================================="
echo "Starting Batch Job at $(date)"
echo "Job submitted to partition ${SLURM_JOB_PARTITION} on ${SLURM_CLUSTER_NAME}"
echo "Job name: ${SLURM_JOB_NAME}, Job ID: ${SLURM_JOB_ID}"
echo "Requested ${SLURM_CPUS_ON_NODE} CPUs on compute node $(hostname)"
echo "Working directory: $(pwd)"
echo "=================================================================="
###################### Optional for Pythonnutzer*innen #######################
# Die folgenden Umgebungsvariablen stellen sicher, dass
# Modellgewichte von Huggingface und PIP Packages nicht unter
# /home/$USER/.cache landen.
CACHE_DIR=/nfs/scratch/students/$USER/.cache
export PIP_CACHE_DIR=$CACHE_DIR
export TRANSFORMERS_CACHE=$CACHE_DIR
export HF_HOME=$CACHE_DIR
mkdir -p CACHE_DIR

echo $0 $@

########################################################

for modelName in CNN3DSeqMeanPool CNN3DSeqReducedMeanPool CNN3DSeqMeanStdPool CNN3DSeqReducedMeanStdPool CNN3DSeqAttnPool CNN3DSeqReducedAttnPool CNN3DSeqLSTM CNN3DSeqReducedLSTM CNN2DSeqReducedMeanStdPool CNN2DSeqReducedAttnPool CNN2DSeqMeanStdPool CNN2DSeqAttnPool CNN2DSeqReducedLSTM CNN2DSeqLSTM; do

    srun python3 covidecg/train.py --model $modelName --feats lfcc data/processed/mmc_tasks_lfcc/mmc_covid_vs_ctrl610

done

########################################################
exit 0
