This folder is divided into 2 sections:
1. centralized training - includes only Patch Core training and 3 results with 3 different configurations. The one we are using to compare with Federated learning is the 3 one "rezultate3". In each directory of the results is included the configuration for running on Politehnica's SLURM, with 2 text files for errors and outputs.

Steps for running centralized training:

1) cd centralized
2) sbatch proiect.slurm

2. federated - specific for federated training, also includes the Patch Core training used in the centralized training. This directory includes the code used for server and client design, and also the config files for running on FEP. We divided this into a job containing the server and the first client, and the second job involves the 2 clients remaining. The results from this directory are used with the same configuration from the third results from the centralized training.

Steps for running federated training, assuming you have downloade dataset:

1) cd federated
2) sbatch slurm_fl_server_client1.slurm
3) sbatch slurm_fl_client2_client3.slurm

Packages needed:
pip install mlcroissant numpy Pillow scikit-learn torch torchvision tqdm flwr


For comparision will be used logs from:

aitwdm-project/federated/rezultate1_federated/server_output.txt
aitwdm-project/centralized/rezultate3_centralized/results_centralized_output_max-samples-5000_ratio-0.1_batch-4_image-size-512.txt
