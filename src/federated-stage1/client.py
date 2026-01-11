#!/usr/bin/env python
"""Federated PatchCore Client using Flower framework."""
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch

from patchcore_training import (
    PatchCoreTrainer,
    PatchCoreTrainingConfig,
)


class PatchCoreFlowerClient(fl.client.NumPyClient):
    """
    Flower client for PatchCore anomaly detection.
    
    Instead of training a neural network, this client:
    1. Extracts features from local training data
    2. Builds a local memory bank (coreset)
    3. Sends the memory bank to the server
    4. Receives aggregated global memory bank
    5. Evaluates on local test data
    """
    
    def __init__(
        self,
        client_id: int,
        categories: List[str],
        config: PatchCoreTrainingConfig,
        device: str = "cuda",
    ):
        self.client_id = client_id
        self.categories = categories
        self.config = config
        self.device = device
        self.trainer = PatchCoreTrainer(config)
        
        # Store local memory banks per category
        self.local_memory_banks: Dict[str, torch.Tensor] = {}
        self.patch_shapes: Dict[str, Tuple[int, int]] = {}
        
    def get_parameters(self, config=None) -> List[np.ndarray]:
        """
        Return local memory banks as list of numpy arrays.
        
        Format: [category1_mb, category2_mb, ...]
        We also include metadata as the first element.
        """
        if not self.local_memory_banks:
            # Not trained yet, return empty
            return []
        
        parameters = []
        
        # Add metadata (category names and patch shapes)
        metadata = {
            "categories": self.categories,
            "patch_shapes": self.patch_shapes,
            "client_id": self.client_id,
        }
        # Convert metadata to numpy array (we'll encode it as bytes)
        import pickle
        metadata_bytes = pickle.dumps(metadata)
        parameters.append(np.frombuffer(metadata_bytes, dtype=np.uint8))
        
        # Add memory banks for each category
        for category in self.categories:
            if category in self.local_memory_banks:
                mb = self.local_memory_banks[category].cpu().numpy()
                parameters.append(mb)
            else:
                # Empty placeholder
                parameters.append(np.array([]))
        
        return parameters
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Receive global memory banks from server.
        
        Parameters format: [metadata, category1_mb, category2_mb, ...]
        """
        if len(parameters) < 2:
            return
        
        import pickle
        
        # Decode metadata
        metadata_bytes = parameters[0].tobytes()
        metadata = pickle.loads(metadata_bytes)
        
        # Load global memory banks
        categories = metadata.get("categories", self.categories)
        patch_shapes = metadata.get("patch_shapes", {})
        
        for i, category in enumerate(categories):
            if i + 1 < len(parameters):
                mb_array = parameters[i + 1]
                if mb_array.size > 0:
                    mb_tensor = torch.from_numpy(mb_array).to(self.device)
                    self.local_memory_banks[category] = mb_tensor
                    
                    # Load memory bank into trainer
                    if category in patch_shapes:
                        self.trainer.load_memory_bank(
                            mb_tensor,
                            patch_shapes[category]
                        )
    
    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, any],
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train local PatchCore model (extract features and build memory bank).
        
        Args:
            parameters: Global model parameters (not used in first round)
            config: Training configuration from server
        
        Returns:
            - Updated parameters (local memory banks)
            - Number of training samples
            - Training metrics
        """
        print(f"[Client {self.client_id}] Starting local training...")
        
        total_samples = 0
        training_metrics = {}
        
        for category in self.categories:
            print(f"[Client {self.client_id}] Training on category: {category}")
            
            # Extract features and build memory bank
            checkpoint_path, memory_bank, patch_shape = self.trainer.fit(category)
            
            # Store locally
            self.local_memory_banks[category] = memory_bank
            self.patch_shapes[category] = patch_shape
            
            # Count samples
            num_samples = memory_bank.shape[0]
            total_samples += num_samples
            
            training_metrics[f"{category}_samples"] = num_samples
            training_metrics[f"{category}_feature_dim"] = memory_bank.shape[1]
            
            print(f"[Client {self.client_id}] {category}: {num_samples} features extracted")
        
        # Return local memory banks
        metrics = {
            "total_samples": total_samples,
            **training_metrics,
        }
        
        return self.get_parameters(), total_samples, metrics
    
    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, any],
    ) -> Tuple[float, int, Dict]:
        """
        Evaluate global model on local test data.
        
        Args:
            parameters: Global memory banks from server
            config: Evaluation configuration
        
        Returns:
            - Loss (we use negative AUROC as loss)
            - Number of test samples
            - Evaluation metrics (AUROC, AP)
        """
        print(f"[Client {self.client_id}] Starting evaluation...")
        
        # Load global memory banks
        self.set_parameters(parameters)
        
        total_test_samples = 0
        all_metrics = {}
        avg_auroc = 0.0
        
        for category in self.categories:
            if category not in self.local_memory_banks:
                continue
            
            print(f"[Client {self.client_id}] Evaluating on category: {category}")
            
            # Evaluate
            metrics = self.trainer.evaluate(category)
            
            # Store metrics
            for key, value in metrics.items():
                all_metrics[f"{category}_{key}"] = value
            
            # Accumulate AUROC
            if not np.isnan(metrics.get("image_roc_auc", 0)):
                avg_auroc += metrics["image_roc_auc"]
            
            # We don't know exact test sample count, use placeholder
            total_test_samples += 100
            
            # ADAUGĂ: Curățare după fiecare categorie
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            print(f"[Client {self.client_id}] Memory cleaned after {category}")
        
        # Average AUROC across categories
        if len(self.categories) > 0:
            avg_auroc /= len(self.categories)
        
        # Use negative AUROC as loss (lower is better)
        loss = -avg_auroc
        
        all_metrics["avg_auroc"] = avg_auroc
        all_metrics["loss"] = loss
        
        # ADAUGĂ: Curățare finală
        torch.cuda.empty_cache()
        gc.collect()
        print(f"[Client {self.client_id}] Final memory cleanup completed")
        
        return float(loss), total_test_samples, all_metrics


def main():
    parser = argparse.ArgumentParser(description="PatchCore Federated Client")
    parser.add_argument("--client-id", type=int, required=True)
    parser.add_argument("--server", type=str, default="localhost:8080")
    parser.add_argument("--categories", nargs="+", required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/federated"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--coreset-ratio", type=float, default=0.1)
    parser.add_argument("--coreset-method", type=str, default="random")

    # ADAUGĂ ACESTE LINII NOI:
    parser.add_argument("--coreset-max-samples", type=int, default=100000,
                        help="Maximum number of samples in coreset")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--coreset-chunk-size", type=int, default=16384,
                        help="Chunk size for coreset computation")
    parser.add_argument("--distance-chunk-size", type=int, default=8192,
                        help="Chunk size for distance computation")
    parser.add_argument("--log-level", type=str, default="INFO",
                        help="Logging level")

    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    # Create client output directory
    client_output_dir = args.output_dir / f"client_{args.client_id}"
    client_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create PatchCore config
    config = PatchCoreTrainingConfig(
        dataset_root=args.dataset_root,
        output_dir=client_output_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        coreset_sampling_ratio=args.coreset_ratio,
        coreset_method=args.coreset_method,
        device=args.device,
    )
    
    # Create Flower client
    device = config.resolved_device()
    client = PatchCoreFlowerClient(
        client_id=args.client_id,
        categories=args.categories,
        config=config,
        device=device,
    )
    
    # Start Flower client
    print(f"[Client {args.client_id}] Connecting to server at {args.server}...")
    print(f"[Client {args.client_id}] Categories: {args.categories}")
    
    fl.client.start_client(
        server_address=args.server,
        client=client.to_client(),
    )


if __name__ == "__main__":
    main()

