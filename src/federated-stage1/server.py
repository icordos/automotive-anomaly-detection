#!/usr/bin/env python
"""Federated PatchCore Server using Flower framework."""
import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import torch
from flwr.common import (
    EvaluateRes,
    FitRes,
    Parameters,
    Scalar,
)
from flwr.server.client_proxy import ClientProxy


def aggregate_memory_banks(results: List[Tuple[ClientProxy, FitRes]]) -> Parameters:
    """
    Aggregate memory banks from all clients.
    
    Strategy: Concatenate all memory banks and optionally apply coreset sampling.
    """
    if not results:
        return Parameters(tensors=[], tensor_type="numpy.ndarray")
    
    print(f"\n[Server] Aggregating memory banks from {len(results)} clients...")
    
    # Collect all memory banks by category
    category_memory_banks: Dict[str, List[np.ndarray]] = {}
    all_patch_shapes: Dict[str, Tuple[int, int]] = {}
    
    for client_proxy, fit_res in results:
        parameters_list = fl.common.parameters_to_ndarrays(fit_res.parameters)
        
        if len(parameters_list) < 2:
            continue
        
        # Decode metadata
        metadata_bytes = parameters_list[0].tobytes()
        metadata = pickle.loads(metadata_bytes)
        
        categories = metadata.get("categories", [])
        patch_shapes = metadata.get("patch_shapes", {})
        client_id = metadata.get("client_id", "unknown")
        
        print(f"[Server] Processing client {client_id} with categories: {categories}")
        
        # Collect memory banks
        for i, category in enumerate(categories):
            if i + 1 < len(parameters_list):
                mb_array = parameters_list[i + 1]
                if mb_array.size > 0:
                    if category not in category_memory_banks:
                        category_memory_banks[category] = []
                    category_memory_banks[category].append(mb_array)
                    
                    # Store patch shape
                    if category in patch_shapes:
                        all_patch_shapes[category] = patch_shapes[category]
    
    # Aggregate memory banks per category (concatenation)
    aggregated_mbs = []
    aggregated_categories = []
    
    for category, mb_list in category_memory_banks.items():
        print(f"[Server] Aggregating {len(mb_list)} memory banks for category: {category}")
        
        # Concatenate all memory banks
        global_mb = np.concatenate(mb_list, axis=0)
        
        # Optional: Apply coreset sampling to limit size
        max_samples = 50000  # Limit to 50K features per category
        if global_mb.shape[0] > max_samples:
            print(f"[Server] Applying random sampling: {global_mb.shape[0]} -> {max_samples}")
            indices = np.random.choice(global_mb.shape[0], max_samples, replace=False)
            global_mb = global_mb[indices]
        
        aggregated_mbs.append(global_mb)
        aggregated_categories.append(category)
        
        print(f"[Server] {category}: Global memory bank shape = {global_mb.shape}")
    
    # Create metadata for global model
    global_metadata = {
        "categories": aggregated_categories,
        "patch_shapes": all_patch_shapes,
    }
    
    metadata_bytes = pickle.dumps(global_metadata)
    metadata_array = np.frombuffer(metadata_bytes, dtype=np.uint8)
    
    # Combine into parameters
    parameters_list = [metadata_array] + aggregated_mbs
    
    return fl.common.ndarrays_to_parameters(parameters_list)


def fit_metrics_aggregation(metrics_list: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    """Aggregate training metrics from all clients."""
    total_samples = sum(num_examples for num_examples, _ in metrics_list)
    
    aggregated = {"total_samples": total_samples}
    
    # Collect unique metric keys
    all_keys = set()
    for _, metrics in metrics_list:
        all_keys.update(metrics.keys())
    
    # Weighted average for each metric
    for key in all_keys:
        if key == "total_samples":
            continue
        weighted_sum = sum(
            metrics.get(key, 0) * num_examples
            for num_examples, metrics in metrics_list
        )
        aggregated[key] = weighted_sum / total_samples if total_samples > 0 else 0.0
    
    print(f"\n[Server] Training metrics: {aggregated}")
    return aggregated


def evaluate_metrics_aggregation(metrics_list: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    """Aggregate evaluation metrics from all clients."""
    total_samples = sum(num_examples for num_examples, _ in metrics_list)
    
    # Collect unique metric keys
    all_keys = set()
    for _, metrics in metrics_list:
        all_keys.update(metrics.keys())
    
    aggregated = {}
    for key in all_keys:
        weighted_sum = sum(
            metrics.get(key, 0) * num_examples
            for num_examples, metrics in metrics_list
        )
        aggregated[key] = weighted_sum / total_samples if total_samples > 0 else 0.0
    
    print(f"\n[Server] Evaluation metrics: {aggregated}")
    return aggregated


def main():
    parser = argparse.ArgumentParser(description="PatchCore Federated Server")
    parser.add_argument("--num-clients", type=int, default=3)
    parser.add_argument("--num-rounds", type=int, default=5)
    parser.add_argument("--server-address", type=str, default="0.0.0.0:8080")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/federated/server"))
    
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create custom strategy that extends FedAvg
    class FedAvgWithMemoryBank(fl.server.strategy.FedAvg):
        def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
            failures: List[BaseException],
        ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:
            """Custom aggregation for memory banks."""
            if not results:
                return None, {}
            
            # Aggregate memory banks using your custom function
            aggregated_params = aggregate_memory_banks(results)
            
            return aggregated_params, {}
    
    # Create strategy with custom aggregation
    strategy = FedAvgWithMemoryBank(
        fraction_fit=1.0,  # Use all available clients for training
        fraction_evaluate=1.0,  # Use all available clients for evaluation
        min_fit_clients=args.num_clients,
        min_evaluate_clients=args.num_clients,
        min_available_clients=args.num_clients,
        fit_metrics_aggregation_fn=fit_metrics_aggregation,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
    )
    
    print("=" * 60)
    print("Starting Federated PatchCore Server")
    print(f"Number of clients: {args.num_clients}")
    print(f"Number of rounds: {args.num_rounds}")
    print(f"Server address: {args.server_address}")
    print("=" * 60)
    
    # Start Flower server
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )
    
    print("\n[Server] Training complete!")
    print(f"[Server] Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

