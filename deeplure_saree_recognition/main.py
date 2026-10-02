"""
Command-Line Interface for Color-Invariant Saree Recognition.
Author: Computer Vision Engineering Team
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import SareeConfig
from src.train import train_model
from src.evaluate import run_full_evaluation
from src.benchmark_efficiency import run_full_efficiency_benchmark
from src.inference import SareeRecognizer
from src.models import ColorInvariantSareeNet

def main():
    parser = argparse.ArgumentParser(
        description="DeepLure Color-Invariant Saree Design Recognition System"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available Commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train model end-to-end")
    train_parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    train_parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    train_parser.add_argument("--backbone", type=str, default="efficientnet_b0", help="Backbone architecture")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Run full evaluation on gallery/query split")
    eval_parser.add_argument("--backbone", type=str, default="efficientnet_b0", help="Backbone architecture")

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Profile parameters, FLOPs, and latency")
    bench_parser.add_argument("--backbone", type=str, default="efficientnet_b0", help="Backbone architecture")

    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify whether two sarees carry the same design")
    verify_parser.add_argument("--img1", type=str, required=True, help="Path to first saree image")
    verify_parser.add_argument("--img2", type=str, required=True, help="Path to second saree image")
    verify_parser.add_argument("--threshold", type=float, default=0.65, help="Cosine threshold")

    # Identify command
    ident_parser = subparsers.add_parser("identify", help="Match a query saree against gallery")
    ident_parser.add_argument("--query", type=str, required=True, help="Path to query image")
    ident_parser.add_argument("--topk", type=int, default=5, help="Number of matches to return")

    args = parser.parse_args()

    if args.command == "train":
        config = SareeConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            backbone_name=args.backbone
        )
        train_model(config)

    elif args.command == "evaluate":
        config = SareeConfig(backbone_name=args.backbone)
        run_full_evaluation(config)

    elif args.command == "benchmark":
        print(f"[*] Benchmarking architecture: {args.backbone}...")
        model = ColorInvariantSareeNet(backbone_name=args.backbone, embedding_dim=256, pretrained=False)
        stats = run_full_efficiency_benchmark(model)
        import json
        print(json.dumps(stats, indent=2))

    elif args.command == "verify":
        recognizer = SareeRecognizer(verification_threshold=args.threshold)
        result = recognizer.verify(args.img1, args.img2)
        print("\n" + "=" * 45)
        print("         PAIRWISE VERIFICATION RESULT")
        print("=" * 45)
        print(f"Image 1:           {args.img1}")
        print(f"Image 2:           {args.img2}")
        print(f"Cosine Similarity: {result['cosine_similarity']:.4f}")
        print(f"Threshold:         {result['threshold']:.4f}")
        print(f"Same Design?:      {'YES (MATCH)' if result['is_same_design'] else 'NO (DISTINCT)'}")
        print(f"Confidence:        {result['confidence'] * 100:.1f}%")
        print("=" * 45)

    elif args.command == "identify":
        config = SareeConfig()
        from src.dataset import load_saree_corpus
        image_paths, labels, _ = load_saree_corpus(str(config.data_dir))
        recognizer = SareeRecognizer()
        recognizer.index_gallery(image_paths, labels)
        results = recognizer.identify(args.query, top_k=args.topk)
        print("\n" + "=" * 55)
        print(f"   TOP-{args.topk} GALLERY MATCHES FOR: {args.query}")
        print("=" * 55)
        for r in results:
            match_str = "[MATCH]" if r["is_confident_match"] else "[LOW CONFIDENCE]"
            print(f"Rank {r['rank']}: {r['gallery_path']} | Sim: {r['cosine_similarity']:.4f} {match_str}")
        print("=" * 55)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
