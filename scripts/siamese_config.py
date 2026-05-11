"""Configuration for full Siamese training with physical residuals"""
import argparse

def get_full_training_args():
    parser = argparse.ArgumentParser(description="Full Siamese Network Training")
    
    # Input/output arguments
    parser.add_argument('--residuals', type=str, default='residuals_dict.pth',
                        help='Path to the standardized residuals file (.pth)')
    parser.add_argument('--output-dir', type=str, default='siamese_results_physical', 
                        help='Output directory')
    
    # Data processing arguments
    parser.add_argument('--fft-length', type=int, default=128, 
                        help='Length of FFT features')
    parser.add_argument('--num-channels', type=int, default=8,  # Updated for 8 channels
                        help='Number of channels in residuals [default: 8 when including physical]')
    parser.add_argument('--include-physical', action='store_true', default=True,  # Enable physical residuals
                        help='Include physical residuals when available')
    parser.add_argument('--num-triplets', type=int, default=20000, 
                        help='Number of triplets to generate')
    parser.add_argument('--hierarchy-level', type=int, default=3, choices=[1, 2, 3],
                        help='Hierarchy level for training (1=group, 2=data_type, 3=data_type+speed)')
    
    # Training arguments - Optimized for full training
    parser.add_argument('--batch-size', type=int, default=128, 
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=150,  # More epochs for better training
                        help='Maximum number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.0005, 
                        help='Learning rate')
    parser.add_argument('--embedding-size', type=int, default=128,  # Larger embedding for more channels
                        help='Size of embedding vector')
    parser.add_argument('--margin', type=float, default=1.0, 
                        help='Margin for triplet loss')
    parser.add_argument('--distance', type=str, default='cosine', choices=['cosine', 'euclidean'],
                        help='Distance metric for triplet loss')
    parser.add_argument('--n-folds', type=int, default=5, 
                        help='Number of cross-validation folds')
    parser.add_argument('--patience', type=int, default=20,  # More patience for more epochs
                        help='Patience for early stopping')
    parser.add_argument('--workers', type=int, default=20, 
                        help='Number of worker processes for data loading')
    
    # CNN arguments - Deeper network for full training with 8 channels
    parser.add_argument('--cnn-layers', type=int, default=3,
                        help='Number of CNN layers')
    parser.add_argument('--kernel-sizes', type=int, nargs='+', default=[5, 3, 3],
                        help='Kernel sizes for CNN layers')
    parser.add_argument('--filters', type=int, nargs='+', default=[64, 128, 256],  # More filters for 8 channels
                        help='Number of filters for CNN layers')
    parser.add_argument('--pool-sizes', type=int, nargs='+', default=[2, 2, 2],
                        help='Pool sizes for CNN layers')
    parser.add_argument('--activation', type=str, default='leaky_relu',  # Better for deeper networks
                        choices=['relu', 'leaky_relu', 'elu'],
                        help='Activation function for CNN')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate for CNN')
    
    # Grid search - enable by default
    parser.add_argument('--grid-search', action='store_true', default=True,
                        help='Perform grid search for hyperparameters')
    
    # Other arguments - keep defaults or enhance
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], 
                        help='Device to use (default: auto-detect)')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    parser.add_argument('--verbose', type=int, default=2,  # More verbose output
                        help='Verbosity level (0=silent, 1=progress, 2=detailed)')
    
    # Add all other parameters with their defaults
    parser.add_argument('--wavelet', type=str, default='db4', 
                        help='Wavelet type for feature extraction')
    parser.add_argument('--wavelet-level', type=int, default=3, 
                        help='Wavelet decomposition level')
    parser.add_argument('--include-tsfresh', action='store_true', default=False,
                        help='Include tsfresh features (computationally intensive)')
    parser.add_argument('--sampling-rate', type=float, default=50000.0,
                        help='Sampling rate in Hz for frequency-domain features')
    parser.add_argument('--num-workers', type=int, default=20, 
                       help='Number of worker processes for parallel feature extraction')
    parser.add_argument('--feature-batch-size', type=int, default=10,
                       help='Number of samples to process per worker batch for feature extraction')
    parser.add_argument('--threshold', type=float, default=0.5, 
                        help='Similarity threshold for classification')
    parser.add_argument('--tsne-perplexity', type=float, default=30.0,
                        help='Perplexity for t-SNE visualization')
    
    return parser.parse_args([])  # Return the default values

# Example usage:
if __name__ == "__main__":
    from siamese_analysis.cli import run_training
    
    args = get_full_training_args()
    print("Starting training with physical residuals (8 channels)")
    print(f"Using include_physical={args.include_physical}, num_channels={args.num_channels}")
    run_training(args) 