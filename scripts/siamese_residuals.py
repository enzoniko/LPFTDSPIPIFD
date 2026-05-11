import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import argparse
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ResidualProcessor:
    def __init__(self, fft_length=128, num_channels=4):
        """
        Process residuals to fixed-length FFT representations
        
        Args:
            fft_length: Length of FFT output to keep (will take first N frequencies)
            num_channels: Number of channels in the residual data
        """
        self.fft_length = fft_length
        self.num_channels = num_channels
        self.scaler = StandardScaler()
        self.fitted = False
    
    def extract_data(self, sample):
        """Extract data from sample dictionary if needed"""
        if isinstance(sample, dict) and 'data' in sample:
            return sample['data']
        return sample
    
    def process_sample(self, sample):
        """Process a single sample to FFT representation"""
        data = self.extract_data(sample)
        
        # Convert to numpy if needed
        if hasattr(data, 'cpu'):
            data = data.cpu().numpy()
        data = np.asarray(data)
        
        # Ensure we have 2D data (time, channels)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # Apply FFT to each channel
        fft_features = []
        for i in range(min(data.shape[1], self.num_channels)):
            # Compute FFT magnitude
            fft_vals = np.abs(np.fft.rfft(data[:, i]))
            # Take first N components
            if len(fft_vals) >= self.fft_length:
                fft_features.append(fft_vals[:self.fft_length])
            else:
                # Zero-pad if too short
                padding = np.zeros(self.fft_length - len(fft_vals))
                fft_features.append(np.concatenate([fft_vals, padding]))
        
        # If we have fewer channels than expected, pad with zeros
        while len(fft_features) < self.num_channels:
            fft_features.append(np.zeros(self.fft_length))
        
        # Stack channels
        return np.stack(fft_features, axis=0)
    
    def fit_transform(self, samples):
        """Process a list of samples and fit the scaler"""
        processed = [self.process_sample(sample) for sample in tqdm(samples, desc="Processing samples")]
        # Reshape for scaler: (n_samples, n_channels, fft_length) -> (n_samples, n_channels*fft_length)
        flat_data = np.array([p.flatten() for p in processed])
        
        # Fit and transform
        normalized = self.scaler.fit_transform(flat_data)
        self.fitted = True
        
        # Reshape back to original format
        return normalized.reshape(-1, self.num_channels, self.fft_length)
    
    def transform(self, samples):
        """Process a list of samples using pre-fitted scaler"""
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit_transform first.")
        
        processed = [self.process_sample(sample) for sample in tqdm(samples, desc="Processing samples")]
        flat_data = np.array([p.flatten() for p in processed])
        normalized = self.scaler.transform(flat_data)
        return normalized.reshape(-1, self.num_channels, self.fft_length)


class TripletGenerator:
    def __init__(self, data_by_class, processor):
        """
        Generate triplets for training
        
        Args:
            data_by_class: Dictionary mapping class names to lists of samples
            processor: ResidualProcessor instance
        """
        self.data_by_class = data_by_class
        self.processor = processor
        self.processed_data = {}
        
        # Process all data
        for class_name, samples in data_by_class.items():
            if class_name == 'normal':
                # Use fit_transform for normal class to fit the scaler
                self.processed_data[class_name] = self.processor.fit_transform(samples)
            else:
                # Use transform for other classes
                self.processed_data[class_name] = self.processor.transform(samples)
    
    def generate_triplets(self, num_triplets, classes_to_use=None):
        """
        Generate triplets (anchor, positive, negative)
        
        Args:
            num_triplets: Number of triplets to generate
            classes_to_use: List of class names to use (None for all)
        
        Returns:
            anchors, positives, negatives: Arrays of shape (num_triplets, channels, fft_length)
        """
        if classes_to_use is None:
            classes_to_use = list(self.processed_data.keys())
        
        anchors = []
        positives = []
        negatives = []
        
        for _ in tqdm(range(num_triplets), desc="Generating triplets"):
            # Select anchor class
            anchor_class = np.random.choice(classes_to_use)
            
            # Select negative class (different from anchor)
            available_neg_classes = [c for c in classes_to_use if c != anchor_class]
            if not available_neg_classes:  # If only one class is available
                continue
            negative_class = np.random.choice(available_neg_classes)
            
            # Select two different samples from anchor class
            if len(self.processed_data[anchor_class]) < 2:
                continue  # Skip if not enough samples
            
            anchor_idx, positive_idx = np.random.choice(
                len(self.processed_data[anchor_class]), 
                size=2, 
                replace=False
            )
            
            # Select a sample from negative class
            negative_idx = np.random.randint(len(self.processed_data[negative_class]))
            
            # Add to lists
            anchors.append(self.processed_data[anchor_class][anchor_idx])
            positives.append(self.processed_data[anchor_class][positive_idx])
            negatives.append(self.processed_data[negative_class][negative_idx])
        
        return np.array(anchors), np.array(positives), np.array(negatives)


class TripletDataset(Dataset):
    def __init__(self, anchors, positives, negatives):
        """
        Dataset for triplet loss training
        
        Args:
            anchors, positives, negatives: Arrays of shape (num_triplets, channels, fft_length)
        """
        self.anchors = torch.FloatTensor(anchors)
        self.positives = torch.FloatTensor(positives)
        self.negatives = torch.FloatTensor(negatives)
    
    def __len__(self):
        return len(self.anchors)
    
    def __getitem__(self, idx):
        return {
            'anchor': self.anchors[idx],
            'positive': self.positives[idx],
            'negative': self.negatives[idx]
        }


class SiameseNetwork(nn.Module):
    def __init__(self, in_channels=4, fft_length=128, embedding_size=16):
        """
        Siamese network with CNN encoder
        
        Args:
            in_channels: Number of input channels
            fft_length: Length of FFT features
            embedding_size: Size of embedding vector
        """
        super(SiameseNetwork, self).__init__()
        
        self.encoder = nn.Sequential(
            # First conv block
            nn.Conv1d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Second conv block
            nn.Conv1d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        
        # Calculate size after convolutions and pooling
        conv_output_size = fft_length // 4 * 8
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size, embedding_size)
        )
    
    def forward_one(self, x):
        """Encode one input"""
        x = self.encoder(x)
        x = self.fc(x)
        # L2 normalize embeddings
        x = F.normalize(x, p=2, dim=1)
        return x
    
    def forward(self, x1, x2=None):
        """
        Forward pass
        
        Args:
            x1: First input
            x2: Optional second input
        
        Returns:
            If x2 is provided, returns embeddings for both inputs
            Otherwise, returns embedding for x1
        """
        embedding1 = self.forward_one(x1)
        
        if x2 is not None:
            embedding2 = self.forward_one(x2)
            return embedding1, embedding2
        
        return embedding1


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        """
        Triplet loss with hard margin
        
        Args:
            margin: Margin for triplet loss
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        """
        Compute triplet loss
        
        Args:
            anchor, positive, negative: Embeddings
        
        Returns:
            loss: Mean triplet loss
        """
        # Compute distances
        pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
        neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
        
        # Compute triplet loss
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        
        # Return mean loss
        return torch.mean(loss)


class SiameseTrainer:
    def __init__(self, model, device, lr=0.001):
        """
        Trainer for Siamese network
        
        Args:
            model: SiameseNetwork instance
            device: torch.device for training
            lr: Learning rate
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = TripletLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        
        for batch in tqdm(dataloader, desc="Training"):
            # Get data
            anchor = batch['anchor'].to(self.device)
            positive = batch['positive'].to(self.device)
            negative = batch['negative'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            anchor_embedding = self.model.forward_one(anchor)
            positive_embedding = self.model.forward_one(positive)
            negative_embedding = self.model.forward_one(negative)
            
            # Compute loss
            loss = self.criterion(anchor_embedding, positive_embedding, negative_embedding)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
        
        return running_loss / len(dataloader)
    
    def train(self, train_dataloader, epochs=100, val_dataloader=None):
        """
        Train the model
        
        Args:
            train_dataloader: DataLoader for training
            epochs: Number of epochs
            val_dataloader: Optional validation DataLoader
        
        Returns:
            history: Dictionary of training metrics
        """
        history = {'train_loss': []}
        if val_dataloader:
            history['val_loss'] = []
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_dataloader)
            history['train_loss'].append(train_loss)
            
            # Validate
            if val_dataloader:
                val_loss = self.evaluate(val_dataloader)
                history['val_loss'].append(val_loss)
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")
        
        return history
    
    def evaluate(self, dataloader):
        """Evaluate the model"""
        self.model.eval()
        running_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Get data
                anchor = batch['anchor'].to(self.device)
                positive = batch['positive'].to(self.device)
                negative = batch['negative'].to(self.device)
                
                # Forward pass
                anchor_embedding = self.model.forward_one(anchor)
                positive_embedding = self.model.forward_one(positive)
                negative_embedding = self.model.forward_one(negative)
                
                # Compute loss
                loss = self.criterion(anchor_embedding, positive_embedding, negative_embedding)
                
                running_loss += loss.item()
        
        return running_loss / len(dataloader)
    
    def save_model(self, path):
        """Save the model"""
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        """Load the model"""
        self.model.load_state_dict(torch.load(path))


class SiameseEvaluator:
    def __init__(self, model, processor, device):
        """
        Evaluator for Siamese network
        
        Args:
            model: Trained SiameseNetwork instance
            processor: ResidualProcessor instance
            device: torch.device for evaluation
        """
        self.model = model.to(device)
        self.processor = processor
        self.device = device
        self.model.eval()
    
    def compute_embeddings(self, data_by_class):
        """
        Compute embeddings for all samples
        
        Args:
            data_by_class: Dictionary mapping class names to lists of samples
        
        Returns:
            embeddings_by_class: Dictionary mapping class names to embeddings
            labels: List of labels corresponding to all embeddings
            all_embeddings: Array of all embeddings
        """
        embeddings_by_class = {}
        all_embeddings = []
        labels = []
        
        with torch.no_grad():
            for class_name, samples in data_by_class.items():
                # Process samples
                processed_samples = self.processor.transform(samples)
                
                # Convert to torch tensor
                tensors = torch.FloatTensor(processed_samples).to(self.device)
                
                # Compute embeddings in batches
                embeddings = []
                batch_size = 32
                for i in range(0, len(tensors), batch_size):
                    batch = tensors[i:i+batch_size]
                    batch_embeddings = self.model.forward_one(batch)
                    embeddings.append(batch_embeddings.cpu().numpy())
                
                # Concatenate batches
                embeddings = np.concatenate(embeddings, axis=0)
                
                # Store
                embeddings_by_class[class_name] = embeddings
                all_embeddings.append(embeddings)
                labels.extend([class_name] * len(embeddings))
        
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        return embeddings_by_class, labels, all_embeddings
    
    def compute_similarity_matrix(self, embeddings_by_class):
        """
        Compute similarity matrix between all classes
        
        Args:
            embeddings_by_class: Dictionary mapping class names to embeddings
        
        Returns:
            similarity_matrix: DataFrame with mean similarities between classes
        """
        classes = list(embeddings_by_class.keys())
        n_classes = len(classes)
        similarity_matrix = np.zeros((n_classes, n_classes))
        
        for i, class1 in enumerate(classes):
            for j, class2 in enumerate(classes):
                # Compute pairwise cosine similarities
                embeddings1 = embeddings_by_class[class1]
                embeddings2 = embeddings_by_class[class2]
                
                # Compute all pairs of similarities
                similarity_sum = 0
                count = 0
                
                for emb1 in embeddings1:
                    for emb2 in embeddings2:
                        # Skip comparing a sample with itself
                        if i == j and np.array_equal(emb1, emb2):
                            continue
                        
                        # Compute cosine similarity
                        similarity = np.dot(emb1, emb2)
                        similarity_sum += similarity
                        count += 1
                
                # Compute mean similarity
                if count > 0:
                    similarity_matrix[i, j] = similarity_sum / count
        
        import pandas as pd
        return pd.DataFrame(similarity_matrix, index=classes, columns=classes)
    
    def evaluate_classification(self, data_by_class, threshold=0.5):
        """
        Evaluate the model's ability to classify samples
        
        Args:
            data_by_class: Dictionary mapping class names to lists of samples
            threshold: Similarity threshold for classification
        
        Returns:
            report: Classification report
            cm: Confusion matrix
        """
        # Compute embeddings
        embeddings_by_class, labels, _ = self.compute_embeddings(data_by_class)
        classes = list(embeddings_by_class.keys())
        
        # Create prototype embeddings for each class
        prototypes = {
            cls: np.mean(embeddings_by_class[cls], axis=0) 
            for cls in classes
        }
        
        # Predict classes for all samples
        y_true = []
        y_pred = []
        
        for true_class, embeddings in embeddings_by_class.items():
            for embedding in embeddings:
                y_true.append(true_class)
                
                # Compute similarities to prototypes
                similarities = {
                    cls: np.dot(embedding, prototypes[cls]) 
                    for cls in classes
                }
                
                # Predict class with highest similarity
                pred_class = max(similarities.items(), key=lambda x: x[1])[0]
                y_pred.append(pred_class)
        
        # Compute confusion matrix and classification report
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        report = classification_report(y_true, y_pred, labels=classes)
        
        return report, cm, classes


def preprocess_residuals(args):
    """
    Load and preprocess residuals
    
    Args:
        args: Command line arguments
    
    Returns:
        data_by_class: Dictionary mapping class names to lists of samples
        data_metadata: Dictionary with metadata
    """
    # Load residuals
    print(f"Loading residuals from {args.residuals}...")
    residuals_dict = torch.load(args.residuals)
    
    if args.source == "direct":
        # Process direct PINN residuals
        from Data.LoadData import data_paths, get_omegas
        
        for key in residuals_dict:
            print(f"Original length for {key}: {len(residuals_dict[key])}")
            file_path = data_paths[key]
            # Compute rotation speeds (e.g., in Hz) from file metadata
            omegas = get_omegas(file_path) / (2 * np.pi)
            num_rotations = 1
            sampling_rate = 50000  # 50kHz
            datapoints_per_rotation = sampling_rate / omegas
            datapoints_needed = np.ceil(num_rotations * datapoints_per_rotation).to(torch.int32)
            n_blocks = len(residuals_dict[key]) // 250000
            segments = []
            for i in range(n_blocks):
                seg_length = int(datapoints_needed[i]) if i < len(datapoints_needed) else 200
                # For "normal" samples, take up to min(30, floor(250000 / seg_length)) segments; for others, only 10 segments.
                if key == 'normal':
                    max_segments = min(30, 250000 // seg_length)
                else:
                    max_segments = min(10, 250000 // seg_length)
                for j in range(max_segments):
                    start_idx = i * 250000 + j * seg_length
                    end_idx = start_idx + seg_length
                    if end_idx <= (i + 1) * 250000:
                        segment = residuals_dict[key][start_idx:end_idx]
                        if len(segment) > 0:
                            segments.append({'data': segment, 'rot_speed': float(omegas[i])})
            residuals_dict[key] = segments
            print(f"Segmented into {len(segments)} samples for {key}")
    
    elif args.source == "hybrid":
        # Hybrid RNN residuals are already segmented with rotation speed information
        processed = {}
        for data_type, sequences in tqdm(residuals_dict.items(), desc="Processing hybrid RNN residuals"):
            processed_sequences = []
            for seq_idx, seq_data in sequences.items():
                # Extract the data and rotation speed from the sequence
                data = seq_data['data'].numpy() if hasattr(seq_data['data'], 'numpy') else seq_data['data']
                rot_speed = seq_data['rot_speed']
                
                # Create a dictionary with the data and rotation speed
                processed_seq = {
                    'data': data,
                    'rot_speed': rot_speed
                }
                processed_sequences.append(processed_seq)
                
            residuals_dict[data_type] = processed_sequences
    
    # Print statistics
    print("\nResiduals statistics:")
    for key, sequences in residuals_dict.items():
        print(f"  {key}: {len(sequences)} sequences")
    
    return residuals_dict


def train_siamese_model(args):
    """Main training function"""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load and preprocess residuals
    data_by_class = preprocess_residuals(args)
    
    # Create processor
    processor = ResidualProcessor(
        fft_length=args.fft_length,
        num_channels=args.num_channels
    )
    
    # Create triplet generator
    generator = TripletGenerator(data_by_class, processor)
    
    # Generate triplets
    print(f"Generating {args.num_triplets} triplets...")
    anchors, positives, negatives = generator.generate_triplets(
        args.num_triplets,
        classes_to_use=args.classes_to_use if args.classes_to_use else None
    )
    
    # Create dataset and dataloader
    dataset = TripletDataset(anchors, positives, negatives)
    
    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers
    )
    
    # Create model
    model = SiameseNetwork(
        in_channels=args.num_channels,
        fft_length=args.fft_length,
        embedding_size=args.embedding_size
    )
    
    # Create trainer
    trainer = SiameseTrainer(model, device, lr=args.learning_rate)
    
    # Train model
    print("Training model...")
    history = trainer.train(
        train_dataloader,
        epochs=args.epochs,
        val_dataloader=val_dataloader
    )
    
    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(os.path.join(args.output_dir, "siamese_model.pt"))
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, "training_history.png"))
    
    # Evaluate model
    print("Evaluating model...")
    evaluator = SiameseEvaluator(model, processor, device)
    
    # Compute similarity matrix
    embeddings_by_class, _, _ = evaluator.compute_embeddings(data_by_class)
    similarity_matrix = evaluator.compute_similarity_matrix(embeddings_by_class)
    
    # Plot similarity matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_matrix, annot=True, cmap='Blues', vmin=0, vmax=1)
    plt.title('Class Similarity Matrix')
    plt.savefig(os.path.join(args.output_dir, "similarity_matrix.png"))
    
    # Evaluate classification
    report, cm, classes = evaluator.evaluate_classification(data_by_class)
    
    # Print classification report
    print("\nClassification Report:")
    print(report)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(args.output_dir, "confusion_matrix.png"))
    
    # Save results
    with open(os.path.join(args.output_dir, "classification_report.txt"), 'w') as f:
        f.write(report)
    
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Siamese Neural Network for Residuals")
    parser.add_argument('--residuals', type=str, required=True, help='Path to the residuals file (.pth)')
    parser.add_argument('--source', type=str, choices=['direct', 'hybrid'], default='direct',
                        help='Source type of residuals: direct PINN or hybrid RNN-PINN')
    parser.add_argument('--output-dir', type=str, default='siamese_results', help='Output directory')
    parser.add_argument('--workers', type=int, default=4, help='Number of worker processes')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--fft-length', type=int, default=128, help='Length of FFT features')
    parser.add_argument('--num-channels', type=int, default=4, help='Number of channels in residuals')
    parser.add_argument('--embedding-size', type=int, default=16, help='Size of embedding vector')
    parser.add_argument('--num-triplets', type=int, default=5000, help='Number of triplets to generate')
    parser.add_argument('--classes-to-use', nargs='+', help='Classes to use for training (default: all)')
    
    args = parser.parse_args()
    
    train_siamese_model(args)
