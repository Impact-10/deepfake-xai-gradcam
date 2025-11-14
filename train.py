import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from utils import DeepfakeDataLoader, get_model, freeze_backbone, validate_model, save_checkpoint

def train_model(model_name, num_epochs=None):
    """Smart training with model-specific epochs and intelligent early stopping"""
    
    # 🧠 SMART EPOCH ALLOCATION based on model characteristics
    if num_epochs is None:
        smart_epochs = {
            'xception': 20,      # Complex architecture, already trained
            'efficientnet': 15,  # Efficient design, converges faster  
            'resnet50': 18       # Residual learning, medium convergence
        }
        num_epochs = smart_epochs.get(model_name, 15)
        print(f"🎯 Smart epoch allocation: {num_epochs} epochs for {model_name.upper()}")
    
    print(f"\n{'='*50}")
    print(f"Training {model_name.upper()}")
    print(f"{'='*50}")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Clear GPU memory
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print("[INFO] Cleared GPU cache.")

    # GPU optimization
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # Load data with smart batch size
    from utils import get_max_batch_size
    batch_size = get_max_batch_size(model_name)
    print(f"[INFO] AGGRESSIVE GPU UTILIZATION - Using batch size: {batch_size}")
    
    dataloader = DeepfakeDataLoader(batch_size=batch_size, model_name=model_name)
    train_loader, val_loader, test_loader = dataloader.get_dataloaders()

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Initialize model
    model = get_model(model_name).to(device)
    freeze_backbone(model, model_name)

    # Count parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")

    # Training setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    scaler = torch.amp.GradScaler('cuda')

    # Resume from checkpoint
    checkpoint_path = f'checkpoints/{model_name}_best.pth'
    start_epoch = 0
    best_val_acc = 0.0

    if os.path.exists(checkpoint_path):
        print(f"Found checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_acc = checkpoint['val_acc']
            print(f"✅ Resumed from epoch {start_epoch}, best_val_acc={best_val_acc:.4f}")
        except Exception as e:
            print(f"⚠️  Checkpoint incompatible: {str(e)}")
            print(f"🔄 Starting fresh training")
            os.remove(checkpoint_path)
            start_epoch = 0
            best_val_acc = 0.0

    # Training loop
    train_start_time = time.time()
    
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()

        # Training phase
        model.train()
        train_loss = 0.0
        print_every = max(1, len(train_loader) // 100)

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels.float().unsqueeze(1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            if batch_idx % print_every == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')

        # Validation phase
        val_loss, val_acc, val_precision, val_recall, val_f1, val_auc = validate_model(
            model, val_loader, criterion, device
        )

        epoch_time = time.time() - epoch_start_time
        print(f'\nEpoch {epoch+1}/{num_epochs} (Time: {epoch_time:.2f}s):')
        print(f'  Train Loss: {train_loss/len(train_loader):.4f}')
        print(f'  Val Loss: {val_loss:.4f}')
        print(f'  Val Accuracy: {val_acc:.4f}')
        print(f'  Val Precision: {val_precision:.4f}')
        print(f'  Val Recall: {val_recall:.4f}')
        print(f'  Val F1: {val_f1:.4f}')
        print(f'  Val AUC: {val_auc:.4f}')
        print('-' * 50)

        # Learning rate scheduling
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        print(f'  Current LR: {current_lr:.2e}')
        
        # Save checkpoint if improved
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_acc, model_name, is_best=True)
            print(f"🎯 NEW BEST ACCURACY: {val_acc:.4f} - Model saved!")
            
            # 🚀 INTELLIGENT EARLY STOPPING with optimized thresholds
            if val_acc >= 0.975:  # 97.5% - Excellent threshold
                print(f"🚀 EXCELLENT ACCURACY ACHIEVED! {val_acc:.4f} >= 97.5%")
                print("🏆 PERFECT DEEPFAKE DETECTOR - EARLY STOPPING!")
                print(f"💨 Saved {num_epochs - epoch - 1} epochs, moving to next model...")
                break  # Exit training loop early
            elif val_acc >= 0.96:  # 96% - Outstanding threshold  
                print(f"🎯 OUTSTANDING ACCURACY! {val_acc:.4f} >= 96%")
                print("🔥 PRODUCTION-READY DEEPFAKE DETECTOR!")
            elif val_acc >= 0.95:
                print(f"✅ EXCELLENT ACCURACY! {val_acc:.4f} >= 95%")
                print("🎪 READY FOR DEEPFAKE DETECTION!")
        
        # Progress tracking
        print(f"📊 Progress: {val_acc:.1%} accuracy (Target: 97.5%+)")
        print(f"📉 Loss trend: {val_loss:.4f}")

    total_train_time = time.time() - train_start_time
    actual_epochs = epoch + 1
    epochs_saved = num_epochs - actual_epochs
    
    print(f"\n{'='*50}")
    print(f"✅ {model_name.upper()} TRAINING COMPLETED!")
    print(f"🎯 Best accuracy: {best_val_acc:.4f}")
    print(f"⏱️  Training time: {total_train_time:.2f}s")
    print(f"📊 Epochs completed: {actual_epochs}/{num_epochs}")
    if epochs_saved > 0:
        print(f"⚡ Epochs saved by early stopping: {epochs_saved}")
        print(f"💨 Time saved: ~{epochs_saved * (total_train_time/actual_epochs):.0f}s")
    print(f"{'='*50}")

    return best_val_acc, total_train_time

def main():
    # RTX 3050 optimization
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory // 1024**3
        print(f"🔥 NVIDIA GPU DETECTED: {gpu_name}")
        print(f"💾 GPU Memory: {gpu_memory}GB")
        
        if "3050" in gpu_name:
            print("🎯 RTX 3050 DETECTED - OPTIMIZED TRAINING!")
        
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.95)

    # 🚀 SMART MODEL TRAINING SEQUENCE
    models_to_train = ['xception', 'efficientnet', 'resnet50']
    results = {}
    
    print(f"\n🎯 TRAINING MODELS")
    print(f"Training: {', '.join([m.upper() for m in models_to_train])}")

    for model_name in models_to_train:
        try:
            best_acc, train_time = train_model(model_name)
            results[model_name] = {'best_acc': best_acc, 'train_time': train_time}
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"⚠️  CUDA OOM for {model_name}! Try reducing batch size")
                torch.cuda.empty_cache()
            results[model_name] = {'best_acc': 0.0, 'train_time': 0.0}
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            results[model_name] = {'best_acc': 0.0, 'train_time': 0.0}

    # Final summary
    print(f"\n{'='*60}")
    print("🏆 FINAL TRAINING SUMMARY")
    print(f"{'='*60}")
    for model_name in models_to_train:
        result = results.get(model_name, {'best_acc': 0.0, 'train_time': 0.0})
        print(f"{model_name.upper()}: Acc={result['best_acc']:.4f}, Time={result['train_time']:.2f}s")
    
    print(f"\n🎯 ALL MODELS TRAINED - DEEPFAKE DETECTION SYSTEM READY!")

if __name__ == "__main__":
    main()