import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from utils import DeepfakeDataLoader, get_model, freeze_backbone, validate_model, save_checkpoint, get_max_batch_size

def train_model_improved(model_name, num_epochs=None):
    """Train with IMPROVED REGULARIZATION to prevent overfitting"""
    
    # Smart epoch allocation - REDUCED to prevent overfitting
    if num_epochs is None:
        smart_epochs = {
            'xception': 12,      # Reduced from 20
            'efficientnet': 8,   # Reduced from 15  
            'resnet50': 10       # Reduced from 18
        }
        num_epochs = smart_epochs.get(model_name, 10)
        print(f"🎯 ANTI-OVERFITTING: {num_epochs} epochs for {model_name.upper()}")
    
    print(f"\n{'='*50}")
    print(f"🔧 IMPROVED TRAINING: {model_name.upper()}")
    print("🎯 ANTI-OVERFITTING MEASURES ACTIVATED")
    print(f"{'='*50}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print("[INFO] Cleared GPU cache.")

    # GPU optimization
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # REDUCED batch size to prevent overfitting
    batch_size = get_max_batch_size(model_name) // 2  # Half the original batch size
    print(f"[INFO] ANTI-OVERFITTING: Reduced batch size to {batch_size}")
    
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

    # IMPROVED TRAINING SETUP - STRONGER REGULARIZATION
    criterion = nn.BCEWithLogitsLoss()
    
    # LOWER learning rate and HIGHER weight decay
    optimizer = optim.AdamW(model.parameters(), 
                           lr=1e-4,           # Reduced from 2e-4
                           weight_decay=5e-4) # Increased from 1e-4
    
    # More aggressive learning rate scheduling
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.3, patience=2, verbose=True)
    scaler = torch.amp.GradScaler('cuda')

    # Backup old checkpoint and start fresh
    checkpoint_path = f'checkpoints/{model_name}_best.pth'
    if os.path.exists(checkpoint_path):
        backup_path = f'checkpoints/{model_name}_overfitted_backup.pth'
        os.rename(checkpoint_path, backup_path)
        print(f"🔄 Backed up overfitted model to {backup_path}")
    
    start_epoch = 0
    best_val_acc = 0.0

    # Training loop with early stopping based on VAL-TEST GAP
    train_start_time = time.time()
    val_test_gap_history = []
    
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()

        # Training phase with DROPOUT ENABLED
        model.train()
        train_loss = 0.0
        print_every = max(1, len(train_loader) // 50)  # More frequent logging

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

        # Test on a small sample to check overfitting
        test_loss, test_acc, _, _, _, _ = validate_model(
            model, test_loader, criterion, device
        )

        val_test_gap = val_acc - test_acc
        val_test_gap_history.append(val_test_gap)

        epoch_time = time.time() - epoch_start_time
        print(f'\nEpoch {epoch+1}/{num_epochs} (Time: {epoch_time:.2f}s):')
        print(f'  Train Loss: {train_loss/len(train_loader):.4f}')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        print(f'  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
        print(f'  🚨 VAL-TEST GAP: {val_test_gap:.4f} (target: <0.05)')
        print(f'  Val Precision: {val_precision:.4f}')
        print(f'  Val Recall: {val_recall:.4f}')
        print(f'  Val F1: {val_f1:.4f}')
        print(f'  Val AUC: {val_auc:.4f}')
        print('-' * 50)

        # Learning rate scheduling based on validation accuracy
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        print(f'  Current LR: {current_lr:.2e}')
        
        # IMPROVED SAVING STRATEGY - Consider both val acc and generalization gap
        generalization_good = val_test_gap < 0.05  # Gap less than 5%
        
        if val_acc > best_val_acc and generalization_good:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_acc, model_name, is_best=True)
            print(f"🎯 NEW BEST GENERALIZABLE MODEL: Val={val_acc:.4f}, Test={test_acc:.4f}, Gap={val_test_gap:.4f}")
            
            # Early stopping if we achieve good accuracy with good generalization
            if val_acc >= 0.90 and val_test_gap < 0.03:  # 90% with <3% gap
                print(f"🚀 EXCELLENT GENERALIZATION ACHIEVED!")
                print(f"Val: {val_acc:.4f}, Test: {test_acc:.4f}, Gap: {val_test_gap:.4f}")
                break
        elif val_acc > best_val_acc:
            print(f"⚠️  Val acc improved ({val_acc:.4f}) but gap too large ({val_test_gap:.4f})")
        
        # Stop if overfitting is getting worse
        if epoch >= 3 and val_test_gap > 0.15:  # Gap > 15%
            print(f"🛑 STOPPING: Severe overfitting detected (gap: {val_test_gap:.4f})")
            break

    total_train_time = time.time() - train_start_time
    actual_epochs = epoch + 1
    
    print(f"\n{'='*50}")
    print(f"✅ IMPROVED {model_name.upper()} TRAINING COMPLETED!")
    print(f"🎯 Best validation accuracy: {best_val_acc:.4f}")
    print(f"⏱️  Training time: {total_train_time:.2f}s")
    print(f"📊 Epochs completed: {actual_epochs}/{num_epochs}")
    print(f"{'='*50}")

    return best_val_acc, total_train_time

def main():
    """Main function for improved anti-overfitting training"""
    
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        print(f"🔥 GPU: {gpu_name}")
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.95)

    # Train all models with anti-overfitting measures
    models_to_train = ['xception', 'efficientnet', 'resnet50']
    results = {}
    
    print(f"\n🎯 ANTI-OVERFITTING TRAINING FOR ALL MODELS")
    print(f"Models: {', '.join([m.upper() for m in models_to_train])}")

    for model_name in models_to_train:
        try:
            best_acc, train_time = train_model_improved(model_name)
            results[model_name] = {'best_acc': best_acc, 'train_time': train_time}
        except Exception as e:
            print(f"❌ Error training {model_name}: {str(e)}")
            results[model_name] = {'best_acc': 0.0, 'train_time': 0.0}

    # Final summary
    print(f"\n{'='*60}")
    print("🏆 ANTI-OVERFITTING TRAINING SUMMARY")
    print(f"{'='*60}")
    for model_name, result in results.items():
        print(f"{model_name.upper():12}: {result['best_acc']:.4f} (Time: {result['train_time']:.2f}s)")
    
    print(f"\n🎯 Now run 'python evaluate.py' to check if overfitting is fixed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()