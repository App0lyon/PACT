import os
import json
from datetime import datetime

class Logger:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.log_file = os.path.join(save_dir, "train_log.txt")
        self.metrics_file = os.path.join(save_dir, "metrics.json")
        self.metrics = []
        self._log(f"Logging started: {datetime.now()}")

    def _log(self, msg):
        print(msg)
        with open(self.log_file, "a") as f:
            f.write(msg + "\n")

    def log(self, msg):
        self._log(msg)

    def log_metrics(self, epoch, train_loss, val_loss, acc):
        entry = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "acc": acc}
        self.metrics.append(entry)
        with open(self.metrics_file, "w") as f:
            json.dump(self.metrics, f, indent=2)

    def save_checkpoint(self, model, optimizer, epoch, is_best=False):
        ckpt_path = os.path.join(self.save_dir, f"checkpoint_epoch{epoch}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, ckpt_path)
        if is_best:
            best_path = os.path.join(self.save_dir, "best_model.pth")
            torch.save(model.state_dict(), best_path)
            self._log(f"✅ Saved best model at epoch {epoch}")
