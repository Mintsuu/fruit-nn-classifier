# ----------- experiment_log.py (or append to Util.py) -----------------------
import csv, os, datetime, json, torch.nn as nn

CSV_PATH = "experiments.csv"          # saved at repo root

def _model_spec(model: nn.Module) -> str:
    """
    Returns a semi-human string like:
    conv1:Conv2d(432) ; conv2:Conv2d(4 608) ; fc1:Linear(102 528) ; …
    showing leaf modules and their trainable param counts.
    """
    leaf_lines = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:            # leaf layer
            params = sum(p.numel() for p in module.parameters()
                         if p.requires_grad)
            leaf_lines.append(f"{name}:{module.__class__.__name__}({params})")
    return " ; ".join(leaf_lines)

def log_experiment(model, augmentation_tag,
                   batch_size, total_inputs,
                   accuracy, precision, recall, f1, confusion_matrix,
                   epochs, model_notes="",
                   ts=None):
    """
    Appends a row to experiments.csv, creating header if file doesn't exist.
    """
    if ts is None:
        ts = datetime.datetime.now().isoformat(timespec="seconds")
    row = {
         "timestamp"     : ts,
         "model_name"    : model.__class__.__name__,
         "model_layers"  : _model_spec(model),
         "augmentation"  : augmentation_tag,
         "train_batch"   : batch_size,
         "train_inputs"  : total_inputs,
         "epochs"        : epochs,
         "accuracy"      : round(float(accuracy),  4),
         "precision"     : round(float(precision), 4),
         "recall"        : round(float(recall),    4),
         "f1_score"      : round(float(f1),        4),
         "conf_matrix"   : json.dumps(confusion_matrix.tolist()),
         "model_notes"   : model_notes
     }

    new_file = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if new_file:
            writer.writeheader()
        writer.writerow(row)
# ---------------------------------------------------------------------------
