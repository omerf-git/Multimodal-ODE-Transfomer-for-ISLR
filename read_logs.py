from tensorboard.backend.event_processing import event_accumulator

# TensorBoard event dosyasının yolu
event_file = "/home/omer/Masaüstü/tez_calismasi/codebase/ChaLearn-2021-LAP/logs/VTN_HCPF/version_22/events.out.tfevents.1750525745.omer-System-Product-Name"

# EventAccumulator ile dosyayı yükle
ea = event_accumulator.EventAccumulator(event_file)
ea.Reload()

# İlgili metrikler
metrics = [
    "epoch",
    "train_loss",
    "train_accuracy",
    "val_loss",
    "val_accuracy",
    "lr-Adam",
    "hp_metric"
]

# Her metrik için step->value eşlemesi oluştur
def get_last_per_epoch(events):
    """Her epoch için son değeri döndürür (step -> value)"""
    last_per_epoch = {}
    for event in events:
        last_per_epoch[event.step] = event.value
    return last_per_epoch

metric_steps = {m: get_last_per_epoch(ea.Scalars(m)) for m in metrics}

# Epoch numaralarını sırala
epochs = sorted(metric_steps["epoch"].values())
steps = sorted(metric_steps["epoch"].keys())

print(f"{'Epoch':<5} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12} {'LR':<10} {'HP Metric':<10}")
print("-" * 75)

def format_value(val, width=12):
    if val is None:
        return f"{'-':<{width}}"
    else:
        return f"{val:<{width}.6f}"

for step, epoch in zip(steps, epochs):
    train_loss = metric_steps["train_loss"].get(step)
    train_acc = metric_steps["train_accuracy"].get(step)
    val_loss = metric_steps["val_loss"].get(step)
    val_acc = metric_steps["val_accuracy"].get(step)
    lr = metric_steps["lr-Adam"].get(step)
    hp_metric = metric_steps["hp_metric"].get(step)

    print(f"{int(epoch):<5} "
          f"{format_value(train_loss)}"
          f"{format_value(train_acc)}"
          f"{format_value(val_loss)}"
          f"{format_value(val_acc)}"
          f"{format_value(lr, width=10)}"
          f"{format_value(hp_metric, width=10)}")