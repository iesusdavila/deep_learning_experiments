def warmup_decay_schedule(base_lr, warmup_steps, total_steps, current_step):
    """
    Compute the learning rate at a given step using warmup + linear decay.
    """
    if current_step < warmup_steps:
        # Linear warmup
        lr = base_lr * (current_step / warmup_steps)
    else:
        # Linear decay
        lr = base_lr * (total_steps - current_step) / (total_steps - warmup_steps)
    return lr

# base_lr = 0.1, warmup_steps = 10, total_steps = 100, current_step = 5
print(warmup_decay_schedule(base_lr=0.1, warmup_steps=10, total_steps=100, current_step=5))  # Expected output: 0.05

