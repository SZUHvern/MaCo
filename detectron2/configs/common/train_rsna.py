# Common training-related configs that are designed for "tools/lazyconfig_train_net.py"
# You can use your own instead, together with your own train_net.py
train = dict(
    output_dir="./output",
    init_checkpoint="",
    max_iter=358400,
    amp=dict(enabled=False),  # options for Automatic Mixed Precision
    ddp=dict(  # options for DistributedDataParallel
        broadcast_buffers=False,
        find_unused_parameters=False,
        fp16_compression=False,
    ),
    checkpointer=dict(period=1120, max_to_keep=100),  # options for PeriodicCheckpointer
    eval_period=112,
    log_period=112,
    device="cuda"
    # ...
)
