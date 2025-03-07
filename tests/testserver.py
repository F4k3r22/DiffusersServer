from DiffusersServer import DiffusersServerApp

app = DiffusersServerApp(
    model='black-forest-labs/FLUX.1-schnell',
    type_model='t2im',
    threads=3,
    enable_memory_monitor=True
)