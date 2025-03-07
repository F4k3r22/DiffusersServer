from DiffusersServer import DiffusersServerApp

app = DiffusersServerApp(
    model='stabilityai/stable-diffusion-3.5-medium',
    threads=3,
    enable_memory_monitor=True
)