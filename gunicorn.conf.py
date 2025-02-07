import multiprocessing

# Server Socket
bind = "0.0.0.0:$PORT"  # For Render
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = 'sync'
worker_connections = 1000
timeout = 300  # 5 minutes
keepalive = 2

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'

# Process Naming
proc_name = 'horse-diagnosis'

# SSL config will be handled by Render
keyfile = None
certfile = None

def on_starting(server):
    """Run when server starts"""
    import os
    # Create required directories
    for folder in ['models', 'uploads', 'backups', 'logs', 'static', 'tmp']:
        os.makedirs(folder, exist_ok=True)