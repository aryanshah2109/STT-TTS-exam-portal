bind = "0.0.0.0:8000"

worker_class = "uvicorn.workers.UvicornWorker"

workers = 1

preload_app = False
timeout = 180
keepalive = 5

accesslog = "-"
errorlog = "-"
loglevel = "info"
