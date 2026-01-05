bind = "0.0.0.0:8000"

worker_class = "uvicorn.workers.UvicornWorker"

workers = 3

preload_app = True 
timeout = 180
keepalive = 5

accesslog = "-"
errorlog = "-"
loglevel = "info"
