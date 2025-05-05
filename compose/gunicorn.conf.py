bind = "0.0.0.0:8000"
workers = 4
threads = 8
timeout = 30            # kill unresponsive workers
accesslog = "-"         # stdout → docker log
errorlog  = "-"         # stderr → docker log
