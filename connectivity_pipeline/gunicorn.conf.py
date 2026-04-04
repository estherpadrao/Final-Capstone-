# Gunicorn configuration for Render deployment
#
# WHY gthread instead of sync:
# The default sync worker uses SIGALRM for its timeout. When the signal fires
# during osmnx's time.sleep() (Overpass API rate-limit pause), gunicorn kills
# the worker mid-request. gthread workers isolate SIGALRM to the main thread,
# so sleeping inside a request-handling thread is never interrupted.

worker_class = "gthread"
workers      = 1        # Render free tier: 1 is enough; avoids OOM
threads      = 4        # Each request runs in its own thread
timeout      = 600      # 10 min — OSM + network build can take ~8 min on Render
keepalive    = 5
