import os
from app import app

if __name__ == "__main__":
    # Get port from environment variable or default to 9099
    port = int(os.environ.get("PORT", 9099))
    app.run(host='0.0.0.0', port=port) 