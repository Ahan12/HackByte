from flask import Flask, jsonify
import requests

app = Flask(__name__)

# Replace with the YOLO device's actual IP
YOLO_IP = "192.168.80.166"
YOLO_SERVER_URL = f"http://{YOLO_IP}:5001/download_csv"
LOCAL_FILE_PATH = "downloaded_detected_objects.csv"

@app.route("/fetch_csv", methods=["GET"])
def fetch_csv():
    """Fetches CSV from the YOLO device and saves it locally."""
    try:
        response = requests.get(YOLO_SERVER_URL, stream=True)
        if response.status_code == 200:
            with open(LOCAL_FILE_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
            return jsonify({"status": "success", "message": f"File saved as {LOCAL_FILE_PATH}"})
        else:
            return jsonify({"status": "error", "message": f"Failed to fetch CSV. Status code: {response.status_code}"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
