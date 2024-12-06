from flask import Flask, request, jsonify
from pymongo import MongoClient
import zmq

app = Flask(__name__)


MONGO_URI = "mongodb://localhost:27017/"
DATABASE_NAME = "video_classification"
COLLECTION_NAME = "classification_results"

client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

context = zmq.Context()
socket = None

try:
    socket = context.socket(zmq.PUSH)  # Using PUSH socket to send URLs to the worker
    socket.connect("tcp://localhost:54000")  # Connect to the worker's queue
    print("ZeroMQ socket bound successfully to tcp://*:54000")
except zmq.ZMQError as e:
    print(f"Error binding ZeroMQ socket: {str(e)}")
    socket = None  # Mark socket as unavailable
finally:
    if socket is None:
        print("Socket binding failed. Exiting.")
        exit(1)  # Terminate the application if binding fails

# POST Method
@app.route('/classify', methods=['POST'])
def classify_video():
    try:
        data = request.json
        url = data.get("url")
        labels = data.get("labels")

        if not url or not labels:
            return jsonify({"error": "Missing url or labels in request"}), 400

        # Insert into MongoDB
        document = {"url": url, "labels": labels}
        result = collection.insert_one(document)  # Insert and get the result

        # Prepare response with ObjectId converted to string
        document["_id"] = str(result.inserted_id)

        return jsonify({"message": "Classification result saved", "data": document}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# GET Method
@app.route('/video_labels', methods=['GET'])  #Endpoint for inference
def get_classification():
    try:
        url = request.args.get("url")
        if not url:
            return jsonify({"error": "Missing url parameter"}), 400

        # Query MongoDB
        result = collection.find_one({"url": url}, {"_id": 0})  # Exclude `_id` in the response
        if not result:
            return jsonify({"error": "No classification found for the given URL"}), 404

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_labels', methods=['GET'])  # Endpoint for inference
def get_classification_client():
    try:
        url = request.args.get("url")
        if not url:
            return jsonify({"error": "Missing url parameter"}), 400

        # Query MongoDB first
        result = collection.find_one({"url": url}, {"_id": 0})
        if result:
            return jsonify(result), 200  # Return if found in MongoDB

        # If URL not found, send it to the ZeroMQ worker
        try:
            # Send to ZeroMQ (PUSH)
            socket.send_string(url)
            print(f"URL {url} pushed to ZeroMQ queue.")
        except zmq.ZMQError as e:
            return jsonify({"error": f"Failed to push URL to queue: {str(e)}"}), 500

        # Wait for the worker to process and insert the result into MongoDB
        try:
            max_retries = 15000  # Limit the number of retries to avoid infinite loops
            retries = 0

            while retries < max_retries:
                result = collection.find_one({"url": url}, {"_id": 0})
                if result:
                    print(f"Received classification result for {url} from MongoDB.")
                    return jsonify(result), 200
                retries += 1

            # If no result after retries
            return jsonify({"error": "Worker did not insert the classification result in time"}), 500

        except Exception as e:
            return jsonify({"error": f"Failed to process classification: {str(e)}"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run the Flask App
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)



#Test classify using curl
#curl -X GET "http://localhost:5000/get_labels?url=https://www.tiktok.com/@balb444l/video/7444471595550248247"