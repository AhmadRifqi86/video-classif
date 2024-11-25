from flask import Flask, request, jsonify
from pymongo import MongoClient


app = Flask(__name__)

# MongoDB Setup
# MONGO_URI = all_config.MONGO_URI
# DATABASE_NAME = all_config.DATABASE_NAME
# COLLECTION_NAME = all_config.COLLECTION_NAME

MONGO_URI = "mongodb://mongodb:27017/"
DATABASE_NAME = "video_classification"
COLLECTION_NAME = "classification_results"

client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]


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
@app.route('/video_labels', methods=['GET'])
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

# Run the Flask App
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
