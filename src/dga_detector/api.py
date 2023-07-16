from flask import Flask, abort, jsonify, request

from dga_detector.mllib import DgaDetector
from dga_detector.urlparser import get_urls_from_text, is_valid_url

app = Flask(__name__)
detector = DgaDetector("models/base_lr.pkl")


@app.route("/get_prediction/<url_string>", methods=["GET"])
def classify(url_string):
    if is_valid_url(url_string):
        return jsonify(detector.predict(url_string))
    else:
        abort(422, "Invalid domain name")


@app.route("/get_predictions", methods=["POST"])
def batch_classify():
    urls = get_urls_from_text(request.get_json()["url_string"])
    if urls:
        return jsonify(detector.predict_many(urls))
    else:
        abort(422, "No domain name recognized")


if __name__ == "__main__":
    app.run(debug=True)
