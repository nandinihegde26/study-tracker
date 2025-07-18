from flask import Flask, request, jsonify, render_template
from model import recommend_video

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    chapter = data.get("chapter", "").strip().lower()

    if not chapter:
        return jsonify({"error": "No chapter provided"}), 400

    result = recommend_video(chapter)
    return jsonify({"recommendation": result})

if __name__ == '__main__':
    app.run(debug=True)






