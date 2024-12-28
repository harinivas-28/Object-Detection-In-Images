from flask import Flask, request, jsonify
from process_input import main
from flask_cors import CORS

app = Flask(__name__)
CORS(app,resources={r"/api/*": {"origins": "http://localhost:5173"}}, supports_credentials=True)

@app.route('/api/process', methods=['POST'])
def process():
    input_data = request.files.get('input') or request.form.get('input')
    input_type = request.form.get('inputType')
    overlapped = request.form.get('overlapped') == 'true'

    if not input_data or not input_type:
        return jsonify({'error': 'Missing input data or type'}), 400

    try:
        if input_type == 'url':
            count = main(input_data, input_type)
        elif input_type == 'image' and overlapped:
            count = main(input_data.read(), input_type, overlapped)
            count = count.tolist()
        else:
            count = main(input_data.read(), input_type)
        return jsonify(count), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

