from flask import Flask, request, jsonify
from process_input import main

app = Flask(__name__)

@app.route('/api/process', methods=['POST'])
def process():
    input_data = request.files.get('input') or request.form.get('input')
    input_type = request.form.get('inputType')

    if not input_data or not input_type:
        return jsonify({'error': 'Missing input data or type'}), 400

    try:
        if input_type == 'url':
            count = main(input_data, input_type)
        else:
            count = main(input_data.read(), input_type)
        return jsonify({'count': count})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

