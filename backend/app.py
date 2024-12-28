from flask import Flask, request, jsonify, Response, stream_with_context
from process_input import main
from flask_cors import CORS
import traceback

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
        if input_type in ['video', 'url']:
            result = main(input_data, input_type)
            response = Response(
                stream_with_context(result),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )
            response.headers['Cache-Control'] = 'no-cache'
            response.headers['X-Accel-Buffering'] = 'no'
            return response
        elif input_type == 'url':
            # For video URL streams
            result = main(input_data, input_type)
            return Response(
                stream_with_context(result),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )
        elif input_type == 'image':
            # Image processing remains unchanged
            if overlapped:
                count = main(input_data.read(), input_type, overlapped)
                count = count.tolist()
            else:
                count = main(input_data.read(), input_type)
            return jsonify(count), 200
        
        else:
            return jsonify({'error': 'Invalid input type'}), 400
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

