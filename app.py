
from flask import Flask, render_template, request, jsonify
import decision_tree
import config

app = Flask(__name__)


@app.route('/')
def home():
    return render_template( 
                         feature_columns=config.FEATURE_COLUMNS,
                         target_column=config.TARGET_COLUMN)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = {}
        for feature in config.FEATURE_COLUMNS:
            value = request.form.get(feature)
            if value is None or value == '':
                return jsonify({'error': f'Missing value for {feature}'}), 400
            input_data[feature] = float(value)

        dt_prediction = decision_tree.predict(input_data)
        
        # Prepare response
        results = {
            'input': input_data,
            'predictions': {
                'Decision Tree': round(dt_prediction, 2),
            },
            'target': config.TARGET_COLUMN
        }
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/train', methods=['POST'])
def train_models():
    try:
        decision_tree.train_model()
        return jsonify({'message': 'All models trained successfully!'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Train models on startup if they don't exist
    import os
    if not os.path.exists('models'):
        print("Training models for the first time...")
        decision_tree.train_model()
    
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=config.DEBUG_MODE)
