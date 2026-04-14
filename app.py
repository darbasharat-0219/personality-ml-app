from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# =========================
# CLASS FOR MODEL
# =========================
class PersonalityModel:

    def __init__(self):
        # Load model and scaler
        self.model = pickle.load(open('personality_model.pkl', 'rb'))
        self.scaler = pickle.load(open('scaler.pkl', 'rb'))

    def predict(self, data):
        # Convert to numpy array
        data = np.array([data])

        # Apply scaling
        scaled_data = self.scaler.transform(data)

        # Predict
        prediction = self.model.predict(scaled_data)[0]

        # Convert output
        if prediction == 0:
            return "Introvert"
        elif prediction == 1:
            return "Ambivert"
        else:
            return "Extrovert"


# Create object
model_obj = PersonalityModel()


# =========================
# ROUTES
# =========================
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    try:
        # Collect data (IMPORTANT: SAME ORDER)
        data = [
            float(request.form['social_energy']),
            float(request.form['alone_time_preference']),
            float(request.form['talkativeness']),
            float(request.form['deep_reflection']),
            float(request.form['group_comfort']),
            float(request.form['party_liking']),
            float(request.form['listening_skill']),
            float(request.form['empathy']),
            float(request.form['organization']),
            float(request.form['leadership']),
            float(request.form['risk_taking']),
            float(request.form['public_speaking_comfort']),
            float(request.form['curiosity']),
            float(request.form['routine_preference']),
            float(request.form['excitement_seeking']),
            float(request.form['friendliness']),
            float(request.form['planning']),
            float(request.form['spontaneity']),
            float(request.form['adventurousness']),
            float(request.form['reading_habit']),
            float(request.form['sports_interest']),
            float(request.form['online_social_usage']),
            float(request.form['travel_desire']),
            float(request.form['gadget_usage']),
            float(request.form['work_style_collaborative']),
            float(request.form['decision_speed'])
        ]

        # Get prediction from class
        result = model_obj.predict(data)

        return render_template('index.html', prediction=result)

    except Exception as e:
        return str(e)


# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    app.run(debug=True)