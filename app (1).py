from flask import Flask,render_template,redirect,session,flash,request,url_for
import joblib
model=joblib.load("C:/Users/vivek kumbhar/Desktop/failures in 3d printing/models/model.pkl")
le=joblib.load("C:/Users/vivek kumbhar/Desktop/failures in 3d printing/models/encoder.pkl")



app=Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict():

    if request.method == 'POST':

        features = [
            float(request.form['chamber_humidity_pct']),
            float(request.form['bed_temp_actual']),
            float(request.form['filament_diameter_mm']),
            float(request.form['vibration_y_g']),
            float(request.form['travel_speed_mm_s']),
            float(request.form['temp_variance']),
            float(request.form['retraction_distance_mm']),
            float(request.form['retraction_speed_mm_s']),
            float(request.form['fan_speed_pct'])
        ]

        prediction = model.predict([features])[0]
        ans=le.inverse_transform([prediction])

        return render_template(
            'index.html',
            ans=ans
        )

    return render_template('index.html')



if __name__=='__main__':  # running conditions
    app.run(debug=True)