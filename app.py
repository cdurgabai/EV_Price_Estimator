from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, session, flash
from functools import wraps
import joblib
import pandas as pd
import os
from io import BytesIO
import io

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a secure secret key

# Login decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Load dataset
df = pd.read_csv(os.path.join('data', 'train.csv'))
df.drop(columns=[
    'ID', 'VIN (1-10)', 'DOL Vehicle ID', 'Vehicle Location',
    'County', 'State', 'ZIP Code'
], inplace=True, errors='ignore')
df = df[pd.to_numeric(df['Expected Price ($1k)'], errors='coerce').notna()]

# Dropdown values for the form
dropdowns = {
    'makes': sorted(df['Make'].dropna().unique()),
    'ev_types': sorted(df['Electric Vehicle Type'].dropna().unique()),
    'cafv_options': sorted(df['Clean Alternative Fuel Vehicle (CAFV) Eligibility'].dropna().unique()),
    'cities': sorted(df['City'].dropna().unique()),
    'utilities': sorted(df['Electric Utility'].dropna().unique())
}

numerical_info = {
    'model_years': sorted(df['Model Year'].dropna().unique()),
    'range_min': int(df['Electric Range'].min()),
    'range_max': int(df['Electric Range'].max()),
    'msrp_min': int(df['Base MSRP'].min()),
    'msrp_max': int(df['Base MSRP'].max())
}

# Dummy users - In production, use a proper database
users = {"testuser": "password123"}

@app.route('/')
@app.route('/home')
def home():
    """Home page route"""
    return render_template('index.html')

@app.route('/estimate')
@login_required
def estimate():
    """Estimation form page route"""
    return render_template('estimate.html', **dropdowns, **numerical_info)

@app.route('/get_models', methods=['POST'])
@login_required
def get_models():
    """AJAX endpoint to get models for selected make"""
    make = request.json.get('make')
    if not make:
        return jsonify([])
    models = sorted(df[df['Make'] == make]['Model'].dropna().unique())
    return jsonify(models)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    """Handle form submission and make prediction"""
    if request.method == 'POST':
        try:
            form_data = {
                'make': request.form.get('make'),
                'model': request.form.get('model'),
                'model_year': float(request.form.get('model_year')),
                'electric_range': float(request.form.get('electric_range')),
                'base_msrp': float(request.form.get('base_msrp')),
                'ev_type': request.form.get('ev_type'),
                'cafv': request.form.get('cafv'),
                'city': request.form.get('city'),
                'utility': request.form.get('utility')
            }

            algorithm = request.form.get('algorithm')

            if not all(form_data.values()) or not algorithm:
                flash('Please fill in all fields.', 'danger')
                return redirect(url_for('estimate'))

            # Load the selected model
            model_map = {
                'random_forest': 'randomforest_ev_price_model.pkl',
                'gradient_boosting': 'gradientboosting_ev_price_model.pkl'
            }

            model_file = model_map.get(algorithm)
            if not model_file or not os.path.exists(model_file):
                flash('Model not found or not trained.', 'danger')
                return redirect(url_for('estimate'))

            model = joblib.load(model_file)

            model_input = pd.DataFrame([{
    'Make': form_data['make'],
    'Model': form_data['model'],
    'Model Year': form_data['model_year'],
    'Electric Range': form_data['electric_range'],
    'Base MSRP': form_data['base_msrp'],
    'Electric Vehicle Type': form_data['ev_type'],
    'Clean Alternative Fuel Vehicle (CAFV) Eligibility': form_data['cafv'],
    'City': form_data['city'],
    'Electric Utility': form_data['utility']
}])


            # Prepare input DataFrame
            # input_df = pd.DataFrame([form_data])

            # Predict price
            predicted_price = model.predict(model_input)[0] * 1000

            # Store the results
            price=int(predicted_price)
            session['prediction'] = f"Estimated Price: ${price}"
            session['form_data'] = form_data

            flash('Price estimation completed successfully!', 'success')
            return redirect(url_for('results'))

        except Exception as e:
            flash(f'An error occurred during prediction: {str(e)}', 'danger')
            return redirect(url_for('estimate'))

@app.route('/results')
@login_required
def results():
    """Results page route"""
    if 'prediction' not in session or 'form_data' not in session:
        flash('No estimation results available. Please complete the form first.', 'info')
        return redirect(url_for('estimate'))
        
    return render_template('results.html',
                         prediction_text=session.get('prediction'),
                         form_data=session.get('form_data'))

@app.route('/download_csv')
@login_required
def download_csv():
    """Download results as CSV"""
    if 'form_data' not in session or 'prediction' not in session:
        flash('No data available for download. Please complete the estimation first.', 'warning')
        return redirect(url_for('estimate'))
    
    try:
        data = {
            'Parameter': ['Make', 'Model', 'Year', 'Electric Range', 'Base MSRP', 'EV Type', 'CAFV Eligibility', 'City', 'Utility', 'Predicted Price'],
            'Value': [
                session['form_data']['make'],
                session['form_data']['model'],
                session['form_data']['model_year'],
                f"{session['form_data']['electric_range']} miles",
                f"${session['form_data']['base_msrp']}",
                session['form_data']['ev_type'],
                session['form_data']['cafv'],
                session['form_data']['city'],
                session['form_data']['utility'],
                session['prediction']
            ]
        }
        df_results = pd.DataFrame(data)
        
        output = io.StringIO()
        df_results.to_csv(output, index=False)
        
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name='ev_price_estimation.csv'
        )
    except Exception as e:
        flash(f'Error generating CSV: {str(e)}', 'danger')
        return redirect(url_for('results'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page route"""
    if 'username' in session:
        return redirect(url_for('home'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            flash('Please provide both username and password.', 'danger')
            return redirect(url_for('login'))
            
        if users.get(username) == password:
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password.', 'danger')
            
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Register page route"""
    if 'username' in session:
        return redirect(url_for('home'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        email = request.form.get('email')
        
        if not all([username, password, email]):
            flash('Please fill in all fields.', 'danger')
            return redirect(url_for('register'))
            
        if username in users:
            flash('Username already exists.', 'danger')
        else:
            users[username] = password
            flash('Registration successful! You can now log in.', 'success')
            return redirect(url_for('login'))
            
    return render_template('register.html')

@app.route('/logout')
def logout():
    """Logout route"""
    session.clear()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
