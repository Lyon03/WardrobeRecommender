from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import psycopg2
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set a secret key for flash messages

# Database connection
def get_db_connection():
    conn = psycopg2.connect(
        host='localhost',
        database='clothing_db',  # Update to your database name
        user=os.environ['DB_USERNAME'],
        password=os.environ['DB_PASSWORD']
    )
    return conn

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_file', methods=['POST'])
def upload_file():
    category = request.form['clothing_type']  # Category: top or bottom
    image = request.files['image']
    color = request.form['color']

    if image:
        image_filename = image.filename
        image.save(os.path.join('static/images', image_filename))

        # Insert into database with a default likability score
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('INSERT INTO clothing_images (category, image_filename, color, likability_score) VALUES (%s, %s, %s, %s)',
                    (category, image_filename, color, 0.5))  # Default score of 0.5
        conn.commit()
        cur.close()
        conn.close()

        flash('Image uploaded successfully!', 'success')
    else:
        flash('No image uploaded. Please try again.', 'error')

    return redirect(url_for('index'))

@app.route('/view')
def view_images():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT image_filename, category FROM clothing_images")
    images = cur.fetchall()
    cur.close()
    conn.close()

    return render_template('view.html', images=images)

@app.route('/combinations')
def combinations():
    conn = get_db_connection()
    cur = conn.cursor()

    # Fetch tops and bottoms from the database
    cur.execute("SELECT image_filename, color, likability_score FROM clothing_images WHERE category = 'top'")
    tops = cur.fetchall()

    cur.execute("SELECT image_filename, color, likability_score FROM clothing_images WHERE category = 'bottom'")
    bottoms = cur.fetchall()

    cur.close()
    conn.close()

    # Create all possible pairs of tops and bottoms
    combinations = [(top[0], bottom[0]) for top in tops for bottom in bottoms]

    # Get scores from the database for each combination
    scores = calculate_likability_scores(tops, bottoms)

    # Get approved combinations from the session
    approved_combinations = session.get('approved_combinations', [])

    return render_template('combinations.html', combinations=combinations, scores=scores, approved_combinations=approved_combinations)


@app.route('/approve_combination', methods=['POST'])
def approve_combination():
    top = request.form.get('top')
    bottom = request.form.get('bottom')
    combination = (top, bottom)

    # Initialize the approved_combinations in session if not present
    if 'approved_combinations' not in session:
        session['approved_combinations'] = []

    # Toggle approval status
    if combination in session['approved_combinations']:
        session['approved_combinations'].remove(combination)
        session.modified = True
        status = 'removed'
        score_change = -0.1
    else:
        session['approved_combinations'].append(combination)
        session.modified = True
        status = 'approved'
        score_change = 0.1

    # Connect to the database
    conn = get_db_connection()
    cur = conn.cursor()

    # Check if the combination already exists in the database
    cur.execute("""
        SELECT likability_score FROM combination_scores
        WHERE top_image = %s AND bottom_image = %s
    """, (top, bottom))
    result = cur.fetchone()

    # If the combination does not exist, insert it with a default score of 0.5
    if result is None:
        new_score = 0.5 + score_change  # Apply the score change immediately
        cur.execute("""
            INSERT INTO combination_scores (top_image, bottom_image, likability_score)
            VALUES (%s, %s, %s)
        """, (top, bottom, new_score))
    else:
        # If it exists, update the existing score
        new_score = result[0] + score_change
        cur.execute("""
            UPDATE combination_scores
            SET likability_score = %s
            WHERE top_image = %s AND bottom_image = %s
        """, (new_score, top, bottom))

    conn.commit()
    cur.close()
    conn.close()

    # Return JSON with updated score and status
    return jsonify({'status': status, 'new_score': round(new_score, 2)})


@app.route('/delete/<image_filename>', methods=['POST'])
def delete_image(image_filename):
    # Delete the image file from the 'static/images' folder
    image_path = os.path.join('static/images', image_filename)
    if os.path.exists(image_path):
        os.remove(image_path)

    # Remove the image entry from the database
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM clothing_images WHERE image_filename = %s", (image_filename,))
    conn.commit()
    cur.close()
    conn.close()

    flash(f'Image "{image_filename}" deleted successfully!', 'success')

    return redirect(url_for('view_images'))

@app.route('/reset_preferences', methods=['POST'])
def reset_preferences():
    # Clear approved combinations from the session
    session.pop('approved_combinations', None)
    flash('Preferences have been reset.', 'success')
    return redirect(url_for('combinations'))




# Define the colors and initialize a LabelEncoder to convert colors to numerical features
colors = ['black', 'blue', 'brown', 'green', 'purple', 'red', 'orange']  # Add all colors used in your app
color_encoder = LabelEncoder()
color_encoder.fit(colors)

def calculate_likability_scores(tops, bottoms):
    # Create dictionary for storing likability scores
    scores = {}

    for top in tops:
        for bottom in bottoms:
            # Generate feature vectors for top and bottom items
            top_vector = create_feature_vector(top)
            bottom_vector = create_feature_vector(bottom)
            
            # Calculate cosine similarity between top and bottom feature vectors
            similarity = cosine_similarity([top_vector], [bottom_vector])[0][0]

            # Store the similarity as the likability score, scaling if necessary (e.g., similarity score can range from 0-1)
            scores[(top[0], bottom[0])] = similarity

    return scores

def create_feature_vector(item):
    #Define the colors, adding "unknown" explicitly if needed
    colors = ['black', 'blue', 'brown', 'green', 'purple', 'red', 'orange', 'unknown']
    color_encoder = LabelEncoder()
    color_encoder.fit(colors)


    image_filename, color, likability_score = item

    # Ensure color is in the list of known classes
    if color not in color_encoder.classes_:
        color = "black"  # Assign a default known color if color is unknown or not in list

    # Convert color to a numerical value using the label encoder
    color_encoded = color_encoder.transform([color])[0]

    feature_vector = np.array([color_encoded, likability_score])
    return feature_vector

if __name__ == '__main__':
    app.run(debug=True)
