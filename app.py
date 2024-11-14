from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
import psycopg2
import os
import numpy as np
from PIL import Image
import colorsys
from scipy.spatial import KDTree
from sklearn.preprocessing import LabelEncoder
import torch
from transformers import YolosForObjectDetection, AutoProcessor

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set a secret key for flash messages

# Database connection
def get_db_connection():
    conn = psycopg2.connect(
        host='localhost',
        database='clothing_db',  # Change this to your database name
        user=os.environ['DB_USERNAME'],
        password=os.environ['DB_PASSWORD']
    )
    return conn

def classify_image(image_path):
    # Load pre-trained model and processor
    model = YolosForObjectDetection.from_pretrained("valentinafeve/yolos-fashionpedia")
    processor = AutoProcessor.from_pretrained("valentinafeve/yolos-fashionpedia")

    # Open the image
    image = Image.open(image_path)

    # Preprocess the image and make predictions
    inputs = processor(images=image, return_tensors="pt", size={"shortest_edge": 800, "longest_edge": 800})

    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process the results
    results = processor.post_process_object_detection(outputs, target_sizes=[image.size[::-1]])[0]

    # Extract the bounding boxes, labels, and scores
    boxes = results['boxes']
    labels = results['labels']
    scores = results['scores']

    class_names = model.config.id2label

    top_classes = ['shirt', 'blouse', 'dress', 'top', 't-shirt', 'sweatshirt', 'sweater', 'cardigan', 'jacket', 'vest']
    bottom_classes = ['pants', 'shorts', 'skirt']

    detected = False

    for i in range(len(labels)):
        if detected:
            break

        label = labels[i].item()
        score = scores[i].item()

        if label in class_names:
            class_name = class_names[label]

            if any(top_class in class_name for top_class in top_classes):
                return("top")
                detected = True
            elif any(bottom_class in class_name for bottom_class in bottom_classes):
                return("bottom")
                detected = True
            elif class_name not in top_classes and class_name not in bottom_classes:
                if "sleeve" in class_name.lower():
                    return("top")
                    detected = True
# color_data structure
# color_data structure
color_data = {
    "black": {"matches": ["all"], "seasons": ["All Seasons"]},
    "ivory": {"matches": ["olive", "tan", "black"], "seasons": ["Spring", "Summer"]},
    "navy blue": {"matches": ["white", "black"], "seasons": ["Winter", "Autumn"]},
    "dusty pink": {"matches": ["navy blue", "ivory", "black"], "seasons": ["Spring", "Summer"]},
    "forest green": {"matches": ["beige", "black"], "seasons": ["Autumn"]},
    "coral": {"matches": ["tan", "ivory", "olive", "black"], "seasons": ["Spring"]},
    "olive": {"matches": ["tan", "black"], "seasons": ["Autumn", "Spring"]},
    "tan": {"matches": ["olive", "coral", "black"], "seasons": ["Autumn", "Spring"]},
    "red": {"matches": ["black", "white", "gray"], "seasons": ["Winter", "All Seasons"]},
    "yellow": {"matches": ["black", "gray"], "seasons": ["Spring", "Summer"]},
    "blue": {"matches": ["white", "black", "gray", "beige"], "seasons": ["All Seasons"]},
    "light blue": {"matches": ["white", "tan", "gray", "beige", "black"], "seasons": ["Spring", "Summer"]},
    "white": {"matches": ["all"], "seasons": ["All Seasons"]},
    "beige": {"matches": ["olive", "brown", "black"], "seasons": ["Spring", "Autumn"]},
    "gray": {"matches": ["black"], "seasons": ["Cooler Months"]},
    "brown": {"matches": ["tan", "black"], "seasons": ["Autumn", "Winter"]},
    "orange": {"matches": ["blue", "brown", "gray", "black"], "seasons": ["Autumn", "Spring"]}
}

# Define colors from color_data with their RGB values
COLOR_DATA_RGB = {
    "black": (0, 0, 0),
    "ivory": (255, 255, 240),
    "navy blue": (0, 0, 128),
    "dusty pink": (205, 92, 92),
    "forest green": (34, 139, 34),
    "coral": (255, 127, 80),
    "olive": (128, 128, 0),
    "tan": (210, 180, 140),
    "red": (255, 0, 0),
    "yellow": (255, 255, 0),
    "blue": (0, 0, 255),
    "light blue": (173, 216, 230),
    "white": (255, 255, 255),
    "beige": (245, 245, 220),
    "gray": (128, 128, 128),
    "brown": (139, 69, 19),
    "orange": (255, 165, 0)
}

# Initialize KDTree for color matching
color_names = list(COLOR_DATA_RGB.keys())
color_rgb_values = list(COLOR_DATA_RGB.values())
color_tree = KDTree(color_rgb_values)

def get_closest_color_name(rgb_tuple):
    _, index = color_tree.query(rgb_tuple)
    return color_names[index]

def get_dominant_color(image):
    max_score = 0.0001
    dominant_color = None
    for count, (r, g, b) in image.getcolors(image.size[0] * image.size[1]):
        saturation = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)[1]
        y = min(abs(r * 2104 + g * 4130 + b * 802 + 4096 + 131072) >> 13, 235)
        y = (y - 16.0) / (235 - 16)
        if y > 0.9:
            continue
        score = (saturation + 0.1) * count
        if score > max_score:
            max_score = score
            dominant_color = (r, g, b)
    return get_closest_color_name(dominant_color)

def color_classification(image_path):
    image = Image.open(image_path).convert('RGB')
    return get_dominant_color(image)

@app.route('/')
def index():
    return render_template('index.html')

# Code for uploading images
# Code for viewing images
@app.route('/upload_file', methods=['POST'])
def upload_file():
    image = request.files['image']
    clothing_type =classify_image(image)
    if image:
        # Save the image file
        image_filename = image.filename
        image_path = os.path.join('static/images', image_filename)
        image.save(image_path)

         # Open the image and resize it
        img = Image.open(image)
        img = img.resize((300, 300))  # Resize to 300x300 (adjust as needed)
        img.save(image_path)  # Save the resized image

        # Detect color and save it
        color_name = color_classification(image_path)

        # Insert into database
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('INSERT INTO clothing_images (clothing_type, image_filename, color_name) VALUES (%s, %s, %s)',
                    (clothing_type, image_filename, color_name))
        conn.commit()
        cur.close()
        conn.close()

        flash('Image uploaded successfully!', 'success')
    else:
        flash('No image uploaded. Please try again.', 'error')

    return redirect(url_for('index'))
@app.route('/view')
def view_images():
    clothing_type_filter = request.args.get('clothing_type')  # Get clothing type filter parameter
    color_filter = request.args.get('color_name')  # Get color filter parameter

    conn = get_db_connection()
    cur = conn.cursor()

    # Adjust query based on selected filters
    query = "SELECT image_filename, clothing_type, color_name FROM clothing_images WHERE 1=1"
    params = []

    if clothing_type_filter:
        query += " AND clothing_type = %s"
        params.append(clothing_type_filter)

    if color_filter:
        query += " AND color_name = %s"
        params.append(color_filter)

    cur.execute(query, tuple(params))
    images = cur.fetchall()

    # Fetch distinct color names
    cur.execute("SELECT DISTINCT color_name FROM clothing_images")
    color_names = [row[0] for row in cur.fetchall()]

    cur.close()
    conn.close()

    return render_template('view.html', images=images, clothing_type_filter=clothing_type_filter, color_filter=color_filter, color_names=color_names)

# Route for combinations
from flask import render_template, request

from collections import defaultdict

@app.route('/combinations')
def combinations():
    season_filter = request.args.get('season')  # Get the season filter from the URL query (if provided)
    color_filter = request.args.get('top_color')  # Get the color filter for the top (if provided)
    
    conn = get_db_connection()
    cur = conn.cursor()

    # Fetch tops and bottoms with color info
    cur.execute("SELECT image_filename, color_name FROM clothing_images WHERE clothing_type = 'top'")
    tops = cur.fetchall()
    
    cur.execute("SELECT image_filename, color_name FROM clothing_images WHERE clothing_type = 'bottom'")
    bottoms = cur.fetchall()

    cur.close()
    conn.close()

    # Filter tops based on the selected season
    filtered_tops = []
    if season_filter:
        for top_image, top_color in tops:
            top_seasons = color_data.get(top_color, {}).get("seasons", [])
            if season_filter in top_seasons or "All Seasons" in top_seasons:
                filtered_tops.append((top_image, top_color))
    else:
        # If no season filter, consider all tops
        filtered_tops = [(top_image, top_color) for top_image, top_color in tops]

    # Apply the color filter (if provided) to the filtered tops
    if color_filter:
        filtered_tops = [top for top in filtered_tops if top[1] == color_filter]

    # Group combinations by top color
    combinations_by_color = defaultdict(list)
    for top_image, top_color in filtered_tops:
        top_matches = color_data.get(top_color, {}).get("matches", [])
        for bottom_image, bottom_color in bottoms:
            if bottom_color in top_matches or "all" in top_matches:
                combinations_by_color[top_color].append((top_image, bottom_image))

    # Get the list of top colors for the selected season or all seasons
    top_colors = set()
    if season_filter == "All Seasons":
        # Show all colors that are available in any season or are marked as "All Seasons"
        for top_image, top_color in tops:
            top_seasons = color_data.get(top_color, {}).get("seasons", [])
            if "All Seasons" in top_seasons or any(season in top_seasons for season in ["Spring", "Summer", "Autumn", "Winter"]):
                top_colors.add(top_color)
    elif season_filter:
        # Show only colors available for the selected season
        for top_image, top_color in tops:
            top_seasons = color_data.get(top_color, {}).get("seasons", [])
            if season_filter in top_seasons or "All Seasons" in top_seasons:
                top_colors.add(top_color)
    else:
        # If no season filter, show all top colors
        top_colors = set([color for _, color in tops])

    return render_template(
        'combinations.html', 
        combinations_by_color=combinations_by_color,  # Pass grouped combinations
        season_filter=season_filter, 
        color_filter=color_filter, 
        top_colors=top_colors
    )


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


# Approve Combination Route
@app.route('/approve_combination', methods=['POST'])
def approve_combination():
    top = request.form.get('top')
    bottom = request.form.get('bottom')
    combination = (top, bottom)

    if 'approved_combinations' not in session:
        session['approved_combinations'] = []

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

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT likability_score FROM combination_scores
        WHERE top_image = %s AND bottom_image = %s
    """, (top, bottom))
    result = cur.fetchone()

    if result is None:
        new_score = 0.5 + score_change
        cur.execute("""
            INSERT INTO combination_scores (top_image, bottom_image, likability_score)
            VALUES (%s, %s, %s)
        """, (top, bottom, new_score))
    else:
        new_score = result[0] + score_change
        cur.execute("""
            UPDATE combination_scores
            SET likability_score = %s
            WHERE top_image = %s AND bottom_image = %s
        """, (new_score, top, bottom))

    conn.commit()
    cur.close()
    conn.close()

    return jsonify({'status': status, 'new_score': round(new_score, 2)})

# Reset Preferences Route
@app.route('/reset_preferences', methods=['POST'])
def reset_preferences():
    session.pop('approved_combinations', None)
    flash('Preferences have been reset.', 'success')
    return redirect(url_for('combinations'))

# Color Encoding and Similarity Calculation
colors = list(COLOR_DATA_RGB.keys())
color_encoder = LabelEncoder()
color_encoder.fit(colors)

def create_feature_vector(item):
    image_filename, color, likability_score = item
    color_encoded = color_encoder.transform([color])[0]
    feature_vector = np.array([color_encoded, likability_score])
    return feature_vector

def calculate_likability_scores(tops, bottoms):
    scores = {}
    for top in tops:
        for bottom in bottoms:
            top_vector = create_feature_vector(top)
            bottom_vector = create_feature_vector(bottom)
            similarity = cosine_similarity([top_vector], [bottom_vector])[0][0]
            scores[(top[0], bottom[0])] = similarity
    return scores

if __name__ == '__main__':
    app.run(debug=True)
