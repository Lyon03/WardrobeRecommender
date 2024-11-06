from flask import Flask, render_template, request, redirect, url_for, flash
import psycopg2
import os

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_file', methods=['POST'])
def upload_file():
    clothing_type = request.form['clothing_type']
    image = request.files['image']

    if image:
        # Save the image file
        image_filename = image.filename
        image.save(os.path.join('static/images', image_filename))

        # Insert into database
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('INSERT INTO clothing_images (clothing_type, image_filename) VALUES (%s, %s)',
                    (clothing_type, image_filename))
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
    cur.execute("SELECT image_filename, clothing_type FROM clothing_images")  # Adjust this query as necessary
    images = cur.fetchall()
    cur.close()
    conn.close()

    # Debugging: Print out paths to check them
    for image, clothing_type in images:
        print(f"Image Path: {os.path.join('static/images', image)}")

    return render_template('view.html', images=images)

@app.route('/combinations')
def combinations():
    conn = get_db_connection()
    cur = conn.cursor()

    # Fetch tops and bottoms from the database
    cur.execute("SELECT image_filename FROM clothing_images WHERE clothing_type = 'top'")
    tops = cur.fetchall()
    
    cur.execute("SELECT image_filename FROM clothing_images WHERE clothing_type = 'bottom'")
    bottoms = cur.fetchall()

    cur.close()
    conn.close()

    # Create combinations (all possible pairs of tops and bottoms)
    combinations = [(top[0], bottom[0]) for top in tops for bottom in bottoms]
    
    return render_template('combinations.html', combinations=combinations)

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

if __name__ == '__main__':
    app.run(debug=True)
