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
    cur.execute("SELECT image_filename FROM clothing_images")  # Adjust this query as necessary
    images = cur.fetchall()
    cur.close()
    conn.close()
    
    # Flatten the list of tuples
    images = [image[0] for image in images]
    
    return render_template('view.html', images=images)


if __name__ == '__main__':
    app.run(debug=True)
