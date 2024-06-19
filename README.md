<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Engagement Dashboard</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        h1, h2, h3 {
            color: #333;
        }
        code {
            background: #f4f4f4;
            padding: 2px 5px;
            border-radius: 4px;
        }
        pre {
            background: #f4f4f4;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Student Engagement Dashboard</h1>
        <p>This Streamlit-based application analyzes student engagement using various input modes, including real-time webcam stream, uploaded photos, and videos. The analysis leverages facial emotion recognition and pose detection to classify student engagement states.</p>

        <h2>Features</h2>
        <ul>
            <li><strong>Real-Time Webcam Stream</strong>: Analyze student engagement live using the webcam.</li>
            <li><strong>Upload Student Photos</strong>: Upload a photo to get instant feedback on the engagement state.</li>
            <li><strong>Upload Student Videos</strong>: Analyze pre-recorded videos for engagement.</li>
            <li><strong>Engagement Metrics</strong>: Display key metrics such as attendance rate, participation, and homework submission.</li>
        </ul>

        <h2>Installation</h2>
        <ol>
            <li><strong>Clone the repository</strong>:
                <pre><code>git clone https://github.com/yourusername/student-engagement-dashboard.git
cd student-engagement-dashboard</code></pre>
            </li>
            <li><strong>Create and activate a virtual environment</strong>:
                <pre><code>python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`</code></pre>
            </li>
            <li><strong>Install the required dependencies</strong>:
                <pre><code>pip install -r requirements.txt</code></pre>
            </li>
        </ol>

        <h2>Usage</h2>
        <ol>
            <li><strong>Run the application</strong>:
                <pre><code>streamlit run app.py</code></pre>
            </li>
            <li><strong>Navigate to the application</strong>: Open your web browser and go to <code>http://localhost:8501</code>.</li>
        </ol>

        <h2>Application Structure</h2>
        <ul>
            <li><strong>app.py</strong>: Main application file that sets up the Streamlit dashboard and handles the various input modes for engagement analysis.</li>
            <li><strong>requirements.txt</strong>: Contains the list of dependencies required to run the application.</li>
        </ul>

        <h2>How It Works</h2>
        <h3>Real-Time Webcam Stream</h3>
        <p>The application captures frames from the webcam and processes each frame to detect facial emotions and pose landmarks. Engagement states are determined based on the detected emotions and poses, and results are displayed in real-time.</p>

        <h3>Upload Student Photos</h3>
        <p>Users can upload a photo, which is then processed to detect facial emotions and poses. The engagement state is displayed based on the analysis of the uploaded photo.</p>

        <h3>Upload Student Videos</h3>
        <p>Users can upload a video, which is analyzed frame-by-frame to detect facial emotions and poses. The application summarizes engagement states over time and displays the results in a readable format along with a line chart.</p>

        <h2>Engagement States</h2>
        <p>The application classifies the following engagement states:</p>
        <ul>
            <li>Distracted (mouth closed)</li>
            <li>Distracted (mouth open)</li>
            <li>Fatigue</li>
            <li>Focused (mouth closed)</li>
            <li>Focused (mouth open)</li>
            <li>Listening</li>
            <li>Raise hand</li>
            <li>Sleeping</li>
            <li>Using smartphone</li>
            <li>Writing/Reading</li>
        </ul>
    </div>
</body>
</html>
