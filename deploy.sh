#!/bin/bash

# Deployment script for Swing Trading Analysis App

echo "🚀 Starting deployment process..."

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found. Please run this script from the project root."
    exit 1
fi

# Build the frontend
echo "📦 Building frontend..."
npm run build

if [ $? -eq 0 ]; then
    echo "✅ Frontend built successfully"
else
    echo "❌ Frontend build failed"
    exit 1
fi

# Activate virtual environment and install Python dependencies
echo "🐍 Activating virtual environment and installing Python dependencies..."
source venv/bin/activate
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Python dependencies installed successfully"
else
    echo "❌ Python dependencies installation failed"
    exit 1
fi

# Test the application locally
echo "🧪 Testing application locally..."
source venv/bin/activate
timeout 30s streamlit run app.py --server.headless true --server.port 8501 &
STREAMLIT_PID=$!

# Wait a bit for the app to start
sleep 10

# Check if the app is running
if curl -s http://localhost:8501 > /dev/null; then
    echo "✅ Application is running successfully"
    kill $STREAMLIT_PID
else
    echo "❌ Application failed to start"
    kill $STREAMLIT_PID
    exit 1
fi

echo "🎉 Deployment preparation completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Push your changes to GitHub:"
echo "   git add . && git commit -m 'Deploy bug fixes' && git push origin main"
echo ""
echo "2. Deploy to Streamlit Cloud:"
echo "   - Go to https://share.streamlit.io"
echo "   - Connect your GitHub repository"
echo "   - Set the main file path to: app.py"
echo "   - Deploy!"
echo ""
echo "3. Alternative: Deploy to other platforms:"
echo "   - Heroku: Use the Procfile and requirements.txt"
echo "   - Railway: Connect your GitHub repo"
echo "   - Render: Use the provided configuration"