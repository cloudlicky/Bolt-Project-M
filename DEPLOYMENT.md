# Deployment Guide for Swing Trading Analysis App

## 🚀 Deployment Options

This application can be deployed on multiple platforms. Choose the option that best fits your needs:

### 1. Streamlit Cloud (Recommended - Free)

**Steps:**
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository: `cloudlicky/Bolt-Project-M`
5. Set the main file path: `app.py`
6. Click "Deploy"

**Benefits:**
- Free hosting
- Automatic deployments on Git push
- Built-in analytics
- Easy setup

### 2. Heroku

**Steps:**
1. Install Heroku CLI
2. Login to Heroku: `heroku login`
3. Create a new app: `heroku create your-app-name`
4. Deploy: `git push heroku main`
5. Open the app: `heroku open`

**Note:** The `Procfile` is already configured for Heroku deployment.

### 3. Render

**Steps:**
1. Go to [render.com](https://render.com)
2. Connect your GitHub repository
3. Create a new Web Service
4. Use the `render.yaml` configuration (already included)
5. Deploy automatically

### 4. Railway

**Steps:**
1. Go to [railway.app](https://railway.app)
2. Connect your GitHub repository
3. Deploy automatically

## 🔧 Local Development Setup

```bash
# Clone the repository
git clone https://github.com/cloudlicky/Bolt-Project-M.git
cd Bolt-Project-M

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install frontend dependencies
npm install

# Build frontend
npm run build

# Run the application
streamlit run app.py
```

## 📋 Environment Variables

The following environment variables can be set for production:

- `STREAMLIT_SERVER_PORT`: Port number (default: 8501)
- `STREAMLIT_SERVER_ADDRESS`: Server address (default: 0.0.0.0)
- `STREAMLIT_SERVER_HEADLESS`: Run in headless mode (default: true)

## 🐛 Troubleshooting

### Common Issues:

1. **TA-Lib Installation Issues:**
   - The TA-Lib dependency has been commented out in requirements.txt
   - If you need technical indicators, install TA-Lib manually:
     ```bash
     # Ubuntu/Debian
     sudo apt install libta-lib-dev
     pip install TA-Lib
     
     # macOS
     brew install ta-lib
     pip install TA-Lib
     ```

2. **Port Already in Use:**
   - Change the port in the deployment command
   - Kill existing processes: `pkill -f streamlit`

3. **Memory Issues:**
   - Increase memory allocation for your deployment platform
   - Optimize data processing in the application

## 📊 Monitoring

- **Streamlit Cloud:** Built-in analytics dashboard
- **Heroku:** Use `heroku logs --tail`
- **Render:** Built-in logging and monitoring
- **Railway:** Built-in logging and metrics

## 🔄 Continuous Deployment

The application is configured for automatic deployment:
- Push to `main` branch triggers deployment
- All deployment configurations are included in the repository
- No manual intervention required after initial setup

## 📞 Support

For deployment issues:
1. Check the logs in your deployment platform
2. Verify all dependencies are installed
3. Ensure environment variables are set correctly
4. Test locally before deploying

---

**Last Updated:** $(date)
**Version:** 1.0.0