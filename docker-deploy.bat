@echo off
echo ===================================================
echo Docker Deployment Script
echo Loan Prediction System
echo ===================================================
echo.

echo Checking if Docker is running...
docker --version
if %errorlevel% neq 0 (
    echo ERROR: Docker not found or not running!
    echo Please install Docker Desktop: https://www.docker.com/products/docker-desktop/
    pause
    exit /b 1
)

echo.
echo Step 1: Building Docker image...
docker build -t loan-prediction:latest .
if %errorlevel% neq 0 (
    echo ERROR: Docker build failed!
    pause
    exit /b 1
)

echo.
echo Step 2: Stopping any existing container...
docker stop loan-prediction-app 2>nul
docker rm loan-prediction-app 2>nul

echo.
echo Step 3: Starting container...
docker run -d -p 8080:8080 --name loan-prediction-app --restart unless-stopped loan-prediction:latest
if %errorlevel% neq 0 (
    echo ERROR: Failed to start container!
    pause
    exit /b 1
)

echo.
echo ===================================================
echo Deployment Complete!
echo ===================================================
echo.
echo Your application is running at: http://localhost:8080
echo.
echo Useful commands:
echo   docker logs -f loan-prediction-app    - View logs
echo   docker stop loan-prediction-app       - Stop container
echo   docker start loan-prediction-app      - Start container
echo   docker restart loan-prediction-app    - Restart container
echo.
echo Opening browser...
timeout /t 3 >nul
start http://localhost:8080

echo.
pause
