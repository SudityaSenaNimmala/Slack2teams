#!/bin/bash

# Slack2Teams Chatbot Deployment Script for DigitalOcean

echo "🚀 Starting Slack2Teams Chatbot Deployment..."

# Update system packages
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Docker
echo "🐳 Installing Docker..."
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io
sudo systemctl start docker
sudo systemctl enable docker

# Install Docker Compose
echo "🔧 Installing Docker Compose..."
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install Git
echo "📥 Installing Git..."
sudo apt install -y git

# Clone repository
echo "📂 Cloning repository..."
git clone https://github.com/SudityaSenaNimmala/Slack2teams.git
cd Slack2teams

# Create environment file
echo "⚙️ Creating environment configuration..."
cat > .env << EOF
OPENAI_API_KEY=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
MICROSOFT_CLIENT_ID=63bd4522-368b-4bd7-a84d-9c7f205cd2a6
MICROSOFT_CLIENT_SECRET=g-C8Q~ZCcJmuHOt~wEJinGVfiZGYd9gzEy6Wfb5Y
MICROSOFT_TENANT=common
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com
EOF

# Build and start the application
echo "🏗️ Building and starting the application..."
sudo docker-compose up -d --build

# Check if the application is running
echo "🔍 Checking application status..."
sleep 10
if curl -f http://localhost:8002/health; then
    echo "✅ Application is running successfully!"
    echo "🌐 Your chatbot is available at: http://$(curl -s ifconfig.me):8002"
else
    echo "❌ Application failed to start. Checking logs..."
    sudo docker-compose logs
fi

echo "🎉 Deployment completed!"
