#!/bin/bash
set -e

echo "🔑 Setting up SSH for Lightning AI..."

# Step 1: Check/Generate SSH Keys
if [ ! -f ~/.ssh/id_rsa ]; then
    echo "   Generating SSH key pair..."
    ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ""
else
    echo "   ✅ SSH keys found."
fi

# Step 2: Display Public Key
echo ""
echo "📋 Step 2: Add this public key to your Lightning AI Account:"
echo "   Go to: https://lightning.ai/settings (SSH Keys section)"
echo "   Copy the following block:"
echo "----------------------------------------------------------------------"
cat ~/.ssh/id_rsa.pub
echo "----------------------------------------------------------------------"
echo ""
read -p "   Press Enter once you have added the key to Lightning AI..."

# Step 3: Configure SSH Config
echo ""
echo "⚙️  Step 3: Configuring SSH Client..."
read -p "   Enter your Studio ID (e.g., s_01kgc...): " STUDIO_ID

if [ -z "$STUDIO_ID" ]; then
    echo "   ❌ Studio ID is required."
    exit 1
fi

CONFIG_FILE=~/.ssh/config
touch $CONFIG_FILE

if grep -q "Host ssh.lightning.ai" $CONFIG_FILE; then
    echo "   ⚠️  Config for ssh.lightning.ai already exists in $CONFIG_FILE."
    echo "   Please verify it uses User $STUDIO_ID"
else
    echo "   Appending configuration to $CONFIG_FILE..."
    cat <<EOT >> $CONFIG_FILE

Host ssh.lightning.ai
  User $STUDIO_ID
  IdentityFile ~/.ssh/id_rsa
  StrictHostKeyChecking no
  UserKnownHostsFile /dev/null
EOT
    echo "   ✅ Configuration added."
fi

echo ""
echo "🚀 Setup Complete! You can now connect via VS Code / Cursor."