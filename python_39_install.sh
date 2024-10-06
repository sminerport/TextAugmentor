#!/bin/bash

# Step 1: Update the package list
sudo apt-get update -y

# Step 2: Install Python 3.9 and necessary dependencies
sudo apt-get install python3.9 -y
sudo apt-get install python3.9-distutils -y

# Step 3: Download get-pip.py if not already present
if [ ! -f get-pip.py ]; then
    echo "Downloading get-pip.py..."
    wget https://bootstrap.pypa.io/get-pip.py
else
    echo "get-pip.py already exists. Skipping download."
fi

# Step 4: Install pip for Python 3.9
python3.9 get-pip.py

# Step 5: Run the setup.sh script to install the necessary packages
if [ -f setup.sh ]; then
    echo "Running setup.sh to install necessary packages..."
    bash setup.sh
else
    echo "setup.sh not found. Please ensure setup.sh is present to continue."
    exit 1
fi

# Step 6: Set Python 3.9 as the default Python version
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 2
sudo update-alternatives --set python3 /usr/bin/python3.9

# Step 7: Cleanup
rm get-pip.py

echo "Python 3.9 and necessary packages installed successfully."