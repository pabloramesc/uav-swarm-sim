set -e

# Install required packages
sudo apt update
sudo apt install -y build-essential cmake clang ninja-build cmake-format libgsl-dev libgtk-3-dev libeigen3-dev libc-dev python3-dev python3-pip

# Clone ns-3
git clone https://gitlab.com/nsnam/ns-3-dev.git ns-3
cd ns-3

# Configure and build ns-3 with additional features
./ns3 clean
./ns3 configure --enable-examples --enable-tests
./ns3 build

# Run tests (optional)
./test.py

# Copy simulation code
cd ..
cp -r ./swarm-net-sim ./ns-3/scratch/swarm-net-sim