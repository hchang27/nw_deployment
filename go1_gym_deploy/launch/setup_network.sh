# Check for an interface argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <interface>"
    exit 1
fi

# Assign the first argument to a variable
INTERFACE=$1

# Bringing down the interface
sudo ifconfig $INTERFACE down

# Configuring the interface with a static IP address
sudo ifconfig $INTERFACE 192.168.123.162/24

# Bringing up the interface
sudo ifconfig $INTERFACE up

# Adding a route
sudo ip route add 239.255.76.67 dev $INTERFACE

sudo ip route add 192.168.123.0/24 dev $INTERFACE

echo "Configuration applied to $INTERFACE"
