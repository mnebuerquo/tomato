#!/bin/sh

echo "Activate virtualenv..."
if [ ! -d "./env" ]; then
	echo "Creating env..."
	virtualenv -p python3 "env"
fi
# then activate virtualenv
. "./env/bin/activate"
echo "Env activated."

./env/bin/pip3 install -r ./requirements.txt

./env/bin/python3 tomato.py
