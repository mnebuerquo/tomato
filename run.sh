#!/bin/sh

. ./bin/activate

pip3 install -r ./requirements.txt

python3 tomato.py
