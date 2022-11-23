#!/bin/sh

command -v pylint > /dev/null

if [ $? -ne 0 ]; then
    echo "pylint could not be found. Did you run this script outside virtual environment?"
    exit
fi

if [ "$( dirname "$0" )" != "./scripts" ]; then
    echo "This script should be executed from repo's root directory!"
    exit
fi

pylint *.py