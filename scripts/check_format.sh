#!/bin/sh

command -v autopep8 > /dev/null

if [ $? -ne 0 ]; then
    echo "autopep8 could not be found. Did you run this script outside virtual environment?"
    exit
fi

if [ "$( dirname "$0" )" != "./scripts" ]; then
    echo "This script should be executed from repo's root directory!"
    exit
fi

autopep8 --diff --exit-code *.py models/*.py lightning_data_modules/*.py lightning_modules/*.py utils/*.py