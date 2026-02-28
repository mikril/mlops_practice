#!/bin/bash
required_libs=("pandas" "numpy" "scikit-learn")

for lib_name in "${required_libs[@]}"; do
    if ! python3 -c "import $lib_name" &> /dev/null; then
        pip3 install $lib_name
    fi
done

python3 data_creation.py
python3 data_preprocessing.py
python3 model_preparation.py
python3 model_testing.py