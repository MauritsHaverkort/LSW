For installing dependencies in a new virtual environment, you can use the following commands:

python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

Afterwards you can start the calculations by running:

python LSW.py

Alter Configurations.py to change the settings of the calculations.

In Examples.py a variety of example material input files are provided. You can specify which material is used in LSW.py by changing 