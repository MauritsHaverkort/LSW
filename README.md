For installing dependencies in a new virtual environment, you can use the following commands:

python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

Afterwards you can start the calculations by running:

python LSW.py

Runtime parameters can be be changed via the config.json file, in that case run:
python LSW.py --config config.json

Alterinatively, alter Configurations.py to change the parameters of the calculations.
Furthermore, in Examples.py a variety of example material input files are provided. 

You can specify which material is used in LSW.py by specifying in "call_main_with_material_config" the material e.g. 
"Examples.K2NiF4(config)" and the configuration e.g. "Configurations.config_experimental("main")"