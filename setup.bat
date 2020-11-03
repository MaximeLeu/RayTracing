@echo off

:check_conda_install
echo Trying to install via conda...
call conda install -y --file requirements.txt --json > .packages.json
echo Saved installation logs in .packages.json

:split_installs
echo Installing remaining packages with pip...
python -c "import json; f = open('.packages.json', 'r'); d = json.load(f); req = open('requirements.txt', 'r'); reqs = req.read().splitlines(); conda = open('.conda_packages.txt', 'w'); conda.write('\n'.join((r for r in reqs if r not in d['packages']))); pip = open('.pip_packages.txt', 'w'); pip.write('\n'.join(d['packages'])); f.close(); req.close(); conda.close(); pip.close();"

:conda_install
echo If first install failed, installing found packages via conda...
call conda install -y --file .conda_packages.txt
del .conda_packages.txt

:pip_install
echo Installing invalid not found packages via pip...
pip install -r .pip_packages.txt
del .pip_packages.txt

:setup_project
echo Setup project...
pip install -e .
echo Setup done!