# Commands to run from root folder of this project for a correct seutup

"only first time?:"
git submodule add https://github.com/CMU-MultiComp-Lab/OpenFace-3.0.git OpenFace-3.0

git submodule update --init --recursive

"fails due to OpenFace-3.0 having the submodules but not the .gitmodules file"

"add .gitmodule manually available in setup-info folder and repeat previous command"

python3 -m venv .venv

source .venv/bin/activate

find . -name "requirements.txt" -exec pip install -r {} \;

pip install openface-test

openface download --output aux

"move all files from aux to Openface-3.0/weights without overwritting old ones if named the same, and delete aux folder"