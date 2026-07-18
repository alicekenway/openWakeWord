# Installing openWakeWord with uv

openWakeWord requires Python 3.10 or newer. Python 3.11 is a good default for
a new environment.

## Install the published package

Create and activate an isolated environment, then install openWakeWord:

~~~bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install openwakeword
~~~

Download the pre-trained models once after installation:

~~~bash
python -c "import openwakeword; openwakeword.utils.download_models()"
~~~

## Install this checkout

From the repository root, install the local source in editable mode:

~~~bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
~~~

Install the optional development and training dependencies when needed:

~~~bash
uv pip install -e ".[full]"
~~~

The uv commands automatically use the local .venv directory after it has been
created. You can omit activation and invoke its interpreter directly:

~~~bash
.venv/bin/python -c "import openwakeword; print(openwakeword.__file__)"
~~~
