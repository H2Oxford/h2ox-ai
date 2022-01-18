# https://stackoverflow.com/a/25389715/9940782
# https://stackoverflow.com/questions/2632199/how-do-i-get-the-path-of-the-current-executed-file-in-python
# https://www.delftstack.com/howto/python/python-get-path/
from pathlib import Path


ROOT_DIR = Path(__file__).parent.absolute()
