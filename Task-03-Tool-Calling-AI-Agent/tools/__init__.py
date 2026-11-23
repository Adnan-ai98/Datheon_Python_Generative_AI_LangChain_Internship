from tools.calculator import calculator
from tools.file_reader import read_file
from tools.web_search import web_search

ALL_TOOLS = [web_search, calculator, read_file]
TOOL_MAP = {t.name: t for t in ALL_TOOLS}
