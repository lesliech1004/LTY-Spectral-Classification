from pathlib import Path
import sys

from astrodb_utils import load_astrodb
from simple import REFERENCE_TABLES


# Path to the SIMPLE-db repo (where SIMPLE.sqlite and data/ live)
SIMPLE_DB_DIR = Path.home() / "Documents" / "GitHub" / "SIMPLE-db"

DB_PATH = SIMPLE_DB_DIR / "SIMPLE.sqlite"
SCHEMA_PATH = SIMPLE_DB_DIR / "simple" / "schema.yaml"


def get_db(recreatedb: bool = False):
    db = load_astrodb(
        DB_PATH.as_posix(),
        recreatedb=recreatedb,
        reference_tables=REFERENCE_TABLES,
        felis_schema=SCHEMA_PATH.as_posix(),
    )
    return db
