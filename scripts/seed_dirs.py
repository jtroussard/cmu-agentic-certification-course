#!/usr/bin/env python3
"""
scripts/seed_dirs.py

Bootstrap the canonical Level 1 roots and initial Level 2+ directory tree
into the SQLite database (known_directories table).
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.database import get_connection, init_db, register_directory

# Canonical Level 1 Roots + common entity/category subdirectories
INITIAL_DIRECTORIES = [
    # L1 Roots
    ("01_Finance", "01_Finance"),
    ("01_Finance/Banking", "01_Finance"),
    ("01_Finance/Utilities", "01_Finance"),
    ("02_Home", "02_Home"),
    ("02_Home/45_Craighead_St_Pittsburgh_PA/Utilities", "02_Home"),
    ("02_Home/45_Craighead_St_Pittsburgh_PA/Utilities/Water", "02_Home"),
    ("02_Home/45_Craighead_St_Pittsburgh_PA/Maintenance", "02_Home"),
    ("02_Home/45_Craighead_St_Pittsburgh_PA/Repairs", "02_Home"),
    ("02_Home/45_Craighead_St_Pittsburgh_PA/Taxes", "02_Home"),
    ("02_Home/45_Craighead_St_Pittsburgh_PA/Insurance", "02_Home"),
    ("03_Vehicles", "03_Vehicles"),
    ("04_Pets", "04_Pets"),
    ("05_Legal", "05_Legal"),
    ("06_Employment", "06_Employment"),
    ("07_Insurance", "07_Insurance"),
    ("08_Travel", "08_Travel"),
    ("09_Personal", "09_Personal"),
    ("10_Rental_Property", "10_Rental_Property"),
    ("11_Taxes", "11_Taxes"),
    ("11_Taxes/2024", "11_Taxes"),
    ("90_Archive", "90_Archive"),
    ("99_Unsorted", "99_Unsorted"),
]


def seed(db_path: Path | None = None) -> None:
    init_db(db_path)
    now = datetime.now(timezone.utc).isoformat()
    with get_connection(db_path) as conn:
        for full_path, l1_root in INITIAL_DIRECTORIES:
            register_directory(conn, full_path=full_path, l1_root=l1_root, created_at=now)
            print(f"  [+] {full_path}")


if __name__ == "__main__":
    print("Bootstrapping known directories...")
    seed()
    print("Done.")
