"""Write a synthetic workout history to .local/history.parquet for the local demo."""
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))

from factories import make_history  # noqa: E402

out = ROOT / '.local' / 'history.parquet'
out.parent.mkdir(exist_ok=True)
history = make_history(weeks=40, end=date.today() - timedelta(days=1))
history.to_parquet(out, index=False)
print(f'wrote {len(history)} synthetic sets to {out}')
