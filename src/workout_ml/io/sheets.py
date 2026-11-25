"""Workout history in, predictions out: Google Sheets or local files."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pandas as pd

SCOPES = ['https://www.googleapis.com/auth/spreadsheets']


class SheetsGateway(Protocol):
    def read_history(self) -> pd.DataFrame: ...
    def write_predictions(self, header: list[str], rows: list[list]) -> None: ...


class GoogleSheets:
    def __init__(self, spreadsheet_id: str, source_tab: str, output_tab: str, credentials_path: str = '',
                 client=None):
        if source_tab.strip().lower() == output_tab.strip().lower():
            raise ValueError('refusing to write predictions to the source tab')
        self._id, self._source, self._output = spreadsheet_id, source_tab, output_tab
        self._credentials_path = credentials_path
        self._client = client

    def _book(self):
        if self._client is None:
            import gspread
            if self._credentials_path:
                self._client = gspread.service_account(filename=self._credentials_path, scopes=SCOPES)
            else:
                import google.auth
                creds, _ = google.auth.default(scopes=SCOPES)
                self._client = gspread.authorize(creds)
        return self._client.open_by_key(self._id)

    def read_history(self):
        df = pd.DataFrame(self._book().worksheet(self._source).get_all_records())
        df.columns = [str(c).lower() for c in df.columns]
        return df

    def write_predictions(self, header, rows):
        import gspread
        book = self._book()
        n_cols = len(header)
        try:
            ws = book.worksheet(self._output)
        except gspread.WorksheetNotFound:
            ws = book.add_worksheet(self._output, rows=max(100, len(rows) + 1), cols=n_cols)
        if ws.row_count < len(rows) + 1 or ws.col_count < n_cols:
            ws.resize(rows=max(ws.row_count, len(rows) + 1), cols=max(ws.col_count, n_cols))
        # One update that also blanks leftover rows, so the tab is never half old and half new.
        values = [header, *rows]
        values += [[''] * n_cols] * (ws.row_count - len(values))
        ws.update(values, 'A1', value_input_option='RAW')


class LocalSheets:
    """History from a .parquet/.csv file, predictions to ``predictions.csv``."""

    def __init__(self, history: str | Path, out_dir: Path):
        self._history, self._out = Path(history), Path(out_dir)

    def read_history(self):
        if self._history.suffix == '.parquet':
            return pd.read_parquet(self._history)
        return pd.read_csv(self._history)

    def write_predictions(self, header, rows):
        self._out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=header).to_csv(self._out / 'predictions.csv', index=False)
