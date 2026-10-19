"""
loading.py
==========
Load SocioPatterns contact-sequence datasets into a common DataFrame format.

Every dataset is returned as:
  contacts : pd.DataFrame  columns = [t (int, seconds), i (int), j (int)]
             node ids are remapped to 0 .. n-1 (contiguous integers)
  labels   : np.ndarray of shape (n,) with an integer community id per node,
             or None when the dataset carries no community information.

Supported dataset names
-----------------------
  'primaryschool', 'hospital',
  'hs2011', 'hs2012', 'hs2013',
  'conference',   # SFHH conference
  'workplace',
  'malawi'
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import DATA_DIR, T_RES_SECONDS

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_contacts(dataset_name: str) -> tuple[pd.DataFrame, np.ndarray | None]:
    """Load a SocioPatterns dataset by name.

    Parameters
    ----------
    dataset_name:
        One of 'primaryschool', 'hospital', 'hs2011', 'hs2012', 'hs2013',
        'conference', 'workplace', 'malawi'.

    Returns
    -------
    contacts:
        DataFrame with integer columns ``t`` (seconds), ``i``, ``j``.
        Node ids are contiguous integers 0 .. n-1.
    labels:
        1-D numpy array of length n: integer community id for each node,
        or None when the dataset has no community metadata.

    Raises
    ------
    ValueError
        If dataset_name is not recognised.
    FileNotFoundError
        If the expected data file does not exist.
    """
    loaders: dict[str, callable] = {
        "primaryschool": _load_primaryschool,
        "hospital":      _load_hospital,
        "hs2011":        _load_hs2011,
        "hs2012":        _load_hs2012,
        "hs2013":        _load_hs2013,
        "conference":    _load_conference,
        "workplace":     _load_workplace,
        "malawi":        _load_malawi,
    }

    if dataset_name not in loaders:
        supported = ", ".join(f"'{k}'" for k in loaders)
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Supported names are: {supported}."
        )

    contacts, labels = loaders[dataset_name]()

    # Remap node ids to 0..n-1 and apply the same mapping to labels
    contacts, labels = _remap_nodes(contacts, labels)

    # Sort by time so later algorithms can assume temporal order
    contacts = contacts.sort_values("t", kind="stable").reset_index(drop=True)

    return contacts, labels


def describe_dataset(contacts: pd.DataFrame, labels: np.ndarray | None) -> None:
    """Print a human-readable summary of a loaded dataset.

    Parameters
    ----------
    contacts:
        DataFrame returned by load_contacts (columns t, i, j).
    labels:
        Label array returned by load_contacts, or None.
    """
    n = int(max(contacts["i"].max(), contacts["j"].max())) + 1
    n_contacts = len(contacts)
    t_min, t_max = contacts["t"].min(), contacts["t"].max()
    span_seconds = int(t_max - t_min)
    n_days = span_seconds / 86_400          # 86 400 s per day
    n_communities = (
        int(np.unique(labels).size) if labels is not None else 0
    )

    print("=" * 40)
    print(f"Nodes (n)          : {n}")
    print(f"Contact events     : {n_contacts}")
    print(f"Time span          : {span_seconds:,} s  (~{n_days:.2f} days)")
    print(f"t_min              : {t_min}")
    print(f"t_max              : {t_max}")
    print(f"Communities        : {n_communities if labels is not None else 'none'}")
    print("=" * 40)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_tij_space(filename: str) -> pd.DataFrame:
    """Read a whitespace-separated t i j file with no header."""
    path = DATA_DIR / filename
    _require_file(path)
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["t", "i", "j"],
                     usecols=[0, 1, 2], dtype={"t": int, "i": int, "j": int})
    return df


def _read_tij_tab_with_labels(filename: str) -> pd.DataFrame:
    """Read a tab-separated t i j class_i class_j file with no header."""
    path = DATA_DIR / filename
    _require_file(path)
    df = pd.read_csv(
        path, sep="\t", header=None,
        names=["t", "i", "j", "class_i", "class_j"],
        dtype={"t": int, "i": int, "j": int, "class_i": str, "class_j": str},
    )
    return df


def _require_file(path) -> None:
    """Raise FileNotFoundError with a clear message if path does not exist."""
    if not path.exists():
        raise FileNotFoundError(
            f"Expected data file not found: {path}\n"
            f"Please place the raw dataset file in {DATA_DIR}."
        )


def _labels_from_inline(df: pd.DataFrame) -> pd.Series:
    """Build a per-node label Series from inline class_i / class_j columns.

    Assumes the DataFrame has columns i, j, class_i, class_j.
    Both sides of each edge are used; if a node appears with different labels
    in different rows the first occurrence wins (datasets are consistent).
    """
    left  = df[["i", "class_i"]].rename(columns={"i": "id", "class_i": "class"})
    right = df[["j", "class_j"]].rename(columns={"j": "id", "class_j": "class"})
    mapping = (
        pd.concat([left, right])
        .drop_duplicates(subset="id", keep="first")
        .set_index("id")["class"]
    )
    return mapping


def _remap_nodes(
    contacts: pd.DataFrame,
    labels_input: pd.Series | np.ndarray | None,
) -> tuple[pd.DataFrame, np.ndarray | None]:
    """Remap all node ids to contiguous integers 0 .. n-1.

    Parameters
    ----------
    contacts:
        DataFrame with int columns i, j (original ids).
    labels_input:
        Either a pd.Series keyed by original id (from inline labels or metadata),
        a numpy array already indexed 0..n-1, or None.

    Returns
    -------
    contacts:
        Same DataFrame with i, j replaced by new ids.
    labels:
        numpy int array of length n, or None.
    """
    # Collect all unique node ids that appear in the contact list
    all_ids = np.union1d(contacts["i"].unique(), contacts["j"].unique())
    all_ids = np.sort(all_ids)
    old_to_new: dict[int, int] = {old: new for new, old in enumerate(all_ids)}
    n = len(all_ids)

    contacts = contacts.copy()
    contacts["i"] = contacts["i"].map(old_to_new)
    contacts["j"] = contacts["j"].map(old_to_new)

    if labels_input is None:
        return contacts, None

    if isinstance(labels_input, np.ndarray):
        # Already a dense array aligned to remapped ids; just return it
        return contacts, labels_input

    # labels_input is a pd.Series: index = original node id, values = class str
    # Encode class strings to integers
    unique_classes = sorted(labels_input.unique())
    class_to_int = {c: idx for idx, c in enumerate(unique_classes)}

    labels = np.full(n, fill_value=-1, dtype=int)
    for old_id, new_id in old_to_new.items():
        if old_id in labels_input.index:
            labels[new_id] = class_to_int[labels_input[old_id]]

    return contacts, labels


# ---------------------------------------------------------------------------
# Dataset-specific loaders
# ---------------------------------------------------------------------------

def _load_primaryschool() -> tuple[pd.DataFrame, pd.Series]:
    """Load primaryschool.csv (comma-sep, no header: t,i,j,class_i,class_j)."""
    path = DATA_DIR / "primaryschool.csv"
    _require_file(path)
    df = pd.read_csv(
        path, sep=",", header=None,
        names=["t", "i", "j", "class_i", "class_j"],
        dtype={"t": int, "i": int, "j": int, "class_i": str, "class_j": str},
    )
    labels = _labels_from_inline(df)
    contacts = df[["t", "i", "j"]]
    return contacts, labels


def _load_hospital() -> tuple[pd.DataFrame, pd.Series]:
    """Load hospital.csv (tab-sep, no header: t,i,j,role_i,role_j)."""
    df = _read_tij_tab_with_labels("hospital.csv")
    labels = _labels_from_inline(df)
    contacts = df[["t", "i", "j"]]
    return contacts, labels


def _load_hs2011() -> tuple[pd.DataFrame, pd.Series]:
    """Load hs2011.csv (tab-sep, no header: t,i,j,class_i,class_j)."""
    df = _read_tij_tab_with_labels("hs2011.csv")
    labels = _labels_from_inline(df)
    contacts = df[["t", "i", "j"]]
    return contacts, labels


def _load_hs2012() -> tuple[pd.DataFrame, pd.Series]:
    """Load hs2012.csv (tab-sep, no header: t,i,j,class_i,class_j)."""
    df = _read_tij_tab_with_labels("hs2012.csv")
    labels = _labels_from_inline(df)
    contacts = df[["t", "i", "j"]]
    return contacts, labels


def _load_hs2013() -> tuple[pd.DataFrame, pd.Series]:
    """Load hs2013.csv (space-sep, no header: t i j class_i class_j)."""
    path = DATA_DIR / "hs2013.csv"
    _require_file(path)
    df = pd.read_csv(
        path, sep=" ", header=None,
        names=["t", "i", "j", "class_i", "class_j"],
        dtype={"t": int, "i": int, "j": int, "class_i": str, "class_j": str},
    )
    labels = _labels_from_inline(df)
    contacts = df[["t", "i", "j"]]
    return contacts, labels


def _load_conference() -> tuple[pd.DataFrame, None]:
    """Load SFHH_tij.csv (space-sep, no header: t i j). No community labels."""
    contacts = _read_tij_space("SFHH_tij.csv")
    return contacts, None


def _load_workplace() -> tuple[pd.DataFrame, pd.Series]:
    """Load workplace_tij.csv (space-sep) + workplace_meta.csv (comma: id,dept)."""
    contacts = _read_tij_space("workplace_tij.csv")

    meta_path = DATA_DIR / "workplace_meta.csv"
    _require_file(meta_path)
    meta = pd.read_csv(
        meta_path, sep=",", header=None,
        names=["id", "dept"],
        dtype={"id": int, "dept": str},
    )
    labels_series = meta.set_index("id")["dept"]
    return contacts, labels_series


def _load_malawi() -> tuple[pd.DataFrame, None]:
    """Load malawi_tdij.csv (CSV with leading index col: contact_time,day,id1,id2).

    No community labels are available for this dataset.
    """
    path = DATA_DIR / "malawi_tdij.csv"
    _require_file(path)
    df = pd.read_csv(
        path, sep=",",
        index_col=0,      # drop the leading unnamed integer index column
        dtype={"contact_time": int, "day": int, "id1": int, "id2": int},
    )
    contacts = df.rename(columns={"contact_time": "t", "id1": "i", "id2": "j"})[
        ["t", "i", "j"]
    ]
    return contacts, None
