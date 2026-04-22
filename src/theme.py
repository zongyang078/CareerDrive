"""
CareerDrive Theme — Centralized color definitions.

Burgundy / Rose palette matching the CareerDrive.pdf slide deck.
"""

# Core palette
COLORS = {
    'primary':    '#8B4049',   # Burgundy — main accent
    'secondary':  '#C4918A',   # Rose — secondary
    'tertiary':   '#E8C8C3',   # Light rose — tertiary
    'dark':       '#4A2028',   # Dark burgundy (title slide bg)
    'text':       '#2D2D2D',   # Dark text
    'background': '#FFFFFF',   # White content bg
    'surface':    '#FDF5F3',   # Warm off-white
    'highlight':  '#9B5B5E',   # Muted rose accent
}

# Cluster color mapping (consistent across charts and Streamlit)
CLUSTER_COLORS = {
    'Management/Engineering': '#8B4049',
    'Skilled Trades':         '#C4918A',
    'Entry Level/Operators':  '#E8C8C3',
}

# Three-color sequential palette
PALETTE_3 = ['#8B4049', '#C4918A', '#E8C8C3']

# Extended palette for > 3 categories
PALETTE_EXT = ['#8B4049', '#C4918A', '#E8C8C3', '#9B5B5E', '#D4A59E', '#6B2D35']

# Matplotlib colormaps (custom)
import matplotlib.colors as mcolors

CMAP_SEQUENTIAL = mcolors.LinearSegmentedColormap.from_list(
    'careerdrive', ['#FDF5F3', '#E8C8C3', '#C4918A', '#8B4049', '#4A2028']
)

CMAP_DIVERGING = mcolors.LinearSegmentedColormap.from_list(
    'careerdrive_div', ['#4A2028', '#8B4049', '#C4918A', '#E8C8C3', '#FDF5F3']
)

# Community / network extra palette
COMM_COLORS = ['#8B4049', '#C4918A', '#9B5B5E', '#D4A59E', '#6B2D35']

# Text cluster colors
TEXT_CLUSTER_COLORS = ['#8B4049', '#9B5B5E', '#C4918A']

# ── Font ──────────────────────────────────────────────────────────────────────
import os as _os
import matplotlib.font_manager as _fm
import matplotlib as _mpl

_font_path = _os.path.normpath(_os.path.join(_os.path.dirname(__file__), '..', 'Quicksand.ttf'))
if _os.path.exists(_font_path):
    _fm.fontManager.addfont(_font_path)


def apply_font():
    """Set Quicksand as the matplotlib font family. Call after sns.set_theme()."""
    _mpl.rcParams['font.family'] = 'Quicksand'
    _mpl.rcParams['font.sans-serif'] = ['Quicksand'] + list(_mpl.rcParams.get('font.sans-serif', []))
